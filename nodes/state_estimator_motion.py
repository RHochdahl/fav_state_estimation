#!/usr/bin/env python

PACKAGE = 'fav_state_estimation'
import roslib;roslib.load_manifest(PACKAGE)
import rospy
import numpy as np

from dynamic_reconfigure.server import Server
from fav_state_estimation.cfg import StateEstimationConfig

import tf

import threading
from sensor_msgs.msg import FluidPressure
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from range_sensor.msg import RangeMeasurementArray


class StateEstimatorNode():
   def __init__(self):
      self.data_lock = threading.RLock()

      self.simulate = rospy.get_param("simulate")

      self.pascal_per_meter = 9.78057e3  # g*rho
      if self.simulate:
         self.surface_pressure = 1.01325e5  # according to gazebo
      else:
         self.surface_pressure = None
      self.current_pressure = self.surface_pressure

      self.boundaries = np.array([[0, 1.6],
                                 [0, 3.35],
                                 [-1.4, 0]])

      self.rho = 2.5
      self.phi = 0.3
      self.tau = 0.1

      self.linear_velocity = np.array([0.0, 0.0, 0.0])
      self.x_prev = np.array([0.0, 0.0, -0.5])
      self.x1hat_prev = np.array([0.0, 0.0, -0.5])
      self.x2hat_prev = np.array([0.0, 0.0, 0.0])
      self.prev_smo_time = None

      self.tag_coordinates = [np.array([0.5, 3.35, -0.5]),
                              np.array([1.1, 3.35, -0.5]),
                              np.array([0.5, 3.35, -0.9]),
                              np.array([1.1, 3.35, -0.9])]

      self.range_sensor_position_rel = np.array([[0.2], [0], [0.1]])
      self.range_sensor_position_abs = self.range_sensor_position_rel.copy()
      self.rot_matrix = tf.transformations.identity_matrix()[:3, :3]

      self.orientation = None
      self.angular_velocity = None

      self.mu_reset = np.array([[0.8], [1.675], [-0.7], [0.0], [0.0], [0.0]])
      self.mu = self.mu_reset.copy()

      self.sigma_reset = np.diag([0.01, 0.01, 0.01, 0.0001, 0.0001, 0.0001])
      self.sigma = self.sigma_reset.copy()
      self.R = np.array(np.diag([0.1, 0.1, 0.1, 0.01, 0.01, 0.01]))
      self.Q_press = 0.0001
      self.Q_range_0 = 0.01
      self.Q_range_lin_fac = 0.001
      self.c_scaling = 1e-9
            
      rospy.init_node("state_estimator")
      self.state_pub = rospy.Publisher("estimated_state", Odometry, queue_size=1)
      
      rospy.sleep(5)

      self.time_pressure = 0
      self.time_range = 0
      self.time_imu = 0
      self.last_sensor_time = 0

      self.pressure_sub = rospy.Subscriber("pressure", FluidPressure, self.on_pressure, queue_size=1)
      self.imu_sub = rospy.Subscriber("mavros/imu/data", Imu, self.on_imu, queue_size=1)
      self.range_sub = rospy.Subscriber("ranges", RangeMeasurementArray, self.on_range, queue_size=1)

      self.server = Server(StateEstimationConfig, self.server_callback)

   def run(self):
      rate = rospy.Rate(50.0)

      while not rospy.is_shutdown():
         if self.prev_smo_time is None:
            self.prev_smo_time = rospy.get_time()
         else:
            time = rospy.get_time()
            if (time - self.last_sensor_time) > 0.05:
               with self.data_lock:
                  sigma_prior = self.sigma.copy()
                  self.sigma = (1+self.c_scaling) * sigma_prior
               rospy.logwarn_throttle(5.0, 'No Measurements received!')
            else:
               if time - self.time_imu > 0.05:
                  rospy.logwarn_throttle(5.0, 'No IMU-Measurements received!')
               if time - self.time_range > 0.4:
                  rospy.logwarn_throttle(5.0, 'No Range Measurements received!')
               if time - self.time_pressure > 0.05:
                  rospy.logwarn_throttle(5.0, 'No Pressure Measurements received!')
               # self.linear_velocity = self.smo(self.mu[:3, 0])
               if self.time_imu > 0:
                  self.publish_state()

         rate.sleep()
   
   def publish_state(self):
      with self.data_lock:
         msg = Odometry()
         msg.header.stamp = rospy.Time.now()
         msg.header.frame_id = 'map'
         msg.child_frame_id = 'map'
         msg.pose.pose.position.x = self.mu[0, 0]
         msg.pose.pose.position.y = self.mu[1, 0]
         msg.pose.pose.position.z = self.mu[2, 0]
         msg.twist.twist.linear.x = self.linear_velocity[0]
         msg.twist.twist.linear.y = self.linear_velocity[1]
         msg.twist.twist.linear.z = self.linear_velocity[2]
         if self.angular_velocity is not None:
            msg.twist.twist.angular.x = self.angular_velocity[0]
            msg.twist.twist.angular.y = self.angular_velocity[1]
            msg.twist.twist.angular.z = self.angular_velocity[2]
         if self.orientation is not None:
            msg.pose.pose.orientation = self.orientation
         self.state_pub.publish(msg)

   def server_callback(self, config, level):
      with self.data_lock:
         rospy.loginfo("New Parameters received by State Estimator")
         
         if config.calibrate_surface_pressure:
            self.surface_pressure = self.current_pressure
            config.calibrate_surface_pressure = False
         
         if config.reset_sigma:
            self.sigma = self.sigma_reset.copy()
            config.reset_sigma = False

         if config.reset_lin_vel:
            for i in range(3):
               self.mu[i+3, 0] = 0.0
            config.reset_lin_vel = False

            self.mu = self.mu_reset.copy()

         self.rho = config.rho
         self.phi = config.phi
         self.tau = config.tau

         self.R = np.matrix(np.diag([config.Rx, config.Ry, config.Rz, config.Rdx, config.Rdy, config.Rdz]))
         self.Q_press = config.Q_press
         self.Q_range_0 = config.Q_range_0
         self.Q_range_lin_fac = config.Q_range_lin_fac

         self.tag_coordinates = [np.array([config.groups.groups.tags.groups.tag_1.parameters.x, config.groups.groups.tags.groups.tag_1.parameters.y, config.groups.groups.tags.groups.tag_1.parameters.z]),
                                 np.array([config.groups.groups.tags.groups.tag_2.parameters.x, config.groups.groups.tags.groups.tag_2.parameters.y, config.groups.groups.tags.groups.tag_2.parameters.z]),
                                 np.array([config.groups.groups.tags.groups.tag_3.parameters.x, config.groups.groups.tags.groups.tag_3.parameters.y, config.groups.groups.tags.groups.tag_3.parameters.z]),
                                 np.array([config.groups.groups.tags.groups.tag_4.parameters.x, config.groups.groups.tags.groups.tag_4.parameters.y, config.groups.groups.tags.groups.tag_4.parameters.z])]

         self.c_scaling = config.scaling_variable

      return config

   def on_imu(self, msg):
      with self.data_lock:
         del_t = msg.header.stamp.to_sec() - self.last_sensor_time
         self.last_sensor_time = msg.header.stamp.to_sec()
         self.time_imu = self.last_sensor_time
         self.orientation = msg.orientation
         quaternion = [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w]
         self.rot_matrix = tf.transformations.quaternion_matrix(quaternion)[:3, :3]
         self.range_sensor_position_abs = np.matmul(self.rot_matrix, self.range_sensor_position_rel)
         self.angular_velocity = np.matmul(self.rot_matrix, np.array([[msg.angular_velocity.x], [msg.angular_velocity.y], [msg.angular_velocity.z]]))
         # rospy.loginfo_throttle(1.0, 'pos_range_sensor: ' + str(self.range_sensor_position_abs))
         if abs(msg.linear_acceleration.z) > 8:
            rospy.logwarn_throttle(5.0, "Acceleration measurement probably incorrect.\nMeasured z-acceleration: " + str(msg.linear_acceleration.z))
            acc = np.matrix([[msg.linear_acceleration.x], [msg.linear_acceleration.y], [msg.linear_acceleration.z+9.81]])
         else:
            acc = np.array([[msg.linear_acceleration.x], [msg.linear_acceleration.y], [msg.linear_acceleration.z]])
         acc_cov = np.array([msg.linear_acceleration_covariance]).reshape([3, 3])
         self.ekf_predict(del_t=del_t, acc_loc=acc, acc_loc_cov=acc_cov)

   def on_range(self, msg):
      with self.data_lock:
         self.last_sensor_time = msg.header.stamp.to_sec()
         self.time_range = self.last_sensor_time
         if len(msg.measurements) == 0:
            rospy.logwarn_throttle(1.0, 'No Tags detected!')
         elif len(msg.measurements) == 1:
            rospy.logwarn_throttle(1.0, 'Only one Tag detected!')
         for meas in msg.measurements:
            z = meas.range
            self.ekf_correct(mode=1, z=meas.range, tag_id=meas.id)

   def on_pressure(self, msg):
      with self.data_lock:
         self.last_sensor_time = msg.header.stamp.to_sec()
         self.time_pressure = self.last_sensor_time
         if self.surface_pressure is None:
            self.surface_pressure = msg.fluid_pressure
         self.current_pressure = msg.fluid_pressure
         depth = - (msg.fluid_pressure - self.surface_pressure) / self.pascal_per_meter
         self.ekf_correct(mode=0, z=depth)

   def ekf_predict(self, del_t, acc_loc, acc_loc_cov):
      with self.data_lock:
         sigma_prior = self.sigma.copy()
         mu_prior = self.mu.copy()
         acc_glob = np.matmul(self.rot_matrix, acc_loc)
         for i in range(3):
            self.mu[i, 0] = mu_prior[i, 0] + del_t*mu_prior[i+3, 0]
         self.mu[3, 0] = mu_prior[3, 0] + del_t*acc_glob[0, 0]
         self.mu[4, 0] = mu_prior[4, 0] + del_t*acc_glob[1, 0]
         self.mu[5, 0] = mu_prior[5, 0] + del_t*acc_glob[2, 0]
         G = np.array([[1, 0, 0, del_t, 0, 0],
                        [0, 1, 0, 0, del_t, 0],
                        [0, 0, 1, 0, 0, del_t],
                        [0, 0, 0, 1, 0, 0],
                        [0, 0, 0, 0, 1, 0],
                        [0, 0, 0, 0, 0, 1]])
         del_R = np.zeros([6, 6])
         # del_R[3:6, 3:6] = np.matmul(np.matmul(self.rot_matrix, acc_loc_cov), self.rot_matrix.T)
         R = self.R + del_R
         self.sigma = np.matmul(np.matmul(G, sigma_prior), G.T) + R
         mu_ok = self.check_boundaries()
         if not mu_ok:
            rospy.logwarn('\n\nmode = predict\nmu_prior = ' + str(mu_prior) + '\nacc = ' + str(acc_glob) + '\nmu = ' + str(self.mu) + '\nsigma = ' + str(self.sigma))


   # mode 0: pressure, mode 1: range
   def ekf_correct(self, mode=0, z=None, tag_id=None):
      with self.data_lock:
         # rospy.loginfo('\n\nmode = ' + str(mode) + '\nz = ' + str(z) + '\nid = ' + str(tag_id) + '\nacc = ' + str(acc) + '\nmu = ' + str(self.mu))
         sigma_prior = self.sigma.copy()
         mu_prior = self.mu.copy()
         if mode == 0:
            H = np.array([[0, 0, 1, 0, 0, 0]])
            h = mu_prior[2, 0]
            Q = self.Q_press
         elif mode == 1:
            dx = mu_prior[0, 0]+self.range_sensor_position_abs[0, 0]-self.tag_coordinates[tag_id-1][0]
            dy = mu_prior[1, 0]+self.range_sensor_position_abs[1, 0]-self.tag_coordinates[tag_id-1][1]
            dz = mu_prior[2, 0]+self.range_sensor_position_abs[2, 0]-self.tag_coordinates[tag_id-1][2]
            h = np.sqrt(dx**2+dy**2+dz**2)
            # rospy.loginfo('\nid = ' + str(tag_id) + '\nh = ' + str(h) + '\nz = ' + str(z))
            H = (1/h) * np.array([[dx, dy, dz, 0, 0, 0]])
            Q = self.Q_range_0+self.Q_range_lin_fac*z
         # rospy.loginfo('\nh= ' + str(h) + '\nH = ' + str(H) + '\nQ = ' + str(Q) + '\nsigma = ' + str(self.sigma_prior))
         K = np.linalg.multi_dot([sigma_prior, H.T, np.linalg.inv(np.linalg.multi_dot([H, sigma_prior, H.T]) + Q)])
         # rospy.loginfo('\nh= ' + str(h) + '\nH = ' + str(H) + '\nQ = ' + str(Q) + '\nK = ' + str(K))
         self.mu = mu_prior + K * (z-h)
         self.sigma = np.matmul((np.eye(6) - np.matmul(K, H)), sigma_prior)
         # rospy.loginfo('\nmu = ' + str(self.mu))
         # rospy.loginfo('\nsigma = ' + str(self.sigma))
         mu_ok = self.check_boundaries()
         if not mu_ok:
            rospy.logwarn('\n\nmode = ' + str(mode) + '\nid = ' + str(tag_id) + '\nmu_prior = ' + str(mu_prior) + '\nh = ' + str(h) + '\nz = ' + str(z) + '\nmu = ' + str(self.mu) + '\nsigma = ' + str(self.sigma))


   def check_boundaries(self):
      within_boundaries = True
      mu_buf = self.mu.copy()
      for i in range(3):
         if self.mu[i, 0] < self.boundaries[i, 0]:
            self.mu[i, 0] = self.boundaries[i, 0]
            within_boundaries = False
         elif self.mu[i, 0] > self.boundaries[i, 1]:
            self.mu[i, 0] = self.boundaries[i, 1]
            within_boundaries = False
      if not within_boundaries:
         rospy.logwarn_throttle(1.0, 'Estimated state outside of boundaries!' + '\nmu = ' + str(mu_buf))
         return False
      else:
         return True      

   def smo(self, x):
      time = rospy.get_time()
      del_t = time - self.prev_smo_time
      self.prev_smo_time = time
      x1hat = np.zeros(3)
      x2hat = np.zeros(3)
      for i in range(3):
         x1hat[i] = self.x1hat_prev[i] + del_t*self.x2hat_prev[i]
         x2hat[i] = self.x2hat_prev[i] + (del_t/self.tau) * (-self.x2hat_prev[i]-self.rho*self.sat((self.x1hat_prev[i]-self.x_prev[i])/self.phi))
         self.x_prev[i] = x[i, 0]
         self.x1hat_prev[i] = x1hat[i]
         self.x2hat_prev[i] = x2hat[i]
      return x2hat

   def sat(self, x):
      return min(1.0, max(-1.0, x))


def main():
   node = StateEstimatorNode()
   node.run()


if __name__ == "__main__":
   main()
