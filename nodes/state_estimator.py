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

      tag_system_origin = np.array([0.5, 3.35, -0.5])
      self.calculate_tag_coordinates(tag_system_origin)

      self.range_sensor_position_rel = np.array([[0.2], [0], [0.1]])
      self.range_sensor_position_abs = self.range_sensor_position_rel.copy()
      self.rot_matrix = tf.transformations.identity_matrix()[:3, :3]

      self.orientation = None
      self.angular_velocity = None

      self.mu_reset = np.array([[0.8], [1.675], [-0.7], [0.0], [0.0], [0.0]])
      self.mu = self.mu_reset.copy()

      self.sigma_reset = np.eye(6)
      self.sigma = self.sigma_reset.copy()
      self.R = np.array(np.diag([1e-4, 1e-4, 1e-2, 1e-6, 1e-6, 1e-4]))
      self.Q_press = 0.0001
      self.Q_range_0 = 0.1
      self.Q_range_lin_fac = 0.01
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
         msg.twist.twist.linear.x = self.mu[3, 0]
         msg.twist.twist.linear.y = self.mu[4, 0]
         msg.twist.twist.linear.z = self.mu[5, 0]
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

         if config.reset_mu:
            self.mu = self.mu_reset.copy()
            config.reset_mu = False

         self.R = np.diag([config.Rx, config.Ry, config.Rz, config.Rdx, config.Rdy, config.Rdz])
         self.Q_press = config.Q_press
         self.Q_range_0 = config.Q_range_0
         self.Q_range_lin_fac = config.Q_range_lin_fac

         tag_system_origin = np.array([config.groups.groups.tag_system.parameters.tag_1_x, config.groups.groups.tag_system.parameters.tag_1_y, config.groups.groups.tag_system.parameters.tag_1_z])
         tag_system_orientation = config.groups.groups.tag_system.parameters.orientation
         self.calculate_tag_coordinates(tag_system_origin, tag_system_orientation)

         self.c_scaling = config.scaling_variable

      return config

   def calculate_tag_coordinates(self, origin, orientation=0):
      orientation = orientation * np.pi / 180
      self.tag_coordinates = [np.array([origin[0], origin[1], origin[2]]),
                              np.array([origin[0]+0.6*np.cos(orientation), origin[1], origin[2]+0.6*np.sin(orientation)]),
                              np.array([origin[0]+0.4*np.sin(orientation), origin[1], origin[2]-0.4*np.cos(orientation)]),
                              np.array([origin[0]+0.6*np.cos(orientation)+0.4*np.sin(orientation), origin[1], origin[2]+0.6*np.sin(orientation)-0.4*np.cos(orientation)])]


   def on_imu(self, msg):
      with self.data_lock:
         self.time_imu = msg.header.stamp.to_sec()
         del_t = self.time_imu - self.last_sensor_time
         self.last_sensor_time = self.time_imu
         self.orientation = msg.orientation
         quaternion = [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w]
         self.rot_matrix = tf.transformations.quaternion_matrix(quaternion)[:3, :3]
         self.range_sensor_position_abs = np.matmul(self.rot_matrix, self.range_sensor_position_rel)
         self.angular_velocity = np.matmul(self.rot_matrix, np.array([[msg.angular_velocity.x], [msg.angular_velocity.y], [msg.angular_velocity.z]]))
         # rospy.loginfo_throttle(1.0, 'pos_range_sensor: ' + str(self.range_sensor_position_abs))
         if (msg.linear_acceleration.x == 0) and (msg.linear_acceleration.y == 0) and (msg.linear_acceleration.z == 0):
            rospy.logerr_once("Zero Acceleration measured! Ignoring Measurement.")
            acc = np.zeros([3, 1])
         else:
            acc = np.array([[msg.linear_acceleration.x], [msg.linear_acceleration.y], [msg.linear_acceleration.z-9.81]])
         self.ekf_predict(del_t=del_t, acc_loc=acc)

   def on_range(self, msg):
      with self.data_lock:
         self.last_sensor_time = msg.header.stamp.to_sec()
         self.time_range = self.last_sensor_time
         if len(msg.measurements) == 0:
            rospy.logwarn_throttle(1.0, 'No Tags detected!')
            return
         elif len(msg.measurements) == 1:
            rospy.logwarn_throttle(1.0, 'Only one Tag detected!')
         z = np.empty([0, 1])
         tag_ids = []
         if self.time_imu > 0:
            for meas in msg.measurements:
               z = np.append(z, np.array([[meas.range]]), axis=0)
               tag_ids.append(meas.id)
            self.ekf_correct(mode=1, z=z, tag_ids=tag_ids)

   def on_pressure(self, msg):
      with self.data_lock:
         self.last_sensor_time = msg.header.stamp.to_sec()
         self.time_pressure = self.last_sensor_time
         if self.surface_pressure is None:
            self.surface_pressure = msg.fluid_pressure
         self.current_pressure = msg.fluid_pressure
         depth = - (msg.fluid_pressure - self.surface_pressure) / self.pascal_per_meter
         self.ekf_correct(mode=0, z=np.array([[depth]]))

   def ekf_predict(self, del_t, acc_loc):
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
         self.sigma = np.matmul(np.matmul(G, sigma_prior), G.T) + self.R
         mu_ok = self.check_boundaries()
         if not mu_ok:
            rospy.logwarn('\n\nmode = predict\nmu_prior = ' + str(mu_prior) + '\nacc = ' + str(acc_glob) + '\nmu = ' + str(self.mu) + '\nsigma = ' + str(self.sigma))

   # mode 0: pressure, mode 1: range
   def ekf_correct(self, mode=0, z=None, tag_ids=None):
      with self.data_lock:
         # rospy.loginfo('\n\nmode = ' + str(mode) + '\nz = ' + str(z) + '\nid = ' + str(tag_id) + '\nacc = ' + str(acc) + '\nmu = ' + str(self.mu))
         sigma_prior = self.sigma.copy()
         mu_prior = self.mu.copy()
         if mode == 0:
            H = np.array([[0, 0, 1, 0, 0, 0]])
            h = np.array([[mu_prior[2, 0]]])
            Q = self.Q_press
         elif mode == 1:
            h = np.zeros([len(tag_ids), 1])
            H = np.zeros([len(tag_ids), 6])
            Q = np.zeros([len(tag_ids), len(tag_ids)])
            for i in range(len(tag_ids)):
               dx = mu_prior[0, 0]+self.range_sensor_position_abs[0, 0]-self.tag_coordinates[tag_ids[i]-1][0]
               dy = mu_prior[1, 0]+self.range_sensor_position_abs[1, 0]-self.tag_coordinates[tag_ids[i]-1][1]
               dz = mu_prior[2, 0]+self.range_sensor_position_abs[2, 0]-self.tag_coordinates[tag_ids[i]-1][2]
               h[i, 0] = np.sqrt(dx**2+dy**2+dz**2)
               # rospy.loginfo('\nid = ' + str(tag_ids[i]) + '\n' + str(tag_ids[i]-1) + '\n' + str(self.tag_coordinates[tag_ids[i]-1]) + '\nh = ' + str(h[i, 0]) + '\nz = ' + str(z[i, 0]))
               H[i, :] = (1/h[i, 0]) * np.array([dx, dy, dz, 0, 0, 0])
               Q[i, i] = self.Q_range_0+self.Q_range_lin_fac*z[i, 0]
         # rospy.loginfo_once(str(self.tag_coordinates) + str(self.tag_coordinates[0]) + str(self.tag_coordinates[1]) + str(self.tag_coordinates[2]) + str(self.tag_coordinates[3]))
         # rospy.loginfo('\ntag_ids: ' + str(tag_ids) + '\nh= ' + str(h) + '\nH = ' + str(H) + '\nQ = ' + str(Q) + '\nsigma = ' + str(sigma_prior))
         K = np.linalg.multi_dot([sigma_prior, H.T, np.linalg.inv(np.linalg.multi_dot([H, sigma_prior, H.T]) + Q)])
         # rospy.loginfo('\nh = ' + str(h) + '\nH = ' + str(H) + '\nQ = ' + str(Q) + '\nK = ' + str(K))
         if z.shape[0] == 1:
            self.mu = mu_prior + np.matmul(K, (z-h))
         else:
            self.mu = mu_prior + np.matmul(K, (z-h))
         self.sigma = np.matmul((np.eye(6) - np.matmul(K, H)), sigma_prior)
         # rospy.loginfo('\nmu = ' + str(self.mu))
         # rospy.loginfo('\nsigma = ' + str(self.sigma))
         # for i in range(3):
         #   if abs(self.mu[i, 0] - mu_prior[i, 0]) > 0.2:
         # if self.mu[0, 0] < 0.5:
         #   if mode == 1:
         #      rospy.loginfo('\n\nmode = ' + str(mode) + '\nid = ' + str(tag_id) + '\nmu_prior = ' + str(mu_prior) + '\ndx = ' + str(dx) + '\ndy = ' + str(dy) + '\ndz = ' + str(dz) + '\nh = ' + str(h) + '\nH = ' + str(H) + '\nz = ' + str(z) + '\nK = ' + str(K) + '\nmu = ' + str(self.mu) + '\nsigma_prior = ' + str(sigma_prior) + '\nsigma = ' + str(self.sigma))
         #   else:
         #      rospy.loginfo('\n\nmode = ' + str(mode) + '\nid = ' + str(tag_id) + '\nmu_prior = ' + str(mu_prior) + '\nh = ' + str(h) + '\nH = ' + str(H) + '\nz = ' + str(z) + '\nK = ' + str(K) + '\nmu = ' + str(self.mu) + '\nsigma_prior = ' + str(sigma_prior) + '\nsigma = ' + str(self.sigma))
         mu_ok = self.check_boundaries()
         if not mu_ok:
            rospy.logwarn('\n\nmode = ' + str(mode) + '\ntag_ids = ' + str(tag_ids) + '\nmu_prior = ' + str(mu_prior) + '\nh = ' + str(h) + '\nz = ' + str(z) + '\nmu = ' + str(self.mu) + '\nsigma = ' + str(self.sigma))

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

   def sat(self, x):
      return min(1.0, max(-1.0, x))


def main():
   node = StateEstimatorNode()
   node.run()


if __name__ == "__main__":
   main()
