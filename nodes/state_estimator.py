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
from fav_state_estimation.msg import StateVector2x3D
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

      self.rho = 2.5
      self.phi = 0.3
      self.tau = 0.1

      self.velocity = np.array([0.0, 0.0, 0.0])
      self.x_prev = np.array([0.0, 0.0, -0.5])
      self.x1hat_prev = np.array([0.0, 0.0, -0.5])
      self.x2hat_prev = np.array([0.0, 0.0, 0.0])
      self.prev_smo_time = None

      self.tag_coordinates = [np.array([0.5, 3.35, -0.5]),
                              np.array([1.1, 3.35, -0.5]),
                              np.array([0.5, 3.35, -0.9]),
                              np.array([1.1, 3.35, -0.9])]

      self.range_sensor_position_rel = np.matrix([[0.2], [0], [0.1], [1]])
      self.range_sensor_position_abs = self.range_sensor_position_rel
      self.rot_matrix = tf.transformations.identity_matrix

      self.time_motion = 0.0
      self.mu = np.matrix([[0.8],
                           [1.675],
                           [-0.7],
                           [0.0],
                           [0.0],
                           [0.0]])
      self.mu_prev = self.mu
      self.me_prior = self.mu

      self.sigma_reset = np.matrix(np.diag([5.0, 5.0, 2.0, 1.0, 1.0, 1.0]))
      self.sigma = self.sigma_reset
      self.sigma_prior = self.sigma
      self.R = np.matrix(np.diag([0.1, 0.1, 0.1, 0.1, 0.1, 0.1]))
      self.Q_press = 0.0001
      self.Q_range_0 = 0.0025
      self.Q_range_lin_fac = 0.001
            
      rospy.init_node("state_estimator")
      self.state_pub = rospy.Publisher("estimated_state", StateVector2x3D, queue_size=1)
      
      rospy.sleep(5)

      self.pressure_sub = rospy.Subscriber("pressure", FluidPressure, self.on_pressure, queue_size=1)
      self.imu_sub = rospy.Subscriber("mavros/imu/data", Imu, self.on_imu, queue_size=1)
      self.range_sub = rospy.Subscriber("ranges", RangeMeasurementArray, self.on_range, queue_size=1)

      self.server = Server(StateEstimationConfig, self.server_callback)

      self.last_sensor_time = rospy.get_time()

   def run(self):
      rate = rospy.Rate(50.0)

      while not rospy.is_shutdown():
         if self.prev_smo_time is None:
            self.prev_smo_time = rospy.get_time()
         else:
            with self.data_lock:
               if (rospy.get_time() - self.last_sensor_time) > 0.05:
                  self.sigma = 1.01 * self.sigma
                  rospy.logwarn_throttle(5.0, 'No Measurements received!')
               else:
                  self.velocity = self.smo(self.mu[:3, 0])
                  self.publish_state()

         rate.sleep()
   
   def publish_state(self):
      with self.data_lock:
         msg = StateVector2x3D()
         msg.header.stamp = rospy.Time.now()
         msg.position.x = self.mu[0, 0]
         msg.position.y = self.mu[1, 0]
         msg.position.z = self.mu[2, 0]
         msg.velocity.x = self.velocity[0]
         msg.velocity.y = self.velocity[1]
         msg.velocity.z = self.velocity[2]
         self.state_pub.publish(msg)

   def server_callback(self, config, level):
      with self.data_lock:
         rospy.loginfo("New Parameters received by State Estimator")
         
         if config.calibrate_surface_pressure:
            self.surface_pressure = self.current_pressure
            config.calibrate_surface_pressure = False
         
         if config.reset_sigma:
            self.sigma = self.sigma_reset
            config.reset_sigma = False

         self.rho = config.rho
         self.phi = config.phi
         self.tau = config.tau

         self.R = np.matrix(np.diag([config.Rx, config.Ry, config.Rz, config.Rdx, config.Rdy, config.Rdz]))
         self.Q_press = config.Q_press
         self.Q_range_0 = config.Q_range_0
         self.Q_range_lin_fac = config.Q_range_lin_fac

      return config

   def on_imu(self, msg):
      with self.data_lock:
         self.last_sensor_time = msg.header.stamp.to_sec()
         quaternian = [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w]
         self.rot_matrix = tf.transformations.quaternion_matrix(quaternian)
         self.range_sensor_position_abs = np.matmul(self.rot_matrix, self.range_sensor_position_rel)
         # rospy.loginfo_throttle(1.0, 'pos_range_sensor: ' + str(self.range_sensor_position_abs))
         if self.time_motion is None:
            self.time_motion = msg.header.stamp.to_sec()
         else:
            del_t = msg.header.stamp.to_sec() - self.time_motion
            self.time_motion = msg.header.stamp.to_sec()
            if abs(msg.linear_acceleration.z) > 9:
               rospy.logwarn_throttle(5.0, "Acceleration measurement probably incorrect.\nMeasured z-acceleration: " + str(msg.linear_acceleration.z))
            acc = np.array([msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z])
            acc_cov = msg.linear_acceleration_covariance
            self.ekf(mode=0, del_t=del_t, acc=acc, acc_cov=acc_cov)

   def on_range(self, msg):
      with self.data_lock:
         self.last_sensor_time = msg.header.stamp.to_sec()
         for meas in msg.measurements:
            z = meas.range
            self.ekf(mode=2, z=meas.range, tag_id=meas.id)

   def on_pressure(self, msg):
      with self.data_lock:
         self.last_sensor_time = msg.header.stamp.to_sec()
         if self.surface_pressure is None:
            self.surface_pressure = msg.fluid_pressure
         time = msg.header.stamp.to_sec()
         self.current_pressure = msg.fluid_pressure
         depth = - (msg.fluid_pressure - self.surface_pressure) / self.pascal_per_meter
         self.ekf(mode=1, z=depth)

   # mode 0: imu, mode 1: pressure, mode 2: range
   def ekf(self, mode=0, z=None, tag_id=None, del_t=None, acc=None, acc_cov=None):
      with self.data_lock:
         # rospy.loginfo('\n\nmode = ' + str(mode) + '\nz = ' + str(z) + '\nid = ' + str(tag_id) + '\nacc = ' + str(acc) + '\nmu = ' + str(self.mu))
         if mode == 0:
            self.mu_prior = self.mu
            for i in range(3):
               self.mu[i, 0] = self.mu_prev[i, 0] + del_t*self.mu_prior[i+3, 0]
            self.mu[3, 0] = self.mu_prev[3, 0] + del_t*acc[0]
            self.mu[4, 0] = self.mu_prev[4, 0] + del_t*acc[1]
            self.mu[5, 0] = self.mu_prev[5, 0] + del_t*acc[2]
            # rospy.loginfo('\nmu_prev = ' + str(self.mu_prev))
            self.mu_prev = self.mu_prior
            G = np.matrix([[1, 0, 0, del_t, 0, 0],
                           [0, 1, 0, 0, del_t, 0],
                           [0, 0, 1, 0, 0, del_t],
                           [0, 0, 0, 1, 0, 0],
                           [0, 0, 0, 0, 1, 0],
                           [0, 0, 0, 0, 0, 1]])
            R = self.R+np.matrix(np.diag([0, 0, 0, acc_cov[0], acc_cov[4], acc_cov[8]]))
            self.sigma = np.matmul(np.matmul(G, self.sigma), G.T) + R
         else:
            self.sigma_prior = self.sigma
            self.mu_prior = self.mu
            if mode == 1:
               H = np.matrix([[0, 0, 1, 0, 0, 0]])
               h = self.mu_prior[2]
               Q = self.Q_press
            elif mode == 2:
               h = np.sqrt(np.square(self.mu_prior[0, 0]+self.range_sensor_position_abs[0, 0]-self.tag_coordinates[tag_id-1][0])+
                        np.square(self.mu_prior[1, 0]+self.range_sensor_position_abs[1, 0]-self.tag_coordinates[tag_id-1][1])+
                        np.square(self.mu_prior[2, 0]+self.range_sensor_position_abs[2, 0]-self.tag_coordinates[tag_id-1][2]))
               # rospy.loginfo('\nh = ' + str(h) + '\nz = ' + str(z))
               H = (1/h) * np.matrix([[self.mu_prior[0, 0]-self.tag_coordinates[tag_id-1][0],
                                       self.mu_prior[1, 0]-self.tag_coordinates[tag_id-1][1],
                                       self.mu_prior[2, 0]-self.tag_coordinates[tag_id-1][2],
                                       0.0,
                                       0.0,
                                       0.0]])
               if (rospy.get_time() - self.time_motion) < 0.1:
                  Q = np.matrix([[self.Q_range_0+self.Q_range_lin_fac*z]])
               else:
                  rospy.logwarn_throttle(5.0, 'No Orientation Update received!')
                  Q = np.matrix([[self.Q_range_0+np.square(2*np.linalg.norm(self.range_sensor_position_rel))+self.Q_range_lin_fac*z]])
            # rospy.loginfo('\nh= ' + str(h) + '\nH = ' + str(H) + '\nQ = ' + str(Q) + '\nsigma = ' + str(self.sigma_prior))
            K = np.matmul(np.matmul(self.sigma_prior, H.T), np.linalg.inv(np.matrix(np.matmul(np.matmul(H, self.sigma_prior), H.T) + Q, dtype=np.float64)))
            # rospy.loginfo('\nh= ' + str(h) + '\nH = ' + str(H) + '\nQ = ' + str(Q) + '\nK = ' + str(K))
            self.mu = self.mu_prior + np.matmul(K, np.matrix(z-h))
            self.sigma = np.matmul((np.eye(6) - np.matmul(K, H)), self.sigma_prior)
         # rospy.loginfo('\nmu = ' + str(self.mu))
         # rospy.loginfo('\nsigma = ' + str(self.sigma))

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
