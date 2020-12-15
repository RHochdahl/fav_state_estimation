#!/usr/bin/env python

PACKAGE = 'fav_state_estimation'
import roslib;roslib.load_manifest(PACKAGE)
import rospy
import numpy as np

from dynamic_reconfigure.server import Server
from fav_state_estimation.cfg import SetpointConfig

from fav_control.msg import StateVector3D

class SetpointsNode():
    def __init__(self):
        rospy.init_node("setpoints")
        self.x_setpoint_pub = rospy.Publisher("x_setpoint", StateVector3D, queue_size=1)
        self.y_setpoint_pub = rospy.Publisher("y_setpoint", StateVector3D, queue_size=1)
        self.z_setpoint_pub = rospy.Publisher("z_setpoint", StateVector3D, queue_size=1)
        self.roll_setpoint_pub = rospy.Publisher("roll_setpoint", StateVector3D, queue_size=1)
        self.pitch_setpoint_pub = rospy.Publisher("pitch_setpoint", StateVector3D, queue_size=1)
        self.yaw_setpoint_pub = rospy.Publisher("yaw_setpoint", StateVector3D, queue_size=1)

        self.x_setpoint_default = np.array([0.7, 0.0, 0.0])
        self.y_setpoint_default = np.array([2.0, 0.0, 0.0])
        self.z_setpoint_default = np.array([-0.6, 0.0, 0.0])
        self.roll_setpoint_default = np.array([0, 0.0, 0.0])
        self.pitch_setpoint_default = np.array([0, 0.0, 0.0])
        self.yaw_setpoint_default = np.array([np.pi/2, 0.0, 0.0])

        self.x_setpoint = self.x_setpoint_default.copy()
        self.y_setpoint = self.y_setpoint_default.copy()
        self.z_setpoint = self.z_setpoint_default.copy()
        self.roll_setpoint = self.roll_setpoint_default.copy()
        self.pitch_setpoint = self.pitch_setpoint_default.copy()
        self.yaw_setpoint = self.yaw_setpoint_default.copy()

        self.server = Server(SetpointConfig, self.server_callback)

    def run(self):
        rate = rospy.Rate(50.0)
        while not rospy.is_shutdown():
            x_msg = StateVector3D()
            x_msg.header.stamp = rospy.Time.now()
            x_msg.position = self.x_setpoint[0]
            x_msg.velocity = self.x_setpoint[1]
            x_msg.acceleration = self.x_setpoint[2]
            self.x_setpoint_pub.publish(x_msg)

            y_msg = StateVector3D()
            y_msg.header.stamp = rospy.Time.now()
            y_msg.position = self.y_setpoint[0]
            y_msg.velocity = self.y_setpoint[1]
            y_msg.acceleration = self.y_setpoint[2]
            self.y_setpoint_pub.publish(y_msg)

            z_msg = StateVector3D()
            z_msg.header.stamp = rospy.Time.now()
            z_msg.position = self.z_setpoint[0]
            z_msg.velocity = self.z_setpoint[1]
            z_msg.acceleration = self.z_setpoint[2]
            self.z_setpoint_pub.publish(z_msg)

            roll_msg = StateVector3D()
            roll_msg.header.stamp = rospy.Time.now()
            roll_msg.position = self.roll_setpoint[0]
            roll_msg.velocity = self.roll_setpoint[1]
            roll_msg.acceleration = self.roll_setpoint[2]
            self.roll_setpoint_pub.publish(roll_msg)

            pitch_msg = StateVector3D()
            pitch_msg.header.stamp = rospy.Time.now()
            pitch_msg.position = self.pitch_setpoint[0]
            pitch_msg.velocity = self.pitch_setpoint[1]
            pitch_msg.acceleration = self.pitch_setpoint[2]
            self.pitch_setpoint_pub.publish(pitch_msg)

            yaw_msg = StateVector3D()
            yaw_msg.header.stamp = rospy.Time.now()
            yaw_msg.position = self.yaw_setpoint[0]
            yaw_msg.velocity = self.yaw_setpoint[1]
            yaw_msg.acceleration = self.yaw_setpoint[2]
            self.yaw_setpoint_pub.publish(yaw_msg)

            rate.sleep()

    def server_callback(self, config, level):
        rospy.loginfo("New Parameters received by Setpoint Generator")
        

        return config


def main():
   node = SetpointsNode()
   node.run()


if __name__ == "__main__":
   main()
