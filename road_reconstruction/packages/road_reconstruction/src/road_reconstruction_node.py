#!/usr/bin/env python3
import cv2 as cv
import numpy as np
import rospy
from cv_bridge import CvBridge
from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CompressedImage, CameraInfo
from geometry_msgs.msg import Vector3


class RoadReconstructionNode(DTROS):
    def __init__(self, node_name):
        super(RoadReconstructionNode, self).__init__(
            node_name=node_name, node_type=NodeType.GENERIC
        )

        self.hostname = rospy.get_param("~veh")
        self.bridge = CvBridge()
        self.compressed = None

        self.compressed_sub = rospy.Subscriber(
            f"/{self.hostname}/camera_node/image/compressed",
            CompressedImage,
            self.cb_compressed,
            queue_size=1,
        )

        self.camera_info_sub = rospy.Subscriber(
            f"/{self.hostname}/camera_node/camera_info",
            CameraInfo,
            self.cb_camera_info,
            queue_size=1,
        )

    def cb_compressed(self, compressed):
        self.compressed = compressed

    def cb_camera_info(self, message):
        rospy.loginfo(message)
        self.camera_info_sub.unregister()

    def run(self, rate=1):
        return
        rate = rospy.Rate(rate)

        while not rospy.is_shutdown():
            rate.sleep()

    def onShutdown(self):
        super(RoadReconstructionNode, self).onShutdown()

if __name__ == "__main__":
    camera_node = RoadReconstructionNode(node_name="mallard_eye_node")
    camera_node.run()
