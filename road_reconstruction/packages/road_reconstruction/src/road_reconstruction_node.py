#!/usr/bin/env python3
import cv2 as cv
import numpy as np
import rospy
from cv_bridge import CvBridge
from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CompressedImage, CameraInfo
from std_msgs.msg import UInt8MultiArray, Float32MultiArray
from geometry_msgs.msg import Vector3
from rospy.numpy_msg import numpy_msg

class RoadReconstructionNode(DTROS):
    def __init__(self, node_name):
        super(RoadReconstructionNode, self).__init__(
            node_name=node_name, node_type=NodeType.GENERIC
        )

        self.hostname = rospy.get_param("~veh")
        self.bridge = CvBridge()
        self.camera_info = None

        self.keyframes = None
        self.keypoints = None
        self.mappoints = None

        self.camera_info_sub = rospy.Subscriber(
            f"/{self.hostname}/camera_node/camera_info",
            CameraInfo,
            self.cb_camera_info,
            queue_size=1,
        )

        self.keyframes_sub = rospy.Subscriber(
            "/orb_slam2/keyframes",
            numpy_msg(UInt8MultiArray),
            self.cb_keyframes,
            queue_size=1,
        )

        self.keypoints_sub = rospy.Subscriber(
            "/orb_slam2/keypoints",
            numpy_msg(Float32MultiArray),
            self.cb_keypoints,
            queue_size=1,
        )

        self.mappoints_sub = rospy.Subscriber(
            "/orb_slam2/mappoints",
            numpy_msg(Float32MultiArray),
            self.cb_mappoints,
            queue_size=1,
        )

    def cb_camera_info(self, message: CameraInfo):
        rospy.loginfo(message)
        self.camera_info = message
        self.camera_info_sub.unregister()

    def cb_keyframes(self, message: UInt8MultiArray):
        rospy.loginfo("keyframe_cb")
        keyframes: np.ndarray = np.frombuffer(message.data, dtype=np.uint8)
        self.keyframes = keyframes.reshape(
            [dim.size for dim in message.layout.dim]
        )

    def cb_keypoints(self, message: Float32MultiArray):
        rospy.loginfo("keypoints_cb")
        keypoints: np.ndarray = message.data
        self.keypoints = keypoints.reshape(
            [dim.size for dim in message.layout.dim]
        )

    def cb_mappoints(self, message: Float32MultiArray):
        rospy.loginfo("mappoints_cb")
        mappoints: np.ndarray = message.data
        self.mappoints = mappoints.reshape(
            [dim.size for dim in message.layout.dim]
        )

    def road_reconstruction(self):
        rospy.loginfo("road_reconstruction")
        rospy.loginfo(self.keyframes.shape)
        rospy.loginfo(self.keyframes.dtype)
        image = self.keyframes[2, :, :, :]
        cv.imshow("image", image)
        cv.waitKey(0)

    def run(self, rate=1):
        rate = rospy.Rate(rate)

        while not rospy.is_shutdown():
            if self.keyframes is None:
                rate.sleep()
                continue
            if self.keypoints is None:
                rate.sleep()
                continue
            if self.mappoints is None:
                rate.sleep()
                continue
            self.road_reconstruction()
            rate.sleep()


    # def onShutdown(self):
    #     super(RoadReconstructionNode, self).onShutdown()


if __name__ == "__main__":
    road_reconstruction_node = RoadReconstructionNode(
        node_name="road_reconstruction_node"
    )
    road_reconstruction_node.run()
    rospy.spin()
