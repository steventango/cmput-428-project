#!/usr/bin/env python3
import cv2 as cv
import detailed
import numpy as np
import matplotlib.pyplot as plt
import rospy
from cv_bridge import CvBridge
from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CompressedImage, CameraInfo
from std_msgs.msg import UInt8MultiArray, Float32MultiArray
from geometry_msgs.msg import Vector3
from rospy.numpy_msg import numpy_msg
from pathlib import Path


def ransac_plane(data, max_iterations=2000, threshold=0.01):
    best_outliers = []
    best_inliers = []
    best_inlier_indices = None
    best_model = None
    best_error = np.inf
    iterations = 0
    while iterations < max_iterations:
        iterations += 1
        # sample three random points
        idx = np.random.choice(len(data), 3, replace=False)
        sample = data[idx]
        # fit a plane to the sample
        v1 = sample[1] - sample[0]
        v2 = sample[2] - sample[0]
        normal = np.cross(v1, v2)
        dist = -normal @ sample[0]
        model = np.append(normal, dist)
        # find inliers
        distances = np.abs(data @ model[:3] + model[3]) / np.linalg.norm(model[:3])
        inlier_indices = np.where(distances < threshold)[0]
        inliers = data[distances < threshold]
        outliers = data[distances >= threshold]
        # update best model if we have more inliers
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_inlier_indices = inlier_indices
            best_outliers = outliers
            best_model = model
            best_error = np.mean(distances)

    return best_inliers, best_inlier_indices, best_outliers, best_model, best_error


def get_correspondences(keypoint, mappoints_inliers_indices, mappoints):
    keypoint_to_mappoint_indices = keypoint[:, 2].copy().astype(int)
    keypoint_to_mappoint = {}
    mappoint_to_keypoint = {}
    keypoint_set = set()
    mappoint_set = set()
    for k, i in enumerate(keypoint_to_mappoint_indices):
        if i == 0 or i in mappoint_set or k in keypoint_set:
            continue
        mappoint_to_keypoint[i] = k
        keypoint_to_mappoint[k] = i
        keypoint_set.add(k)
        mappoint_set.add(i)

    keypoint = keypoint[:, :2]
    keypoint_indices = np.arange(len(keypoint))
    corresponding_inlier_mappoints_indices = sorted(
        [keypoint_to_mappoint_indices[k] for k in keypoint_indices if keypoint_to_mappoint_indices[k] in mappoints_inliers_indices and k in keypoint_to_mappoint]
    )
    corresponding_inlier_mappoints = mappoints[corresponding_inlier_mappoints_indices]
    keypoint_inliers = keypoint[[mappoint_to_keypoint[i] for i in mappoints_inliers_indices if i in mappoint_to_keypoint]]
    return keypoint_inliers, corresponding_inlier_mappoints


def project_points(model, points, vh, Rz):
    normal = model[:3]

    points = points @ vh.T
    points = points @ Rz

    projected_points = np.zeros((len(points), 2))
    for i, point in enumerate(points):
        dist = -point @ normal
        projected_point = point - dist * normal
        projected_points[i, :] = projected_point[:2]
    return projected_points


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
        self.last_update_time = None
        self.last_reconstruction_time = None

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

        self.save = False

    def cb_camera_info(self, message: CameraInfo):
        rospy.loginfo(message)
        self.camera_info = message
        self.camera_info_sub.unregister()

    def cb_keyframes(self, message: UInt8MultiArray):
        keyframes: np.ndarray = np.frombuffer(message.data, dtype=np.uint8)
        self.keyframes = keyframes.reshape(
            [dim.size for dim in message.layout.dim]
        )
        self.last_update_time = rospy.get_time()

    def cb_keypoints(self, message: Float32MultiArray):
        keypoints: np.ndarray = message.data
        self.keypoints = keypoints.reshape(
            [dim.size for dim in message.layout.dim]
        )
        self.last_update_time = rospy.get_time()

    def cb_mappoints(self, message: Float32MultiArray):
        mappoints: np.ndarray = message.data
        self.mappoints = mappoints.reshape(
            [dim.size for dim in message.layout.dim]
        )
        self.last_update_time = rospy.get_time()

    def save_numpy(self):
        DIR = Path("/data/road_reconstruction_node")
        DIR.mkdir(parents=True, exist_ok=True)
        np.save(DIR / f"keyframes_{self.last_update_time}.npy", self.keyframes)
        np.save(DIR / f"keypoints_{self.last_update_time}.npy", self.keypoints)
        np.save(DIR / f"mappoints_{self.last_update_time}.npy", self.mappoints)

    def road_reconstruction(self):
        if self.last_reconstruction_time == self.last_update_time:
            return
        if self.keyframes is None:
            return
        if self.keypoints is None:
            return
        if self.mappoints is None:
            return
        self.last_reconstruction_time = self.last_update_time
        if self.save:
            self.save_numpy()

        _, mappoints_inlier_indices, _, model, best_error = ransac_plane(
            self.mappoints
        )
        mappoints_inliers = self.mappoints[mappoints_inlier_indices]

        # use SVD to rotate points so the direction of most variance is aligned with 45 degrees
        u, s, vh = np.linalg.svd(mappoints_inliers)
        points = mappoints_inliers @ vh.T
        t = -np.pi / 4
        Rz = np.array([
            [np.cos(t), -np.sin(t), 0],
            [np.sin(t), np.cos(t), 0],
            [0, 0, 1]
        ])

        mappoints_inliers_2d = project_points(model, mappoints_inliers, vh, Rz)
        offset = -mappoints_inliers_2d.min(axis=0)
        mappoints_inliers_2d += offset
        centroid = mappoints_inliers_2d.mean(axis=0)
        std = mappoints_inliers_2d.std(axis=0)
        sigma = 2.5
        mappoints_inliers_2d = mappoints_inliers_2d[np.linalg.norm(mappoints_inliers_2d - centroid, axis=1) < sigma * std.max()]

        h, w = 1024, 1024
        s = np.array([w / np.max(mappoints_inliers_2d[:, 0]), h / np.max(mappoints_inliers_2d[:, 1])])
        canvas = np.full((h, w, 3), 255, dtype=np.uint8)
        areas = []
        for j, (keyframe, keypoint) in enumerate(zip(self.keyframes, self.keypoints)):
            keypoint_inliers, corresponding_inlier_mappoints = get_correspondences(keypoint, mappoints_inlier_indice, self.mappoints)
            corresponding_inlier_mappoints_2d = project_points(model, corresponding_inlier_mappoints, vh, Rz)
            corresponding_inlier_mappoints_2d += offset
            centroid_filter = np.linalg.norm(
                corresponding_inlier_mappoints_2d - centroid, axis=1
            ) < sigma * std.max()
            keypoint_inliers = keypoint_inliers[centroid_filter]
            corresponding_inlier_mappoints_2d = corresponding_inlier_mappoints_2d[centroid_filter]
            if len(corresponding_inlier_mappoints_2d) < 4:
                continue

            corresponding_inlier_mappoints_2d[:, :] *= s
            H = cv.findHomography(keypoint_inliers, corresponding_inlier_mappoints_2d, cv.RANSAC)[0]
            hull = cv.convexHull(keypoint_inliers)
            if hull is None:
                continue
            mask = cv.fillConvexPoly(np.zeros(keyframe.shape[:2], dtype=np.uint8), hull.astype(np.int32), 1)
            warped_mask = cv.warpPerspective(mask, H, (w, h))
            area = cv.countNonZero(warped_mask)
            areas.append(area)
            if area > 0.08 * h * w:
                continue

            warped_keyframe = keyframe.copy()
            warped_keyframe = cv.cvtColor(warped_keyframe, cv.COLOR_BGR2RGB)

            warped_keyframe[mask == 0] = 0
            warped_keyframe = cv.warpPerspective(warped_keyframe, H, (w, h), cv.INTER_LINEAR)

            slicer = (canvas == 1) & (warped_keyframe != 0)
            canvas[slicer] = warped_keyframe[slicer]
            alpha = 0.5
            canvas[warped_keyframe != 0] = (1 - alpha) * canvas[warped_keyframe != 0] + alpha * warped_keyframe[warped_keyframe != 0]

            warped_keyframe = cv.cvtColor(warped_keyframe, cv.COLOR_RGB2BGR)

        canvas = cv.cvtColor(canvas, cv.COLOR_RGB2BGR)
        cv.imshow("canvas", canvas)
        cv.waitKey(1)


    def run(self, rate=1):
        rate = rospy.Rate(rate)

        while not rospy.is_shutdown():
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
