import math
import os
from dataclasses import dataclass

import cv2
import mediapipe as mp
import numpy as np
from scipy.signal import find_peaks
from tslearn.metrics.dtw_variants import dtw_path
import utils
import data_filtering
from video_rider import VideoReader
import report

mp_pose = mp.solutions.pose

import visualization


@dataclass
class Skeleton:
    right_heel = []  # 30
    right_foot_index = []  # 32
    right_knee = []  # 26
    right_hip = []  # 24
    right_ankle = []  # 28
    right_shoulder = []  # 12
    left_heel = []  # 29
    left_foot_index = []  # 31
    left_knee = []  # 25
    left_feet_angle = []
    right_feet_angle = []


def main(video: VideoReader, save_path, visualize):
    mask = 0
    skeleton = Skeleton()
    with mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            enable_segmentation=False,
            min_detection_confidence=0.5) as pose:
        for i, image in enumerate(video):
            if i < mask:
                if visualize:
                    cv2.imwrite(os.path.join(save_path, "frame%d.jpg" % i), image)
                continue
            if i > 100:
                break

            image_height, image_width, _ = image.shape
            results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            if not results.pose_landmarks:
                cv2.imwrite(os.path.join(save_path, "frame%d.jpg" % (i - mask)), image)
                continue

            utils.collect_data(skeleton, results.pose_landmarks.landmark)
            if visualize:
                annotated_image = visualization.create_annotation(image, results, skeleton)
                cv2.imwrite(os.path.join(save_path, "frame%d.jpg" % (i - mask)), annotated_image)
    utils.save_data(skeleton, save_path)
    report.create_report(save_path, i, video, visualize)

#
