import math
import os
from dataclasses import dataclass

import cv2
import mediapipe as mp
import numpy as np
from scipy.signal import find_peaks
from tslearn.metrics.dtw_variants import dtw_path

import analysis
import visualization
from video_rider import VideoReader

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

# For static images:
BG_COLOR = (192, 192, 192)  # gray


@dataclass
class Skeleton:
    right_heel = []  # 30
    right_foot_index = []  # 32
    right_knee = []  # 26

    left_heel = []  # 29
    left_foot_index = []  # 31
    left_knee = []  # 25

    left_feet_angle = []
    right_feet_angle = []


skeleton = Skeleton()


def angle_calculating(a, b):
    return math.atan(abs(a.y - b.y) / abs(a.x - b.x))


def collect_data(landmarks):
    skeleton.right_heel.append([landmarks[30].x, np.abs(1 - landmarks[30].y)])
    skeleton.right_foot_index.append([landmarks[32].x, np.abs(1 - landmarks[32].y)])
    skeleton.right_knee.append([landmarks[26].x, 1 - np.abs(landmarks[26].y)])
    skeleton.left_heel.append([landmarks[29].x, 1 - np.abs(landmarks[29].y)])
    skeleton.left_foot_index.append([landmarks[31].x, 1 - np.abs(landmarks[31].y)])
    skeleton.left_knee.append([landmarks[25].x, 1 - np.abs(landmarks[25].y)])
    skeleton.left_feet_angle.append(angle_calculating(landmarks[29], landmarks[31]))
    skeleton.right_feet_angle.append(angle_calculating(landmarks[30], landmarks[32]))


def first_double_support():
    y = np.loadtxt("/Users/diana.kulich/Documents/Masters/dissertation/personal_idea/collect_left_feet_angle.txt")
    y_2 = np.loadtxt("/Users/diana.kulich/Documents/Masters/dissertation/personal_idea/collect_left_heel.txt")[:, 0]
    y_2 = 1 - y_2
    path_dtw = dtw_path(y, y_2)[0]
    path_dtw_1 = {a[0]: a[1] for a in path_dtw}
    path_dtw_2 = {a[1]: a[0] for a in path_dtw}
    peaks_y = find_peaks(y, height=0, distance=20)[0]
    peaks_y_2 = find_peaks(y_2, height=0, distance=20)[0]
    all_picks_mean = []
    all_picks = []

    for p in peaks_y:
        if path_dtw_1[p] in peaks_y_2:
            all_picks.append(p)
            all_picks_mean.append(np.mean([p, path_dtw_1[p]]))
    for p in peaks_y_2:
        if path_dtw_2[p] in peaks_y:
            all_picks.append(p)
            all_picks_mean.append(np.mean([p, path_dtw_2[p]]))

    print(f"common_picks: {all_picks}")
    print(f"common_mean_picks: {all_picks_mean}")  # это и есть дабл сапорт первый
    return all_picks_mean


def second_double_support():
    y_2 = np.loadtxt("/Users/diana.kulich/Documents/Masters/dissertation/personal_idea/collect_right_heel.txt")[:, 0]
    y = np.loadtxt("/Users/diana.kulich/Documents/Masters/dissertation/personal_idea/collect_right_feet_angle.txt")
    y_2 = 1 - y_2
    path_dtw = dtw_path(y, y_2)[0]
    path_dtw_1 = {a[0]: a[1] for a in path_dtw}
    path_dtw_2 = {a[1]: a[0] for a in path_dtw}
    peaks_y = find_peaks(y, height=0, distance=20)[0]
    peaks_y_2 = find_peaks(y_2, height=0, distance=20)[0]
    all_picks_mean = []
    all_picks = []

    for p in peaks_y:
        if path_dtw_1[p] in peaks_y_2:
            all_picks.append(p)
            all_picks_mean.append(np.mean([p, path_dtw_1[p]]))
    for p in peaks_y_2:
        if path_dtw_2[p] in peaks_y:
            all_picks.append(p)
            all_picks_mean.append(np.mean([p, path_dtw_2[p]]))

    print(f"common_picks: {all_picks}")
    print(f"common_mean_picks: {all_picks_mean}")  # это и есть дабл сапорт второй
    return all_picks_mean


#


def save_data(save_path):
    filter_right_heel = np.array(analysis.filter_data(skeleton.right_heel))
    np.savetxt(os.path.join(save_path, "filter_right_heel.txt"), filter_right_heel)

    filter_right_foot_index = np.array(analysis.filter_data(skeleton.right_foot_index))
    np.savetxt(os.path.join(save_path, "filter_right_foot_index.txt"), filter_right_foot_index)

    filter_right_knee = np.array(analysis.filter_data(skeleton.right_knee))
    np.savetxt(os.path.join(save_path, "filter_right_knee.txt"), filter_right_knee)

    filter_left_heel = np.array(analysis.filter_data(skeleton.left_heel))
    np.savetxt(os.path.join(save_path, "filter_left_heel.txt"), filter_left_heel)

    filter_left_foot_index = np.array(analysis.filter_data(skeleton.left_foot_index))
    np.savetxt(os.path.join(save_path, "filter_left_foot_index.txt"), filter_left_foot_index)

    filter_left_knee = np.array(analysis.filter_data(skeleton.left_knee))
    np.savetxt(os.path.join(save_path, "filter_left_knee.txt"), filter_left_knee)

    filter_left_feet_angle = np.array(analysis.filter_data(skeleton.left_feet_angle))
    np.savetxt(os.path.join(save_path, "filter_left_feet_angle.txt"), filter_left_feet_angle)

    filter_right_feet_angle = np.array(analysis.filter_data(skeleton.right_feet_angle))
    np.savetxt(os.path.join(save_path, "filter_right_feet_angle.txt"), filter_right_feet_angle)


def main(video: VideoReader, save_path):
    with mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            enable_segmentation=False,
            min_detection_confidence=0.5) as pose:
        for i, image in enumerate(video):
            # if i < 440 :
            #     cv2.imwrite(os.path.join(save_path, "frame%d.jpg" % i), image)
            #
            #     continue
            if i > 30:
                # if i > 300:
                break
            image_height, image_width, _ = image.shape
            results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            if not results.pose_landmarks:
                cv2.imwrite(os.path.join(save_path, "frame%d.jpg" % i), image)
                continue

            collect_data(results.pose_landmarks.landmark)
            annotated_image = image.copy()
            bg_image = np.zeros(image.shape, dtype=np.uint8)
            bg_image[:] = BG_COLOR
            mp_drawing.draw_landmarks(
                annotated_image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

            steps_count, _ = find_peaks(np.array(skeleton.right_heel)[:, 0], height=0)# recognize start os the new step like max of right heel
            visualization.visualize_landmarks_coordinates(skeleton)
            annotated_image = visualization.concat_visulisations(annotated_image, steps_count)
            cv2.imwrite(os.path.join(save_path, "frame%d.jpg" % i), annotated_image)

    save_data(save_path)
    post_analysis(save_path)


def post_analysis(save_path, ):
    frames_first_double_support = first_double_support()
    frames_second_double_support = second_double_support()
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i in frames_first_double_support:
        i = int(i) + (i - int(i) != 0) + 1
        image = cv2.imread(os.path.join(save_path, "frame%d.jpg" % i))
        image = cv2.putText(image, 'Ends first double support', (1600, 150), font,
                            2, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.imwrite(os.path.join(save_path, "frame%d.jpg" % i), image)
    for i in frames_second_double_support:
        i = int(i) + (i - int(i) != 0) + 1
        image = cv2.imread(os.path.join(save_path, "frame%d.jpg" % i))
        image = cv2.putText(image, 'Ends second double support', (1600, 150), font,
                            2, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.imwrite(os.path.join(save_path, "frame%d.jpg" % i), image)
