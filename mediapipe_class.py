import math
import os
from dataclasses import dataclass

import cv2
import mediapipe as mp
import numpy as np
from scipy.signal import find_peaks
from tslearn.barycenters import dtw_barycenter_averaging_subgradient
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
    right_hip = []  # 24
    right_ankle = []  # 28

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
    skeleton.right_ankle.append([landmarks[28].x, np.abs(1 - landmarks[28].y)])
    skeleton.right_hip.append([landmarks[24].x, np.abs(1 - landmarks[24].y)])
    skeleton.right_foot_index.append([landmarks[32].x, np.abs(1 - landmarks[32].y)])
    skeleton.right_knee.append([landmarks[26].x, 1 - np.abs(landmarks[26].y)])
    skeleton.left_heel.append([landmarks[29].x, 1 - np.abs(landmarks[29].y)])
    skeleton.left_foot_index.append([landmarks[31].x, 1 - np.abs(landmarks[31].y)])
    skeleton.left_knee.append([landmarks[25].x, 1 - np.abs(landmarks[25].y)])
    skeleton.left_feet_angle.append(angle_calculating(landmarks[29], landmarks[31]))
    skeleton.right_feet_angle.append(angle_calculating(landmarks[30], landmarks[32]))


def barycenter_coordinates(save_path, side):
    x_heel = 1 - np.loadtxt(os.path.join(save_path, f"filter_{side}_heel.txt"))[0, :]
    x_foot_index = 1 - np.loadtxt(os.path.join(save_path, f"filter_{side}_foot_index.txt"))[0, :]
    X = np.stack([x_heel, x_foot_index])
    coordinates_barycenter = dtw_barycenter_averaging_subgradient(X, max_iter=50, tol=1e-3)

    return np.squeeze(coordinates_barycenter)


def barycenter_angle(save_path, side):
    angle = np.loadtxt(os.path.join(save_path, f"filter_{side}_feet_angle.txt"))
    knee = 1 - np.loadtxt(os.path.join(save_path, f"filter_{side}_knee.txt"))[1, :]
    X = np.stack([angle, knee])
    angle_barycenter = dtw_barycenter_averaging_subgradient(X, max_iter=50, tol=1e-3)
    return np.squeeze(angle_barycenter)


def end_first_double_support(save_path):
    coordinates = barycenter_coordinates(save_path, "left")
    angle = barycenter_angle(save_path, "left")
    path_dtw = dtw_path(coordinates, angle)[0]
    path_dtw_1 = {a[0]: a[1] for a in path_dtw}
    path_dtw_2 = {a[1]: a[0] for a in path_dtw}
    peaks_y = find_peaks(coordinates, height=0, distance=20)[0]
    peaks_y_2 = find_peaks(angle, height=0, distance=20)[0]
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


def end_second_double_support(save_path):
    coordinates = barycenter_coordinates(save_path, "right")
    angle = barycenter_angle(save_path, "right")
    path_dtw = dtw_path(coordinates, angle)[0]
    path_dtw_1 = {a[0]: a[1] for a in path_dtw}
    path_dtw_2 = {a[1]: a[0] for a in path_dtw}
    peaks_y = find_peaks(coordinates, height=0, distance=20)[0]
    peaks_y_2 = find_peaks(angle, height=0, distance=20)[0]
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


#


def save_data(save_path, steps_count):
    filter_right_heel = np.array(analysis.filter_data_coordinates(skeleton.right_heel))
    np.savetxt(os.path.join(save_path, "filter_right_heel.txt"), filter_right_heel)

    filter_right_foot_index = np.array(analysis.filter_data_coordinates(skeleton.right_foot_index))
    np.savetxt(os.path.join(save_path, "filter_right_foot_index.txt"), filter_right_foot_index)

    filter_right_knee = np.array(analysis.filter_data_coordinates(skeleton.right_knee))
    np.savetxt(os.path.join(save_path, "filter_right_knee.txt"), filter_right_knee)

    filter_left_heel = np.array(analysis.filter_data_coordinates(skeleton.left_heel))
    np.savetxt(os.path.join(save_path, "filter_left_heel.txt"), filter_left_heel)

    filter_left_foot_index = np.array(analysis.filter_data_coordinates(skeleton.left_foot_index))
    np.savetxt(os.path.join(save_path, "filter_left_foot_index.txt"), filter_left_foot_index)

    filter_left_knee = np.array(analysis.filter_data_coordinates(skeleton.left_knee))
    np.savetxt(os.path.join(save_path, "filter_left_knee.txt"), filter_left_knee)

    filter_left_feet_angle = np.array(analysis.filter_data_angle(skeleton.left_feet_angle))
    np.savetxt(os.path.join(save_path, "filter_left_feet_angle.txt"), filter_left_feet_angle)

    filter_right_feet_angle = np.array(analysis.filter_data_angle(skeleton.right_feet_angle))
    np.savetxt(os.path.join(save_path, "filter_right_feet_angle.txt"), filter_right_feet_angle)

    np.savetxt(os.path.join(save_path, "steps_count.txt"), steps_count)

    np.savetxt(os.path.join(save_path, "right_knee.txt"), skeleton.right_knee)
    np.savetxt(os.path.join(save_path, "right_hip.txt"), skeleton.right_hip)
    np.savetxt(os.path.join(save_path, "right_ankle.txt"), skeleton.right_ankle)
    np.savetxt(os.path.join(save_path, "right_heel.txt"), skeleton.right_heel)


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
            if i > 700:
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

            steps_count, _ = find_peaks(np.array(skeleton.right_heel)[:, 0],
                                        height=0)  # recognize start os the new step like max of right heel

            visualization.visualize_landmarks_coordinates(skeleton)
            annotated_image = visualization.concat_visulisations(annotated_image, steps_count)
            cv2.imwrite(os.path.join(save_path, "frame%d.jpg" % i), annotated_image)

    save_data(save_path, steps_count)
    post_analysis(save_path, i)


def start_second_double_support(save_path):
    # left_heel = np.loadtxt(os.path.join(save_path, f"filter_left_knee.txt"))[1, :]
    # left_foot_index = np.loadtxt(os.path.join(save_path, f"filter_left_knee.txt"))[0, :]
    left_heel = np.loadtxt(os.path.join(save_path, f"filter_left_heel.txt"))[0, :]
    left_foot_index = np.loadtxt(os.path.join(save_path, f"filter_left_foot_index.txt"))[0, :]

    path_dtw = dtw_path(left_heel, left_foot_index)[0]
    path_dtw_1 = {a[0]: a[1] for a in path_dtw}
    path_dtw_2 = {a[1]: a[0] for a in path_dtw}
    peaks_y = find_peaks(left_heel, height=0, distance=20)[0]
    peaks_y_2 = find_peaks(left_foot_index, height=0, distance=20)[0]
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

    print(f"single support common_picks: {all_picks}")
    print(f"single support common_mean_picks: {all_picks_mean}")  # это и есть дабл сапорт первый
    return all_picks_mean


def frames_borders(data, minf, maxf):
    filter_data = []
    for f in data:
        if f > minf and f < maxf:
            filter_data.append(f)
    return filter_data


def skip_unnecessary_frames(frames_with_steps, frames_start_second_double_support, frames_first_double_support,
                            frames_second_double_support):
    min_frame = np.min(frames_with_steps)
    max_frame = np.max(frames_with_steps)
    frames_start_second_double_support = frames_borders(frames_start_second_double_support, min_frame, max_frame)
    frames_first_double_support = frames_borders(frames_first_double_support, min_frame, max_frame)
    frames_second_double_support = frames_borders(frames_second_double_support, min_frame, max_frame)
    return frames_start_second_double_support, frames_first_double_support, frames_second_double_support


all_information = {
    "Продолжительность цикла шага": [],
    "Продолжительность фазы опоры": [],
    "Продолжительность фазы переноса": [],

    "Тазобедренный сустав начальный контакт": [],
    "Коленный сустав начальный контакт": [],
    "Голеностопный сустав начальный контакт": [],

    "Тазобедренный сустав период двойной опоры завершение": [],
    "Коленный сустав период двойной опоры завершение": [],
    "Голеностопный сустав период двойной опоры завершение": [],

    "Тазобедренный сустав период двойной опоры начало": [],
    "Коленный сустав период двойной опоры начало": [],
    "Голеностопный сустав период двойной опоры начало": [],

    "Тазобедренный сустав период переноса вертикальное положение большеберцовой кости": [],
    "Коленный сустав период переноса вертикальное положение большеберцовой кости": [],
    "Голеностопный сустав период переноса вертикальное положение большеберцовой кости": [],

}


@dataclass
class DataReport:
    right_heel = []  # 30
    right_knee = []  # 26
    right_hip = []  # 24
    right_ankle = []  # 28


data_report = DataReport()


def find_angle_for_report(save_path):
    # тазобедренный сустав 24 26
    # коленный сустав 26 28
    # голеностопный сустав 28 30

    # right_heel = []  # 30
    # right_foot_index = []  # 32
    # right_knee = []  # 26
    # right_hip = []  # 24
    # right_ankle = []  # 28
    data_report.right_knee = np.loadtxt(os.path.join(save_path, f"right_knee.txt"))
    data_report.right_hip = np.loadtxt(os.path.join(save_path, f"right_hip.txt"))
    data_report.right_ankle = np.loadtxt(os.path.join(save_path, f"right_ankle.txt"))
    data_report.right_heel = np.loadtxt(os.path.join(save_path, f"right_heel.txt"))


def angle_hip_joint(frame_no):  # todo check coordinates order
    knee = data_report.right_knee[frame_no]
    hip = data_report.right_hip[frame_no]
    return math.degrees(math.atan(np.abs(knee[0] - hip[0]) / np.abs(knee[1] - hip[1])))


def angle_knee_joint(frame_no):  # todo check coordinates order
    knee = data_report.right_knee[frame_no]
    ankle = data_report.right_ankle[frame_no]
    return math.degrees(math.atan(np.abs(knee[0] - ankle[0]) / np.abs(knee[1] - ankle[1])))


def angle_ankle_joint(frame_no):  # todo check coordinates order
    knee = data_report.right_knee[frame_no]
    heel = data_report.right_heel[frame_no]
    return math.degrees(math.atan(np.abs(knee[0] - heel[0]) / np.abs(knee[1] - heel[1])))


def post_analysis(save_path, frames_number):
    find_angle_for_report(save_path)
    frames_start_second_double_support = start_second_double_support(save_path)
    frames_with_steps = np.loadtxt(os.path.join(save_path, f"steps_count.txt")).astype(np.int)
    frames_end_first_double_support = end_first_double_support(save_path)
    frames_end_second_double_support = end_second_double_support(save_path)
    frames_start_second_double_support, frames_end_first_double_support, frames_end_second_double_support = skip_unnecessary_frames(
        frames_with_steps, frames_start_second_double_support, frames_end_first_double_support,
        frames_end_second_double_support)
    font = cv2.FONT_HERSHEY_SIMPLEX
    state_dict = {i: "None" for i in range(frames_number)}
    for i in range(len(frames_with_steps) - 1):
        current_step_start = frames_with_steps[i]
        current_step_end = frames_with_steps[i + 1] - 1

        all_information["Продолжительность цикла шага"].append(current_step_end - current_step_start)
        all_information["Тазобедренный сустав начальный контакт"].append(angle_hip_joint(current_step_start))
        all_information["Коленный сустав начальный контакт"].append(angle_knee_joint(current_step_start))
        all_information["Голеностопный сустав начальный контакт"].append(angle_ankle_joint(current_step_start))

        current_position = current_step_start
        if len(frames_end_first_double_support) > 0 and (
                frames_end_first_double_support[0] > current_position and frames_end_first_double_support[
            0] < current_step_end):
            state_frame = frames_end_first_double_support.pop(0)
            k = int(state_frame) + (state_frame - int(state_frame) != 0) + 1
            for j in range(current_position, k + 1): state_dict[j] = "First Double support"
            current_position = k
            all_information["Тазобедренный сустав период двойной опоры завершение"].append(
                angle_hip_joint(current_position))
            all_information["Коленный сустав период двойной опоры завершение"].append(
                angle_knee_joint(current_position))
            all_information["Голеностопный сустав период двойной опоры завершение"].append(
                angle_ankle_joint(current_position))

        if len(frames_start_second_double_support) > 0 and (
                frames_start_second_double_support[0] > current_position and frames_start_second_double_support[
            0] < current_step_end):
            state_frame = frames_start_second_double_support.pop(0)
            k = int(state_frame) + (state_frame - int(state_frame) != 0) + 1
            for j in range(current_position, k + 1): state_dict[j] = "Single support"
            all_information["Продолжительность фазы опоры"].append(np.abs(current_position - k))
            current_position = k
            all_information["Тазобедренный сустав период двойной опоры начало"].append(
                angle_hip_joint(current_position))
            all_information["Коленный сустав период двойной опоры начало"].append(
                angle_knee_joint(current_position))
            all_information["Голеностопный сустав период двойной опоры начало"].append(
                angle_ankle_joint(current_position))

        if len(frames_end_second_double_support) > 0 and (
                frames_end_second_double_support[0] > current_position and frames_end_second_double_support[
            0] < current_step_end):
            state_frame = frames_end_second_double_support.pop(0)
            k = int(state_frame) + (state_frame - int(state_frame) != 0) + 1
            for j in range(current_position, k + 1): state_dict[j] = "Second double support"
            current_position = k

        if current_position < current_step_end:
            all_angles = []
            for j in range(current_position, current_step_end + 1):
                state_dict[j] = "Transfer period"
                all_angles = angle_knee_joint(j)
            all_information["Продолжительность фазы переноса"].append(np.abs(current_position - current_step_end))

            statistic_frame = np.argmax(all_angles)
            all_information["Тазобедренный сустав период переноса вертикальное положение большеберцовой кости"].append(
                angle_hip_joint(statistic_frame))
            all_information["Коленный сустав период переноса вертикальное положение большеберцовой кости"].append(
                angle_knee_joint(statistic_frame))
            all_information["Голеностопный сустав период переноса вертикальное положение большеберцовой кости"].append(
                angle_ankle_joint(statistic_frame))

    for k, v in state_dict.items():
        image = cv2.imread(os.path.join(save_path, "frame%d.jpg" % k))
        image = cv2.putText(image, v, (1600, 150), font,
                            2, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.imwrite(os.path.join(save_path, "frame%d.jpg" % k), image)

    import json
    with open(os.path.join(save_path, "all_information.json"), "w", encoding='ascii') as json_file:
        json.dump(all_information, json_file, default=str)
    # print(all_information)
    # Reading with the json module
    with open(os.path.join(save_path, "all_information.json"), encoding='ascii') as f:
        data = json.load(f)
    print(data)
