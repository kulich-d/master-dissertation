import numpy as np
import math
import os
from tslearn.barycenters import dtw_barycenter_averaging_subgradient
import data_filtering
import json


def save_data(skeleton, save_path):
    filter_right_heel = np.array(data_filtering.filter_data_coordinates(skeleton.right_heel))
    np.savetxt(os.path.join(save_path, "filter_right_heel.txt"), filter_right_heel)

    filter_right_foot_index = np.array(data_filtering.filter_data_coordinates(skeleton.right_foot_index))
    np.savetxt(os.path.join(save_path, "filter_right_foot_index.txt"), filter_right_foot_index)

    filter_right_knee = np.array(data_filtering.filter_data_coordinates(skeleton.right_knee))
    np.savetxt(os.path.join(save_path, "filter_right_knee.txt"), filter_right_knee)

    filter_left_heel = np.array(data_filtering.filter_data_coordinates(skeleton.left_heel))
    np.savetxt(os.path.join(save_path, "filter_left_heel.txt"), filter_left_heel)

    filter_left_foot_index = np.array(data_filtering.filter_data_coordinates(skeleton.left_foot_index))
    np.savetxt(os.path.join(save_path, "filter_left_foot_index.txt"), filter_left_foot_index)

    filter_left_knee = np.array(data_filtering.filter_data_coordinates(skeleton.left_knee))
    np.savetxt(os.path.join(save_path, "filter_left_knee.txt"), filter_left_knee)

    filter_left_feet_angle = np.array(data_filtering.filter_data_angle(skeleton.left_feet_angle))
    np.savetxt(os.path.join(save_path, "filter_left_feet_angle.txt"), filter_left_feet_angle)

    filter_right_feet_angle = np.array(data_filtering.filter_data_angle(skeleton.right_feet_angle))
    np.savetxt(os.path.join(save_path, "filter_right_feet_angle.txt"), filter_right_feet_angle)

    # np.savetxt(os.path.join(save_path, "steps_count.txt"), steps_count)

    np.savetxt(os.path.join(save_path, "right_knee.txt"), skeleton.right_knee)
    np.savetxt(os.path.join(save_path, "right_hip.txt"), skeleton.right_hip)
    np.savetxt(os.path.join(save_path, "right_ankle.txt"), skeleton.right_ankle)
    np.savetxt(os.path.join(save_path, "right_heel.txt"), skeleton.right_heel)
    np.savetxt(os.path.join(save_path, "right_shoulder.txt"), skeleton.right_shoulder)



def read_data(data_report, save_path):
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
    data_report.right_shoulder = np.loadtxt(os.path.join(save_path, f"right_shoulder.txt"))
    return data_report

def angle_calculating(a, b):
    return math.atan(abs(a.y - b.y) / abs(a.x - b.x))



def collect_data(skeleton, landmarks):
    skeleton.right_heel.append([landmarks[30].x, np.abs(1 - landmarks[30].y)])
    skeleton.right_ankle.append([landmarks[28].x, np.abs(1 - landmarks[28].y)])
    skeleton.right_hip.append([landmarks[24].x, np.abs(1 - landmarks[24].y)])
    skeleton.right_foot_index.append([landmarks[32].x, np.abs(1 - landmarks[32].y)])
    skeleton.right_knee.append([landmarks[26].x, 1 - np.abs(landmarks[26].y)])
    skeleton.left_heel.append([landmarks[29].x, 1 - np.abs(landmarks[29].y)])
    skeleton.left_foot_index.append([landmarks[31].x, 1 - np.abs(landmarks[31].y)])
    skeleton.left_knee.append([landmarks[25].x, 1 - np.abs(landmarks[25].y)])
    skeleton.right_shoulder.append([landmarks[12].x, landmarks[12].y])
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


def save_report(save_path, report_information):
    with open(os.path.join(save_path, "report_information.json"), "w", encoding='ascii') as json_file:
        json.dump(report_information, json_file, default=str)
    with open(os.path.join(save_path, "report_information.json"), encoding='ascii') as f:
        data = json.load(f)
    print(data)