import numpy as np

import math


def angle_hip_joint(data_report, frame_no, video):
    shoulder = data_report.right_shoulder[frame_no]
    knee = data_report.right_knee[frame_no]
    hip = data_report.right_hip[frame_no]

    shoulder_hip = np.abs([shoulder[0] - hip[0], shoulder[1] - hip[1]])
    hip_knee = np.abs([knee[0] - hip[0], knee[1] - hip[1]])
    shoulder_knee = np.abs([shoulder[0] - knee[0], shoulder[1] - knee[1]])

    shoulder_hip_len = math.sqrt((shoulder_hip[0] * video.width) ** 2 + (shoulder_hip[1] * video.height) ** 2)
    hip_knee_len = math.sqrt((hip_knee[0] * video.width) ** 2 + (hip_knee[1] * video.height) ** 2)
    shoulder_knee_len = math.sqrt((shoulder_knee[0] * video.width) ** 2 + (shoulder_knee[1] * video.height) ** 2)
    return math.degrees(math.acos(
        (shoulder_hip_len ** 2 + hip_knee_len ** 2 - 2 * shoulder_hip_len * hip_knee_len) / shoulder_knee_len ** 2))


def angle_knee_joint(data_report, frame_no, video):
    hip = data_report.right_hip[frame_no]
    knee = data_report.right_knee[frame_no]
    ankle = data_report.right_ankle[frame_no]
    hip_knee = np.abs([knee[0] - hip[0], knee[1] - hip[1]])
    knee_ankle = np.abs([knee[0] - ankle[0], knee[1] - ankle[1]])
    hip_ankle = np.abs([hip[0] - ankle[0], hip[1] - ankle[1]])

    hip_knee_len = math.sqrt((hip_knee[0] * video.width) ** 2 + (hip_knee[1] * video.height) ** 2)
    knee_ankle_len = math.sqrt((knee_ankle[0] * video.width) ** 2 + (knee_ankle[1] * video.height) ** 2)
    hip_ankle_len = math.sqrt((hip_ankle[0] * video.width) ** 2 + (hip_ankle[1] * video.height) ** 2)
    return math.degrees(math.acos(
        (knee_ankle_len ** 2 + hip_knee_len ** 2 - 2 * knee_ankle_len * hip_knee_len) / hip_ankle_len ** 2))


def angle_ankle_joint(data_report, frame_no, video):
    knee = data_report.right_ankle[frame_no]
    heel = data_report.right_heel[frame_no]
    ankle = data_report.right_ankle[frame_no]

    knee_ankle = np.abs([knee[0] - ankle[0], knee[1] - ankle[1]])
    knee_heel = np.abs([knee[0] - heel[0], knee[1] - heel[1]])
    ankle_heel = np.abs([heel[0] - ankle[0], heel[1] - ankle[1]])

    knee_ankle_len = math.sqrt((knee_ankle[0] * video.width) ** 2 + (knee_ankle[1] * video.height) ** 2)
    knee_heel_len = math.sqrt((knee_heel[0] * video.width) ** 2 + (knee_heel[1] * video.height) ** 2)
    ankle_heel_len = math.sqrt((ankle_heel[0] * video.width) ** 2 + (ankle_heel[1] * video.height) ** 2)

    return math.degrees(math.acos(
        (knee_ankle_len ** 2 + ankle_heel_len ** 2 - 2 * knee_ankle_len * ankle_heel_len) / knee_heel_len ** 2))
