import math
import os

import cv2
import mediapipe as mp
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import savgol_filter
from tslearn.metrics.dtw_variants import dtw_path

from video_rider import VideoReader

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

# For static images:
IMAGE_FILES = []
BG_COLOR = (192, 192, 192)  # gray

collect_right_heel = []  # 30
collect_right_foot_index = []  # 32
collect_right_knee = []  # 26

collect_left_heel = []  # 29
collect_left_foot_index = []  # 31
collect_left_knee = []  # 25
from scipy.signal import find_peaks

collect_left_feet_angle = []
collect_right_feet_angle = []


def angle_calculating(a, b):
    return math.atan(abs(a.y - b.y) / abs(a.x - b.x))


def collect_data(landmarks):
    collect_right_heel.append([landmarks[30].x, np.abs(1 - landmarks[30].y)])
    collect_right_foot_index.append([landmarks[32].x, np.abs(1 - landmarks[32].y)])
    collect_right_knee.append([landmarks[26].x, 1 - np.abs(landmarks[26].y)])
    collect_left_heel.append([landmarks[29].x, 1 - np.abs(landmarks[29].y)])
    collect_left_foot_index.append([landmarks[31].x, 1 - np.abs(landmarks[31].y)])
    collect_left_knee.append([landmarks[25].x, 1 - np.abs(landmarks[25].y)])

    collect_left_feet_angle.append(angle_calculating(landmarks[29], landmarks[31]))
    collect_right_feet_angle.append(angle_calculating(landmarks[30], landmarks[32]))


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


def add_information(image, i, peaks,
                    Current_step):
    graph_img_x = cv2.resize(cv2.imread("temp_x.png"), (image.shape[1], image.shape[0]))
    graph_img_y = cv2.resize(cv2.imread("temp_y.png"), (image.shape[1], image.shape[0]))
    graph_img_z = cv2.resize(cv2.imread("temp_z.png"), (image.shape[1], image.shape[0]))
    font = cv2.FONT_HERSHEY_SIMPLEX
    image = cv2.putText(image, f'{len(peaks)}', (50, 150), font,
                        2, (255, 0, 0), 2, cv2.LINE_AA)

    # if len(peaks) > 0:
    #     overlay = image.copy()
    #     cv2.rectangle(overlay, (0, 0), image.shape[:2][::-1], (0, 200, 0), -1)
    #     alpha = 0.4  # Transparency factor.
    #     image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    # information_image = np.zeros(image.shape, dtype=np.uint8)
    graph_img = np.concatenate([graph_img_x, graph_img_y], axis=1)
    image = np.concatenate([image, graph_img_z], axis=1)
    image = np.concatenate([image, graph_img], axis=0)
    return image, Current_step


#

def visualize():
    #
    # plt.plot(x, y, "o", label="observation")
    # plt.plot(x, y_filter, 'g', lw=3)
    # fig_x.update_xaxes(title_text="value", title_standoff = 0)
    # fig_x.update_yaxes(title_text="frame_number", title_standoff = 0)

    fig_x = make_subplots(rows=3, cols=2, x_title="value", y_title="frame_number")
    fig_y = make_subplots(rows=3, cols=2, x_title="value", y_title="frame_number")
    fig_z = make_subplots(rows=1, cols=2, x_title="value", y_title="frame_number")

    plot_results(collect_right_heel, 1, fig_x, fig_y, "collect_right_heel")

    plot_results(collect_right_foot_index, 2, fig_x, fig_y, "collect_right_foot_index")
    plot_results(collect_right_knee, 3, fig_x, fig_y, "collect_righ t_knee")

    plot_results(collect_left_heel, 4, fig_x, fig_y, "collect_left_heel")
    plot_results(collect_left_foot_index, 5, fig_x, fig_y, "collect_left_foot_index")
    plot_results(collect_left_knee, 6, fig_x, fig_y, "collect_left_knee")
    plot_len(collect_left_feet_angle, 1, fig_z, "collect_left_feet_angle")
    plot_len(collect_right_feet_angle, 2, fig_z, "collect_right_feet_angle")
    fig_x.write_image(f"temp_x.png")
    fig_y.write_image(f"temp_y.png")
    fig_z.write_image(f"temp_z.png")


def plot_len(data, i, fig_y, name):
    color = px.colors.sequential.Plasma[i]

    fig_y.append_trace(go.Scatter(
        x=[k for k in range(len(data))], y=np.array(data),
        name=f'{name}_y',
        mode='markers',
        marker_color=color
    ), row=1, col=i)
    color = px.colors.sequential.Plasma[i + 1]

    y_filter = np.array(data)
    window_size = min(len(y_filter), 15)
    window_size = window_size if window_size % 2 == 1 else window_size - 1
    y_filter = savgol_filter(y_filter, window_size, min(window_size - 1, 2), mode='interp')

    fig_y.append_trace(go.Scatter(
        x=[k for k in range(len(data))], y=y_filter,
        name=f'{name}_y',
        mode='lines',
        marker_color="red"
    ), row=1, col=i)
    color = px.colors.sequential.Plasma[i + 1]


def plot_results(data, i, fig_x, fig_y, name):
    color = px.colors.sequential.Plasma[i]

    row = i % 3
    if row == 0: row = 3

    fig_y.append_trace(go.Scatter(
        x=[k for k in range(len(data))], y=np.array(data)[:, 1],
        name=f'{name}_y',
        mode='markers',
        marker_color=color
    ), row=row, col=int((i - 1) / 3) + 1)
    color = px.colors.sequential.Plasma[i + 1]

    fig_x.append_trace(go.Scatter(
        x=[k for k in range(len(data))], y=np.array(data)[:, 0],
        name=f'{name}_x',
        mode='markers',
        marker_color=color
    ), row=row, col=int((i - 1) / 3) + 1)

    x_filter = np.array(data)[:, 1]
    window_size = min(len(x_filter), 15)
    window_size = window_size if window_size % 2 == 1 else window_size - 1
    print(window_size)
    x_filter = savgol_filter(x_filter, window_size, min(window_size - 1, 2), mode='interp')

    y_filter = np.array(data)[:, 0]
    y_filter = savgol_filter(y_filter, window_size, min(window_size - 1, 2), mode='interp')

    fig_y.append_trace(go.Scatter(
        x=[k for k in range(len(data))], y=x_filter,
        name=f'{name}_y',
        mode='lines',
        marker_color="red"
    ), row=row, col=int((i - 1) / 3) + 1)
    color = px.colors.sequential.Plasma[i + 1]

    fig_x.append_trace(go.Scatter(
        x=[k for k in range(len(data))], y=y_filter,
        name=f'{name}_x',
        mode='lines',
        marker_color="red"
    ), row=row, col=int((i - 1) / 3) + 1)

    # np.savetxt(f"here_{i}_x.csv", np.array(data)[:, 0], delimiter=",")
    # fig_x.update_layout(xaxis_title="fame number", yaxis_title="value",)
    # fig_y.update_layout(xaxis_title="fame number", yaxis_title="value",)
    # fig_x.update_xaxes(title_text="value", title_standoff = 0)
    # fig_x.update_yaxes(title_text="frame_number", title_standoff = 0)


def main(video: VideoReader, save_path):
    Current_step = ""

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
            if i > 300:
                break
            image_height, image_width, _ = image.shape
            # Convert the BGR image to RGB before processing.
            results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            if not results.pose_landmarks:
                cv2.imwrite(os.path.join(save_path, "frame%d.jpg" % i), image)

                continue
            collect_data(results.pose_landmarks.landmark)
            annotated_image = image.copy()
            # Draw segmentation on the image.
            # To improve segmentation around boundaries, consider applying a joint
            # bilateral filter to "results.segmentation_mask" with "image".
            # condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
            bg_image = np.zeros(image.shape, dtype=np.uint8)
            bg_image[:] = BG_COLOR
            # annotated_image = np.where(condition, annotated_image, bg_image)
            # Draw pose landmarks on the image.
            mp_drawing.draw_landmarks(
                annotated_image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
            peaks_steps, _ = find_peaks(np.array(collect_right_heel)[:, 0], height=0)
            #
            # print(i)
            visualize()
            annotated_image, Current_step = add_information(annotated_image, i, peaks_steps, Current_step)
            cv2.imwrite(os.path.join(save_path, "frame%d.jpg" % i), annotated_image)
            # Plot pose world landmarks.
            # mp_drawing.plot_landmarks(save_path, "world_%6d" % i,
            #                           results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)

    collect_right_heel_array = np.array(collect_right_heel)
    collect_right_heel_array[:, 0] = savgol_filter(collect_right_heel_array[:, 0], 15, 2, mode='interp')
    collect_right_heel_array[:, 1] = savgol_filter(collect_right_heel_array[:, 1], 15, 2, mode='interp')
    np.savetxt("collect_right_heel.txt", collect_right_heel_array)

    collect_right_foot_index_array = np.array(collect_right_foot_index)
    collect_right_foot_index_array[:, 1] = savgol_filter(collect_right_foot_index_array[:, 1], 15, 2, mode='interp')
    collect_right_foot_index_array[:, 0] = savgol_filter(collect_right_foot_index_array[:, 0], 15, 2, mode='interp')
    np.savetxt("collect_right_foot_index.txt", collect_right_foot_index_array)

    collect_right_knee_array = np.array(collect_right_knee)
    collect_right_knee_array[:, 1] = savgol_filter(collect_right_knee_array[:, 1], 15, 2, mode='interp')
    collect_right_knee_array[:, 0] = savgol_filter(collect_right_knee_array[:, 0], 15, 2, mode='interp')
    np.savetxt("collect_right_knee.txt", collect_right_knee_array)

    collect_left_heel_array = np.array(collect_left_heel)
    collect_left_heel_array[:, 1] = savgol_filter(collect_left_heel_array[:, 1], 15, 2, mode='interp')
    collect_left_heel_array[:, 0] = savgol_filter(collect_left_heel_array[:, 0], 15, 2, mode='interp')
    np.savetxt("collect_left_heel.txt", collect_left_heel_array)

    collect_left_foot_index_array = np.array(collect_left_foot_index)
    collect_left_foot_index_array[:, 1] = savgol_filter(collect_left_foot_index_array[:, 1], 15, 2, mode='interp')
    collect_left_foot_index_array[:, 0] = savgol_filter(collect_left_foot_index_array[:, 0], 15, 2, mode='interp')
    np.savetxt("collect_left_foot_index.txt", collect_left_foot_index_array)

    collect_left_knee_array = np.array(collect_left_knee)
    collect_left_knee_array[:, 1] = savgol_filter(collect_left_knee_array[:, 1], 15, 2, mode='interp')
    collect_left_knee_array[:, 0] = savgol_filter(collect_left_knee_array[:, 0], 15, 2, mode='interp')
    np.savetxt("collect_left_knee.txt", collect_left_knee_array)

    np.savetxt("collect_left_feet_angle.txt", collect_left_feet_angle)

    np.savetxt("collect_right_feet_angle.txt", collect_right_feet_angle)
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
