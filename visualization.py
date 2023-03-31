import cv2
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import savgol_filter
import os
import data_filtering
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

# For static images:
BG_COLOR = (192, 192, 192)  # gray

x_file_name = "temp_x.png"
y_file_name = "temp_y.png"
z_file_name = "temp_z.png"


def create_annotation(image, results, skeleton):
    annotated_image = image.copy()
    bg_image = np.zeros(image.shape, dtype=np.uint8)
    bg_image[:] = BG_COLOR
    mp_drawing.draw_landmarks(
        annotated_image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

    visualize_landmarks_coordinates(skeleton)
    annotated_image = concat_visulisations(annotated_image)
    return annotated_image


def concat_visulisations(image):
    graph_img_x = cv2.resize(cv2.imread(x_file_name), (image.shape[1], image.shape[0]))
    graph_img_y = cv2.resize(cv2.imread(y_file_name), (image.shape[1], image.shape[0]))
    graph_img_z = cv2.resize(cv2.imread(z_file_name), (image.shape[1], image.shape[0]))
    graph_img = np.concatenate([graph_img_x, graph_img_y], axis=1)
    image = np.concatenate([image, graph_img_z], axis=1)
    image = np.concatenate([image, graph_img], axis=0)
    return image


def visualize_landmarks_coordinates(skeleton):
    fig_x = make_subplots(rows=3, cols=2, x_title="value", y_title="frame_number")
    fig_y = make_subplots(rows=3, cols=2, x_title="value", y_title="frame_number")
    fig_z = make_subplots(rows=1, cols=2, x_title="value", y_title="frame_number")

    visualize_coordinates(skeleton.right_heel, 1, fig_x, fig_y, "right_heel")
    visualize_coordinates(skeleton.right_foot_index, 2, fig_x, fig_y, "right_foot_index")
    visualize_coordinates(skeleton.right_knee, 3, fig_x, fig_y, "right_knee")
    visualize_coordinates(skeleton.left_heel, 4, fig_x, fig_y, "left_heel")
    visualize_coordinates(skeleton.left_foot_index, 5, fig_x, fig_y, "left_foot_index")
    visualize_coordinates(skeleton.left_knee, 6, fig_x, fig_y, "left_knee")
    visualize_angle(skeleton.left_feet_angle, 1, fig_z, "left_feet_angle")
    visualize_angle(skeleton.right_feet_angle, 2, fig_z, "right_feet_angle")
    fig_x.write_image(x_file_name)
    fig_y.write_image(y_file_name)
    fig_z.write_image(z_file_name)


def visualize_angle(data, i, fig_y, name):
    color = px.colors.sequential.Plasma[i]

    fig_y.append_trace(go.Scatter(
        x=[k for k in range(len(data))], y=np.array(data),
        name=f'{name}_y',
        mode='markers',
        marker_color=color
    ), row=1, col=i)

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


def visualize_coordinates(data, i, fig_x, fig_y, name):
    color = px.colors.sequential.Plasma[i]
    x_filter, y_filter = data_filtering.filter_data_coordinates(data)

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

    fig_y.append_trace(go.Scatter(
        x=[k for k in range(len(data))], y=y_filter,
        name=f'{name}_y',
        mode='lines',
        marker_color="red"
    ), row=row, col=int((i - 1) / 3) + 1)

    fig_x.append_trace(go.Scatter(
        x=[k for k in range(len(data))], y=x_filter,
        name=f'{name}_x',
        mode='lines',
        marker_color="red"
    ), row=row, col=int((i - 1) / 3) + 1)


def report_visualization(state_dict, step_start, save_path):
    steps_count = 0
    font = cv2.FONT_HERSHEY_SIMPLEX
    for k, v in state_dict.items():

        image = cv2.imread(os.path.join(save_path, "frame%d.jpg" % k))
        if k in step_start:
            steps_count += 1
        image = cv2.putText(image, f'{steps_count}', (50, 150), font,
                            2, (255, 0, 0), 2, cv2.LINE_AA)
        image = cv2.putText(image, v, (1600, 150), font,
                            2, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.imwrite(os.path.join(save_path, "frame%d.jpg" % k), image)
