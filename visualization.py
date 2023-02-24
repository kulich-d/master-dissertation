import cv2
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import savgol_filter

import analysis


def concat_visulisations(image, steps_count):
    graph_img_x = cv2.resize(cv2.imread("temp_x.png"), (image.shape[1], image.shape[0]))
    graph_img_y = cv2.resize(cv2.imread("temp_y.png"), (image.shape[1], image.shape[0]))
    graph_img_z = cv2.resize(cv2.imread("temp_z.png"), (image.shape[1], image.shape[0]))
    font = cv2.FONT_HERSHEY_SIMPLEX
    image = cv2.putText(image, f'{len(steps_count)}', (50, 150), font,
                        2, (255, 0, 0), 2, cv2.LINE_AA)

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
    fig_x.write_image(f"temp_x.png")
    fig_y.write_image(f"temp_y.png")
    fig_z.write_image(f"temp_z.png")


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
    x_filter, y_filter = analysis.filter_data(data)

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
