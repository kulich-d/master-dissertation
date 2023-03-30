import numpy as np
from scipy.signal import savgol_filter


def filter_data_coordinates(data):
    x_filter = np.array(data)[:, 0]
    window_size = min(len(x_filter), 15)
    window_size = window_size if window_size % 2 == 1 else window_size - 1
    x_filter = savgol_filter(x_filter, window_size, min(window_size - 1, 2), mode='interp')

    y_filter = np.array(data)[:, 1]
    y_filter = savgol_filter(y_filter, window_size, min(window_size - 1, 2), mode='interp')
    return x_filter, y_filter


def filter_data_angle(data):
    x_filter = np.array(data)
    window_size = min(len(x_filter), 15)
    window_size = window_size if window_size % 2 == 1 else window_size - 1
    x_filter = savgol_filter(x_filter, window_size, min(window_size - 1, 2), mode='interp')
    return x_filter

