from tslearn.metrics.dtw_variants import dtw_path
import numpy as np
from scipy.signal import find_peaks
import os
import utils

import posprocessing


def run(save_path):
    frames_start_second_double_support = start_second_double_support(save_path)
    start_steps, _ = find_peaks(
        np.loadtxt(os.path.join(save_path, f"right_heel.txt")).astype(np.float)[:, 0],
        height=0.3, distance=20)
    start_steps = [f + 1 for f in start_steps]
    frames_end_first_double_support = end_first_double_support(save_path)
    frames_end_first_double_support.sort()
    frames_end_second_double_support = end_second_double_support(save_path)
    frames_end_second_double_support.sort()
    frames_end_second_double_support = [f + 1 for f in frames_end_second_double_support]
    frames_start_second_double_support, frames_end_first_double_support, frames_end_second_double_support = posprocessing.skip_unnecessary_frames(
        start_steps, frames_start_second_double_support, frames_end_first_double_support,
        frames_end_second_double_support)
    return start_steps, frames_start_second_double_support, frames_end_first_double_support, frames_end_second_double_support


def end_first_double_support(save_path):
    coordinates = utils.barycenter_coordinates(save_path, "left")
    angle = utils.barycenter_angle(save_path, "left")
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
    coordinates = utils.barycenter_coordinates(save_path, "right")
    angle = utils.barycenter_angle(save_path, "right")
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
    print(f"common_mean_picks: {all_picks_mean}")
    return all_picks_mean


def start_second_double_support(save_path):
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
    return all_picks
