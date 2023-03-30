from dataclasses import dataclass
import numpy as np
import utils
import get_characteristics
import step_segmentation
import visualization

report_info = {
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


def create_report(save_path, frames_number, video, visualize):
    data_report = DataReport()
    data_report = utils.read_data(data_report, save_path)
    step_start, start_second_double_support, end_first_double_support, end_second_double_support = step_segmentation.main(
        save_path)

    state_dict = {i: "None" for i in range(frames_number)}
    for i in range(len(step_start) - 1):
        current_step_start = step_start[i]
        current_step_end = step_start[i + 1] - 1

        report_info["Продолжительность цикла шага"].append(current_step_end - current_step_start)
        report_info["Тазобедренный сустав начальный контакт"].append(
            get_characteristics.angle_hip_joint(data_report, current_step_start, video))
        report_info["Коленный сустав начальный контакт"].append(
            get_characteristics.angle_knee_joint(data_report, current_step_start, video))
        report_info["Голеностопный сустав начальный контакт"].append(
            get_characteristics.angle_ankle_joint(data_report, current_step_start, video))

        current_position = current_step_start
        while len(end_first_double_support) > 0 and end_first_double_support[0] < current_position:
            state_frame = end_first_double_support.pop(0)
        if len(end_first_double_support) > 0 and (
                end_first_double_support[0] > current_position and end_first_double_support[
            0] < current_step_end):
            state_frame = end_first_double_support.pop(0)
            k = int(state_frame) + (state_frame - int(state_frame) != 0) + 1
            for j in range(current_position, k + 1): state_dict[j] = "First Double support"
            current_position = k
            report_info["Тазобедренный сустав период двойной опоры завершение"].append(
                get_characteristics.angle_hip_joint(data_report, current_position, video))
            report_info["Коленный сустав период двойной опоры завершение"].append(
                get_characteristics.angle_knee_joint(data_report, current_position, video))
            report_info["Голеностопный сустав период двойной опоры завершение"].append(
                get_characteristics.angle_ankle_joint(data_report, current_position, video))

        while len(start_second_double_support) > 0 and start_second_double_support[0] < current_position:
            state_frame = start_second_double_support.pop(0)
        if len(start_second_double_support) > 0 and (
                start_second_double_support[0] > current_position and start_second_double_support[
            0] < current_step_end):
            state_frame = start_second_double_support.pop(0)
            k = int(state_frame) + (state_frame - int(state_frame) != 0) + 1
            for j in range(current_position, k + 1): state_dict[j] = "Single support"
            report_info["Продолжительность фазы опоры"].append(np.abs(current_position - k))
            current_position = k
            report_info["Тазобедренный сустав период двойной опоры начало"].append(
                get_characteristics.angle_hip_joint(data_report, current_position, video))
            report_info["Коленный сустав период двойной опоры начало"].append(
                get_characteristics.angle_knee_joint(data_report, current_position, video))
            report_info["Голеностопный сустав период двойной опоры начало"].append(
                get_characteristics.angle_ankle_joint(data_report, current_position, video))

        while len(end_second_double_support) > 0 and end_second_double_support[0] < current_position:
            state_frame = end_second_double_support.pop(0)
        if len(end_second_double_support) > 0 and (
                end_second_double_support[0] > current_position and end_second_double_support[
            0] < current_step_end):
            state_frame = end_second_double_support.pop(0)
            k = int(state_frame) + (state_frame - int(state_frame) != 0) + 1
            for j in range(current_position, k + 1): state_dict[j] = "Second double support"
            current_position = k

        if current_position < current_step_end:
            all_angles = []
            for j in range(current_position, current_step_end + 1):
                state_dict[j] = "Transfer period"
                all_angles = get_characteristics.angle_knee_joint(data_report, j, video)
            report_info["Продолжительность фазы переноса"].append(np.abs(current_position - current_step_end))

            statistic_frame = np.argmax(all_angles)
            report_info["Тазобедренный сустав период переноса вертикальное положение большеберцовой кости"].append(
                get_characteristics.angle_hip_joint(data_report, statistic_frame, video))
            report_info["Коленный сустав период переноса вертикальное положение большеберцовой кости"].append(
                get_characteristics.angle_knee_joint(data_report, statistic_frame, video))
            report_info["Голеностопный сустав период переноса вертикальное положение большеберцовой кости"].append(
                get_characteristics.angle_ankle_joint(data_report, statistic_frame, video))

    if visualize:
        visualization.report_visualization(state_dict, step_start, save_path)
    utils.save_report(save_path, report_info)
