import numpy as np

def frames_borders(data, minf, maxf):
    filter_data = []
    for f in data:
        if f > minf and f < maxf:
            filter_data.append(f)
    return filter_data


def skip_unnecessary_frames(steps_start, start_second_double_support, first_double_support, second_double_support):
    min_frame = np.min(steps_start)
    max_frame = np.max(steps_start)
    start_second_double_support = frames_borders(start_second_double_support, min_frame, max_frame)
    first_double_support = frames_borders(first_double_support, min_frame, max_frame)
    second_double_support = frames_borders(second_double_support, min_frame, max_frame)
    return start_second_double_support, first_double_support, second_double_support

