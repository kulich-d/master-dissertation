import numpy as np

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

