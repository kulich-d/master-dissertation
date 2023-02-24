import cv2
import numpy as np


class VideoWriter:
    def __init__(self, save_path: str, fps: int, w: int, h: int):
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        self.video_writer = cv2.VideoWriter(save_path, fourcc, fps, (w, h))

    def write(self, image: np.ndarray):
        self.video_writer.write(image)

    def end(self):
        self.video_writer.release()