import argparse
import os

import cv2

import pipeline
from video_rider import VideoReader


def treadmill_coordinates(image):
    camera1_coord_template = [(300 / 960, 0 / 480), (750 / 960, 450 / 480)]
    camera_coord = [
        (int(camera1_coord_template[0][0] * image.shape[1]), int(camera1_coord_template[0][1] * image.shape[0])),
        (int(camera1_coord_template[1][0] * image.shape[1]), int(camera1_coord_template[1][1] * image.shape[0]))]

    image = cv2.rectangle(image, camera_coord[0], camera_coord[1], (0, 0, 0), 3)  # 1 camera
    return image


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', type=str, required=True)
    parser.add_argument('--save_path', type=str, required=True)
    parser.add_argument('--exp_name', type=str, required=True)
    parser.add_argument('--visualize', type=bool, default=True)
    args = parser.parse_args()
    video_name: str = args.video_path.split("/")[-1].split(".")[0]

    args.save_path = os.path.join(args.save_path, args.exp_name + video_name)
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    video = VideoReader(args.video_path)
    pipeline.main(video, args.save_path, args.visualize)
