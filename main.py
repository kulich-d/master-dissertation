import argparse
import os

import cv2
import mediapipe_class
from video_rider import VideoReader


def treadmill_coordinates(image):
    # image = cv2.rectangle(image, (380, 0), (600, 450), (0,0,0), 3) # 4 camera
    camera1_coord_template = [(300 / 960, 0 / 480), (750 / 960, 450 / 480)]
    camera_coord = [
        (int(camera1_coord_template[0][0] * image.shape[1]), int(camera1_coord_template[0][1] * image.shape[0])),
        (int(camera1_coord_template[1][0] * image.shape[1]), int(camera1_coord_template[1][1] * image.shape[0]))]

    image = cv2.rectangle(image, camera_coord[0], camera_coord[1], (0, 0, 0), 3)  # 1 camera

    return image


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', type=str,
                        default="/Users/diana.kulich/Documents/Masters/dissertation/data/1/ch01_20180308130614.avi")
    # default="/Users/diana.kulich/Documents/Masters/dissertation/data/каражан/ch01_20210909192521.mp4")
    # default="/Users/diana.kulich/Documents/experiments/source_video/video_2.avi")
    parser.add_argument('--save_path', type=str,
                        default="/Users/diana.kulich/Documents/Masters/dissertation/exp")
                        # default="/Users/diana.kulich/Documents/Masters/dissertation/mediapipe_results")
    parser.add_argument('--exp_name', type=str,
                        default="experimen_february")
                        # default="")
    args = parser.parse_args()
    video_name: str = args.video_path.split("/")[-1].split(".")[0]

    args.save_path = os.path.join(args.save_path, args.exp_name + video_name)
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    video = VideoReader(args.video_path)
    mediapipe_class.main(video, args.save_path)
    #
    # for i, image in enumerate(video):
    #     cv2.imwrite(os.path.join(args.save_path, "%6d.png" % i), treadmill_coordinates(image))
    #     None + 1
