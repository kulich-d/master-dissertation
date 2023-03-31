import argparse
import os

import cv2


class VideoReader:
    def __init__(self, video_path: str):
        self.video = cv2.VideoCapture(video_path)
        self.width = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self.video.get(cv2.CAP_PROP_FPS))

    def __iter__(self):
        return self

    def __next__(self):
        success, image = self.video.read()
        if success:
            return image
        else:
            self.video.release()
            raise StopIteration


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', type=str,
                        default="/Users/diana.kulich/Documents/basketball/video/basketball_small.mp4")
    parser.add_argument('--save_path', type=str,
                        default="/Users/diana.kulich/Documents/basketball/video/basketball_small")
    args = parser.parse_args()
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    video = VideoReader(args.video_path)
    for i, image in enumerate(video):
        cv2.imwrite(os.path.join(args.save_path, "%6d.png" % i), image)
