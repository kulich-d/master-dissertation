import os

import cv2
import numpy as np

video_name = "ch01_20210909192521"
path_mediapipe = "/Users/diana.kulich/Documents/Masters/dissertation/mediapipe_results"
path_openpose = "/Users/diana.kulich/Documents/Masters/dissertation/open_pose_results"
path_yolo = "/Users/diana.kulich/Documents/Masters/dissertation/yolov7/runs/test"
save_path = "/Users/diana.kulich/Documents/Masters/dissertation/detectors_comparison"
count = 0
while True:
    f_name = os.path.join(path_mediapipe, video_name, "frame%d.jpg" % count)
    image_mediapipe = cv2.imread(f_name)  # save frame as JPEG file
    image_mediapipe  = cv2.putText(image_mediapipe, 'Mediapipe', (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                   1, (255, 0, 0), 1, cv2.LINE_AA)
    f_name = os.path.join(path_openpose, video_name, "frame%d.jpg" % count)
    image_openpose = cv2.imread(f_name)  # save frame as JPEG file
    image_openpose = cv2.putText(image_openpose, 'Openpose', (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                                  1, (255, 0, 0), 1, cv2.LINE_AA)
    f_name = os.path.join(path_yolo, video_name, "frame%d_pred.jpg" % count)
    image_yolo = cv2.imread(f_name)  # save frame as JPEG file
    image_yolo = cv2.resize(image_yolo, (960, 480))
    image_yolo = cv2.putText(image_yolo, 'Yolo', (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                                 1, (255, 0, 0), 1, cv2.LINE_AA)

    f_name = os.path.join(save_path, video_name, "frame%d.jpg" % count)
    big_image = np.stack((image_mediapipe, image_openpose, image_yolo)).reshape((480*3, 960, 3))
    cv2.imwrite(f_name, big_image)  # save frame as JPEG file
    print(count)

    count += 1
