import os

import cv2

vidcap = cv2.VideoCapture(
    '/Users/diana.kulich/Documents/Masters/dissertation/open_pose_results/ch01_20210909192521.mp4')
# vidcap = cv2.VideoCapture('/Users/diana.kulich/Documents/experiments/source_video/video_2.avi')
save_path = "/Users/diana.kulich/Documents/Masters/dissertation/open_pose_results/ch01_20210909192521"
# save_path = "/Users/diana.kulich/Documents/Masters/dissertation/video_to_frame/video_2"
success, image = vidcap.read()
count = 0
all_path = []
while success:
    f_name = os.path.join(save_path, "frame%d.jpg" % count)
    cv2.imwrite(f_name, image)  # save frame as JPEG file
    all_path.append(f_name)
    success, image = vidcap.read()
    print('Read a new frame: ', success)
    count += 1

# np.savetxt('/Users/diana.kulich/Documents/Masters/dissertation/yolov7/video_2.txt', np.array(all_path), fmt='%s')
