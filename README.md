# master-dissertation

    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', type=str,
                        # default="/Users/diana.kulich/Documents/Masters/dissertation/data/1/ch01_20180308130614.avi")
                        default="/Users/diana.kulich/Documents/Masters/dissertation/data/каражан/ch01_20210909192521.mp4")
    # default="/Users/diana.kulich/Documents/experiments/source_video/video_2.avi")
    parser.add_argument('--save_path', type=str,
                        default="/Users/diana.kulich/Documents/Masters/dissertation/exp")
    # default="/Users   /diana.kulich/Documents/Masters/dissertation/mediapipe_results")
    parser.add_argument('--exp_name', type=str,
                        # default="Karajan_all_processing")
                        default="Karjan_girl_all_processing_new_angles")
    # default="")