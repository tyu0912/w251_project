import cv2
import time
import argparse
import os
import sys
import shutil
from datetime import datetime

def dump_frames(vid_path, out_path, frame_width, frame_height, frame_count):
    
    cap = cv2.VideoCapture(vid_path)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    current_image_collection = []

    try:
        shutil.rmtree(out_path)
        os.mkdir(out_path)
    except:
        os.mkdir(out_path)

    i=0
    while True:

        try:
            ret, frame = cap.read()

            cv2.imshow('frame', frame)
            filename = '{}/{:06d}.jpg'.format(out_path, i)

            cv2.imwrite(filename, frame)
            current_image_collection.append(filename)
            
            if len(current_image_collection) > frame_count:
                file_to_remove = current_image_collection.pop(0)
                os.remove(file_to_remove)
            
            i += 1
            time.sleep(1)

        except:
            pass

        k = cv2.waitKey(10) & 0xFF
        if k==27:
            break

    print('Done')
    sys.stdout.flush()
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create frames from camera")
    
    parser.add_argument("--vid_path", type=int, default=0)
    parser.add_argument("--frame_width", type=int, default=331)
    parser.add_argument("--frame_height", type=int, default=331)
    parser.add_argument("--frame_count", type=int, default=8)

    args = parser.parse_args()

    out_path = "/temporal-shift-module/videos_rolling"

    dump_frames(
            vid_path = args.vid_path, 
            out_path = out_path, 
            frame_width = args.frame_width, 
            frame_height = args.frame_height, 
            frame_count = args.frame_count
    )
