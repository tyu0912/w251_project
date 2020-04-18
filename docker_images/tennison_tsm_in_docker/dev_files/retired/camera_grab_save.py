import cv2
import time
import argparse
import os
import sys
from datetime import datetime

def dump_frames(vid_path, out_path, frame_width, frame_height, frame_count):
    
    cap = cv2.VideoCapture(vid_path)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    folders = out_path.split("/")

    for i in range(len(folders)+1):
        name = '/'.join(folders[0:i])
        #print(name)

        try:
            os.mkdir(name)
        except OSError:
            pass

    i=0
    output_file = open(out_path + '/file.txt', 'w')
    
    while i <= frame_count:

        try:
            ret, frame = cap.read()

            cv2.imshow('frame', frame)

            cv2.imwrite('{}/{:06d}.jpg'.format(out_path, i), frame)
            access_path = '{}/{:06d}.jpg'.format(out_path, i)
            
            output_file.write(access_path + '\n')
    
            i += 1
            time.sleep(1)

        except:
            pass

        k = cv2.waitKey(10) & 0xFF

    print('Done')
    sys.stdout.flush()
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create frames from camera")
    
    parser.add_argument("--vid_path", type=int, default=0)
    parser.add_argument("--out_path", type=str, default=datetime.now().strftime("%Y-%m-%dT%H-%M-%S"))
    parser.add_argument("--label", type=str, required=True)
    parser.add_argument("--frame_width", type=int, default=331)
    parser.add_argument("--frame_height", type=int, default=331)
    parser.add_argument("--frame_count", type=int, default=8)

    args = parser.parse_args()

    out_path = "/temporal-shift-module/videos/{}/{}".format(args.out_path, args.label)

    dump_frames(
            vid_path = args.vid_path, 
            out_path = out_path, 
            frame_width = args.frame_width, 
            frame_height = args.frame_height, 
            frame_count = args.frame_count
    )
