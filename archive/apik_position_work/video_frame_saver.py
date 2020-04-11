import cv2
import time
vidcap = cv2.VideoCapture('./frames/chute04_cam8_1s.m4v')
success,image = vidcap.read()
count = 10
offset = 0
while success:
    success,image = vidcap.read()
    if offset % 3 == 0:
        cv2.imwrite("./frames/frame%d.jpg" % count, image)     # save frame as JPEG file      
        print('Read a new frame: ', success)
        count += 1
    offset += 1