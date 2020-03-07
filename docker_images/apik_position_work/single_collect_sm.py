###########################################
#
# Human desk posture detection system
# Classifier training data collection script
#
# Thomas Drage, Shane Andrade, Rachael Burns
# August 2019
#
# Copyright, all rights reserved.
# (With exception of components originally from: https://github.com/mks0601/TF-SimpleHumanPose)
#
###########################################


import os
import os.path as osp
import numpy as np
import argparse
from config import cfg
import cv2
import sys
import time
from datetime import datetime
import json
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from matplotlib.animation import FuncAnimation

matplotlib.use('tkagg')

import pickle

import math
import readchar
import pandas as pd

import tensorflow as tf

from tfflat.base import Tester
from tfflat.utils import mem_info
from model import Model

from gen_batch_single import generate_batch
from nms.nms import oks_nms


f = open("/posenet-out/camera-{t}.txt".format(t=str(datetime.now().strftime("%Y%m%d%H%m%S"))), 'a+')
#f = open("./camera-{t}.txt".format(t=str(datetime.now().strftime("%Y%m%d%H%m%S"))), 'a+')

headers = ["time","nose-x", "nose-y", "nose-w",
            "eye_l-x", "eye_l-y", "eye_l-w",
            "eye_r-x", "eye_r-y", "eye_r-w",
            "ear_l-x", "ear_l-y", "ear_l-w",
            "ear_r-x", "ear_r-y", "ear_r-w",
            "shldr_l-x", "shldr_l-y", "shldr_l-w",
            "shldr_r-x", "shldr_r-y", "shldr_r-w",
            "elbw_l-x", "elbw_l-y", "elbw_l-w",
            "elbw_r-x", "elbw_r-y", "elbw_r-w",
            "wrst_l-x", "wrst_l-y", "wrst_l-w",
            "wrst_r-x", "wrst_r-y", "wrst_r-w", 
            "shoulder_mid",  "nose_elevation",  "eye_spacing", "nose_ratio",  
            "shoulder_spacing", "shoulder_nose_left",  "shoulder_nose_right", "nose_shoulder_perp"
             "eye_slope",  "shldr_slope",  "eye_shldr_angle", "arm_left",  "diag_left", "arm_angle_left",  
             "arm_right",  "diag_right",  "arm_angle_right", "ear_eye_left",  "ear_eye_right"]
             
f.write('\t'.join(headers) + '\n')

start_time = time.time()

def test_net(tester, img):

    dump_results = []

    inference_time = time.time()

    kps_result = np.zeros((cfg.num_kps, 3))

    d = {"img": img, 'bbox' : [0,0,img.shape[1],img.shape[0]]}    

    try:
        trans_img, crop_info = generate_batch(d, stage='test')
    except:
        print("Error Here")
        print(trans_img)    

    # forward
    heatmap = tester.predict_one([[trans_img]])[0]
            
    for j in range(cfg.num_kps):
        hm_j = heatmap[0, :, :, j]
        idx = hm_j.argmax()
        y, x = np.unravel_index(idx, hm_j.shape)
                    
        px = int(math.floor(x + 0.5))
        py = int(math.floor(y + 0.5))
        if 1 < px < cfg.output_shape[1]-1 and 1 < py < cfg.output_shape[0]-1:
            diff = np.array([hm_j[py][px+1] - hm_j[py][px-1],
                hm_j[py+1][px]-hm_j[py-1][px]])
            diff = np.sign(diff)
            x += diff[0] * .25
            y += diff[1] * .25
        kps_result[j, :2] = (x * cfg.input_shape[1] / cfg.output_shape[1], y * cfg.input_shape[0] / cfg.output_shape[0])
        kps_result[j, 2] = hm_j.max() / 255 

   # map back to original images
    for j in range(cfg.num_kps):
       kps_result[j, 0] = kps_result[j, 0] / cfg.input_shape[1] * (crop_info[2] - crop_info[0]) + crop_info[0]
       kps_result[j, 1] = kps_result[j, 1] / cfg.input_shape[0] * (crop_info[3] - crop_info[1]) + crop_info[1]
                
    tmpimg = img.astype('uint8')
    tmpkps = np.zeros((3,cfg.num_kps))
    tmpkps[:2,:] = kps_result[:, :2].transpose(1,0)
    tmpkps[2,:] = kps_result[:, 2]
    
    if np.any(kps_result[:,2] > 0.9):        

        #tmpkps = np.zeros((3,cfg.num_kps))
        #tmpkps[:2,:] = kps_result[:, :2].transpose(1,0)
        #tmpkps[2,:] = kps_result[:, 2]

        kps = {}
        kps["nose"] = {"x": tmpkps[0][0], "y": tmpkps[1][0], "w": tmpkps[2][0]}
        kps["eye_l"] = {"x": tmpkps[0][1], "y": tmpkps[1][1], "w": tmpkps[2][1]}
        kps["eye_r"] = {"x": tmpkps[0][2], "y": tmpkps[1][2], "w": tmpkps[2][2]}
        kps["ear_l"] = {"x": tmpkps[0][3], "y": tmpkps[1][3], "w": tmpkps[2][3]}
        kps["ear_r"] = {"x": tmpkps[0][4], "y": tmpkps[1][4], "w": tmpkps[2][4]}
        kps["shldr_l"] = {"x": tmpkps[0][5], "y": tmpkps[1][5], "w": tmpkps[2][5]}
        kps["shldr_r"] = {"x": tmpkps[0][6], "y": tmpkps[1][6], "w": tmpkps[2][6]}
        kps["elbw_l"] = {"x": tmpkps[0][7], "y": tmpkps[1][7], "w": tmpkps[2][7]}
        kps["elbw_r"] = {"x": tmpkps[0][8], "y": tmpkps[1][8], "w": tmpkps[2][8]}
        kps["wrst_l"] = {"x": tmpkps[0][9], "y": tmpkps[1][9], "w": tmpkps[2][9]}
        kps["wrst_r"] = {"x": tmpkps[0][10], "y": tmpkps[1][10], "w": tmpkps[2][10]}

        print("\nNose \t{:.0f}\t{:.0f}\t{:.2f}".format(kps["nose"]["x"], kps["nose"]["y"], kps["nose"]["w"]))
        print("L Eye \t{:.0f}\t{:.0f}\t{:.2f}".format(kps["eye_l"]["x"], kps["eye_l"]["y"], kps["eye_l"]["w"])) 
        print("R Eye \t{:.0f}\t{:.0f}\t{:.2f}".format(kps["eye_r"]["x"], kps["eye_r"]["y"], kps["eye_r"]["w"]))
        print("L Ear \t{:.0f}\t{:.0f}\t{:.2f}".format(kps["ear_l"]["x"], kps["ear_l"]["y"], kps["ear_l"]["w"])) 
        print("R Ear \t{:.0f}\t{:.0f}\t{:.2f}".format(kps["ear_r"]["x"], kps["ear_r"]["y"], kps["ear_r"]["w"])) 
        print("L Shldr\t{:.0f}\t{:.0f}\t{:.2f}".format(kps["shldr_l"]["x"], kps["shldr_l"]["y"], kps["shldr_l"]["w"]))  
        print("R Shldr\t{:.0f}\t{:.0f}\t{:.2f}".format(kps["shldr_r"]["x"], kps["shldr_r"]["y"], kps["shldr_r"]["w"])) 
        print("L Elbw \t{:.0f}\t{:.0f}\t{:.2f}".format(kps["elbw_l"]["x"], kps["elbw_l"]["y"], kps["elbw_l"]["w"]))  
        print("R Elbw \t{:.0f}\t{:.0f}\t{:.2f}".format(kps["elbw_r"]["x"], kps["elbw_r"]["y"], kps["elbw_r"]["w"]))
        print("L Wrist\t{:.0f}\t{:.0f}\t{:.2f}".format(kps["wrst_l"]["x"], kps["wrst_l"]["y"], kps["wrst_l"]["w"]))  
        print("R Wrist\t{:.0f}\t{:.0f}\t{:.2f}".format(kps["wrst_r"]["x"], kps["wrst_r"]["y"], kps["wrst_r"]["w"])) 

        nose_ratio = 99
        nose_shoulder_perp = 99
        eye_shldr_angle = 99
        arm_angle_left = 99
        arm_angle_right = 99
        ear_eye_left = 99
        ear_eye_right = 99

        if(kps["shldr_l"]["w"] > 0.4 and kps["shldr_r"]["w"] > 0.4 and kps["nose"]["w"] > 0.4 and kps["eye_l"]["w"] > 0.4 and kps["eye_r"]["w"] > 0.4):

            shoulder_mid = mid(kps["shldr_l"]["x"], kps["shldr_l"]["y"], kps["shldr_r"]["x"], kps["shldr_r"]["y"])
            nose_elevation = cdist(kps["nose"]["x"], kps["nose"]["y"], shoulder_mid[0], shoulder_mid[1])
            eye_spacing = cdist(kps["eye_l"]["x"], kps["eye_l"]["y"], kps["eye_r"]["x"], kps["eye_r"]["y"])
 
            kps["shoulder_mid"] = shoulder_mid
            kps["nose_elevation"] = nose_elevation
            kps["eye_spacing"] = eye_spacing

            nose_ratio = nose_elevation / eye_spacing
            kps["nose_ratio"] = nose_ratio
    
        print("\nNose Angle Ratio\t{:.1f}".format(nose_ratio)) 
          
        if(kps["shldr_l"]["w"] > 0.4 and kps["shldr_r"]["w"] > 0.4 and kps["nose"]["w"] > 0.4 and kps["eye_l"]["w"] > 0.4 and kps["eye_r"]["w"] > 0.4):

            shoulder_spacing = cdist(kps["shldr_l"]["x"], kps["shldr_l"]["y"], kps["shldr_r"]["x"], kps["shldr_r"]["y"])
            shoulder_nose_left = cdist(kps["shldr_l"]["x"], kps["shldr_l"]["y"], kps["nose"]["x"], kps["nose"]["y"])
            shoulder_nose_right = cdist(kps["shldr_r"]["x"], kps["shldr_r"]["y"], kps["nose"]["x"], kps["nose"]["y"])

            kps["shoulder_spacing"] = shoulder_spacing
            kps["shoulder_nose_left"] = shoulder_nose_left
            kps["shoulder_nose_right"] = shoulder_nose_right
            
            nose_shoulder_perp = tri_height(shoulder_nose_left, shoulder_spacing, shoulder_nose_right) / eye_spacing
            kps["nose_shoulder_perp"] = nose_shoulder_perp

        print("Nose Perp Angle Ratio\t{:.1f}".format(nose_shoulder_perp)) 

        if(kps["shldr_l"]["w"] > 0.4 and kps["shldr_r"]["w"] > 0.4 and kps["eye_l"]["w"] > 0.4 and kps["eye_r"]["w"] > 0.4):

            eye_slope = math.degrees(math.atan((kps["eye_l"]["y"] - kps["eye_r"]["y"])/(kps["eye_l"]["x"] - kps["eye_r"]["x"])))
            shldr_slope = math.degrees(math.atan((kps["shldr_l"]["y"] - kps["shldr_r"]["y"])/(kps["shldr_l"]["x"] - kps["shldr_r"]["x"])))

            kps["eye_slope"] = eye_slope
            kps["shldr_slope"] = shldr_slope
            
            eye_shldr_angle = eye_slope - shldr_slope
            kps["eye_shldr_angle"] = eye_shldr_angle

        print("Eye Shldr Angle\t\t{:.1f}".format(eye_shldr_angle))
 
        if(kps["shldr_l"]["w"] > 0.4 and kps["shldr_r"]["w"] > 0.4 and kps["elbw_l"]["w"] > 0.4):

            arm_left = cdist(kps["shldr_l"]["x"], kps["shldr_l"]["y"], kps["elbw_l"]["x"], kps["elbw_l"]["y"])
            diag_left = cdist(kps["elbw_l"]["x"], kps["elbw_l"]["y"], kps["shldr_r"]["x"], kps["shldr_r"]["y"])

            kps["arm_left"] = arm_left
            kps["diag_left"] = diag_left

            arm_angle_left = cos_angle(arm_left, shoulder_spacing, diag_left)
            
            kps["arm_angle_left"] = arm_angle_left


        if(kps["shldr_l"]["w"] > 0.4 and kps["shldr_r"]["w"] > 0.4 and kps["elbw_r"]["w"] > 0.4):

            arm_right = cdist(kps["shldr_r"]["x"], kps["shldr_r"]["y"], kps["elbw_r"]["x"], kps["elbw_r"]["y"])
            diag_right = cdist(kps["elbw_r"]["x"], kps["elbw_r"]["y"], kps["shldr_l"]["x"], kps["shldr_l"]["y"])

            arm_angle_right = cos_angle(arm_right, shoulder_spacing, diag_right)

            kps["arm_right"] = arm_right
            kps["diag_right"] = diag_right
            kps["arm_angle_right"] = arm_angle_right

        print("Left Arm Angle\t\t{:.1f}".format(arm_angle_left))
        print("Right Arm Angle\t\t{:.1f}".format(arm_angle_right))

        if(kps["eye_l"]["w"] > 0.4 and kps["ear_l"]["w"]):       

            ear_eye_left = math.degrees(math.atan((kps["eye_l"]["y"] - kps["ear_l"]["y"])/(kps["eye_l"]["x"] - kps["ear_l"]["x"])))

            kps["ear_eye_left"] = ear_eye_left

        if(kps["eye_r"]["w"] > 0.4 and kps["ear_r"]["w"]):       

            ear_eye_right = math.degrees(math.atan((kps["eye_r"]["y"] - kps["ear_r"]["y"])/(kps["ear_r"]["x"] - kps["eye_r"]["x"])))

            kps["ear_eye_right"] = ear_eye_right

        print("Left E-E Angle\t\t{:.1f}".format(ear_eye_left))
        print("Right E-E Angle\t\t{:.1f}".format(ear_eye_right))

        tmpimg = cfg.vis_keypoints(tmpimg, tmpkps)

        out = []
        for h in headers:
            keys = h.split("-")
            if len(keys) > 1:
                out.append(str(kps.get(keys[0],  None).get(keys[1], None)))
            else:
                out.append(str(kps.get(keys[0],  None)))
    
        f.write(str(inference_time - start_time) + '\t' + '\t'.join(out) + '\n')
    return tmpimg, tmpkps


def update(i):
    im1.set_data(ax.imshow(tmpimg[:,:,[2,1,0]]))
    

def cos_angle(a, b, c):

    return math.degrees(math.acos((a**2 + b**2 - c**2)/(2*a*b)))

def tri_height(a,b,c):

    return 0.5*(((a + b + c)*(b + c - a)*(a + b - c)*(a - b + c))**0.5)/b

def mid(x1, y1, x2, y2):

    return (x1/2 + x2/2, y1/2 + y2/2)

def cdist(x1, y1, x2, y2):

    return ((x1 - x2)**2 + (y1 - y2)**2)**0.5

# def test(test_model, device):

#     print("Current dir")
#     print(os.getcwd())

#     to_csv = []
#     headers = set()

#     tester = Tester(Model(), cfg)
#     tester.load_weights(test_model)
    
#     device = 0
#     cap = cv2.VideoCapture(1)
    
    
#     cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
#     cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)

#     last = 0

#     while True:

#         ret, img = cap.read() # Apik
        
#         if(time.time() > last + 0.05):
            
    
#             tmpimg, tmpkps = test_net(tester, img)

#             cv2.namedWindow('vis',0)
#             cv2.imshow('vis', tmpimg[:,:,[2,1,0]])
#             k=cv2.waitKey(10) & 0xFF
            
        
#         last = time.time()

def capture_frames(test_model, device):
    import os
    print("I'm in test. Current dir = ")
    print(os.getcwd())
    #vidcap = cv2.VideoCapture('chute04_cam8_1s.m4v')
    vidcap = cv2.VideoCapture('frames/apik.m4v')

    success,image = vidcap.read()
    count = 10
    offset = 0
    if success:
        print("Success")
    else:
        print("Failed")
    to_csv = []
    headers = set()

    tester = Tester(Model(), cfg)
    tester.load_weights(test_model)

    while success:
        success,image = vidcap.read()
        if offset % 3 == 0:
            tmpimg, tmpkps = test_net(tester, image)
            cv2.imwrite("./frames/pose_frame%d.jpg" % count, image)     # save frame as JPEG file 
            cv2.imwrite("./frames/pose_frame_coords%d.jpg" % count, tmpimg)     # save frame with coordinates
#            cv2.imwrite("./frame%d.jpg" % count, image)     # save frame as JPEG file

            print('Read a new frame: ', success)
            count += 1
        offset += 1

if __name__ == '__main__':
    def parse_args():
        parser = argparse.ArgumentParser()

        parser.add_argument('--test_epoch', type=str, dest='test_epoch')
        parser.add_argument('--device', type=int, dest='device')
        args = parser.parse_args()
        
        assert args.test_epoch, 'Test epoch is required.'
        assert args.device, 'Device number is required.'
        return args


    global args
    args = parse_args()
    #test(int(args.test_epoch), int(args.device))
    capture_frames(int(args.test_epoch), int(args.device)) #AZ
