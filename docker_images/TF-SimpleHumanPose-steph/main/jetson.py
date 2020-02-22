#!/usr/bin/python3

###########################################
#
# Human desk posture detection system
# Data capture and classification service
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
import json
from PIL import Image
import paho.mqtt.client as mqtt

import math
import readchar

import joblib

from tfflat.base import Tester
from model import Model

from gen_batch_single import generate_batch

HOST = os.environ['BROKER']
PORT = 1883
USER = os.environ['USERNAME']
SENSITIVITY = 0.5


# Establish connection to local MQTT broker
def on_connect(client, userdata, flags, rc):
    if rc==0:
        client.connected_flag=True #set flag
        print("connected OK Returned code=",rc)
        #client.subscribe(topic)
    else:
        print("Bad connection Returned code= ",rc)

client = mqtt.Client('jetson')
print("Connecting to: ", HOST)
client.on_connect = on_connect
client.connected_flag=False
client.connect(HOST, PORT)
client.loop_start()
while not client.connected_flag: #wait in loop
     time.sleep(1)
client.loop_stop()


def test_net(tester, img):

    # Init output structures
    vec = np.zeros(7)
    kps = {}
    data = {}

    start_time = time.time()

    kps_result = np.zeros((cfg.num_kps, 3))

    d = {"img": img, 'bbox': [0, 0, img.shape[1], img.shape[0]]}

    trans_img, crop_info = generate_batch(d, stage='test')

    # Pose detection model forward step.
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
        kps_result[j, :2] = (x * cfg.input_shape[1] / cfg.output_shape[1],
                             y * cfg.input_shape[0] / cfg.output_shape[0])
        kps_result[j, 2] = hm_j.max() / 255

   # Map keypoints back to original images.
    for j in range(cfg.num_kps):
        kps_result[j, 0] = kps_result[j, 0] / cfg.input_shape[1] * \
            (crop_info[2] - crop_info[0]) + crop_info[0]
        kps_result[j, 1] = kps_result[j, 1] / cfg.input_shape[0] * \
            (crop_info[3] - crop_info[1]) + crop_info[1]

    # Something was detected.
    if np.any(kps_result[:, 2] > 0.9):
        tmpimg = img
        tmpimg = tmpimg.astype('uint8')

        tmpkps = np.zeros((3, cfg.num_kps))
        tmpkps[:2, :] = kps_result[:, :2].transpose(1, 0)
        tmpkps[2, :] = kps_result[:, 2]

        # Store the keypoints.
        kps["nose"] = {"x": tmpkps[0][0], "y": tmpkps[1][0], "w": tmpkps[2][0]}
        kps["eye_l"] = {"x": tmpkps[0][1],
                        "y": tmpkps[1][1], "w": tmpkps[2][1]}
        kps["eye_r"] = {"x": tmpkps[0][2],
                        "y": tmpkps[1][2], "w": tmpkps[2][2]}
        kps["ear_l"] = {"x": tmpkps[0][3],
                        "y": tmpkps[1][3], "w": tmpkps[2][3]}
        kps["ear_r"] = {"x": tmpkps[0][4],
                        "y": tmpkps[1][4], "w": tmpkps[2][4]}
        kps["shldr_l"] = {"x": tmpkps[0][5],
                          "y": tmpkps[1][5], "w": tmpkps[2][5]}
        kps["shldr_r"] = {"x": tmpkps[0][6],
                          "y": tmpkps[1][6], "w": tmpkps[2][6]}
        kps["elbw_l"] = {"x": tmpkps[0][7],
                         "y": tmpkps[1][7], "w": tmpkps[2][7]}
        kps["elbw_r"] = {"x": tmpkps[0][8],
                         "y": tmpkps[1][8], "w": tmpkps[2][8]}
        kps["wrst_l"] = {"x": tmpkps[0][9],
                         "y": tmpkps[1][9], "w": tmpkps[2][9]}
        kps["wrst_r"] = {"x": tmpkps[0][10],
                         "y": tmpkps[1][10], "w": tmpkps[2][10]}

        # print("\nNose \t{:.0f}\t{:.0f}\t{:.2f}".format(
        #     kps["nose"]["x"], kps["nose"]["y"], kps["nose"]["w"]))
        # print("L Eye \t{:.0f}\t{:.0f}\t{:.2f}".format(
        #     kps["eye_l"]["x"], kps["eye_l"]["y"], kps["eye_l"]["w"]))
        # print("R Eye \t{:.0f}\t{:.0f}\t{:.2f}".format(
        #     kps["eye_r"]["x"], kps["eye_r"]["y"], kps["eye_r"]["w"]))
        # print("L Ear \t{:.0f}\t{:.0f}\t{:.2f}".format(
        #     kps["ear_l"]["x"], kps["ear_l"]["y"], kps["ear_l"]["w"]))
        # print("R Ear \t{:.0f}\t{:.0f}\t{:.2f}".format(
        #     kps["ear_r"]["x"], kps["ear_r"]["y"], kps["ear_r"]["w"]))
        # print("L Shldr\t{:.0f}\t{:.0f}\t{:.2f}".format(
        #     kps["shldr_l"]["x"], kps["shldr_l"]["y"], kps["shldr_l"]["w"]))
        # print("R Shldr\t{:.0f}\t{:.0f}\t{:.2f}".format(
        #     kps["shldr_r"]["x"], kps["shldr_r"]["y"], kps["shldr_r"]["w"]))
        # print("L Elbw \t{:.0f}\t{:.0f}\t{:.2f}".format(
        #     kps["elbw_l"]["x"], kps["elbw_l"]["y"], kps["elbw_l"]["w"]))
        # print("R Elbw \t{:.0f}\t{:.0f}\t{:.2f}".format(
        #     kps["elbw_r"]["x"], kps["elbw_r"]["y"], kps["elbw_r"]["w"]))
        # print("L Wrist\t{:.0f}\t{:.0f}\t{:.2f}".format(
        #     kps["wrst_l"]["x"], kps["wrst_l"]["y"], kps["wrst_l"]["w"]))
        # print("R Wrist\t{:.0f}\t{:.0f}\t{:.2f}".format(
        #     kps["wrst_r"]["x"], kps["wrst_r"]["y"], kps["wrst_r"]["w"]))

        # Feature defaults
        nose_ratio = 99
        nose_shoulder_perp = 99
        eye_shldr_angle = 99
        arm_angle_left = 99
        arm_angle_right = 99
        ear_eye_left = 99
        ear_eye_right = 99

        shoulder_spacing = 0

        # Compute features

        if(kps["shldr_l"]["w"] > 0.4 and kps["shldr_r"]["w"] > 0.4 and kps["nose"]["w"] > 0.4 and kps["eye_l"]["w"] > 0.4 and kps["eye_r"]["w"] > 0.4):

            shoulder_mid = mid(kps["shldr_l"]["x"], kps["shldr_l"]
                               ["y"], kps["shldr_r"]["x"], kps["shldr_r"]["y"])
            nose_elevation = cdist(
                kps["nose"]["x"], kps["nose"]["y"], shoulder_mid[0], shoulder_mid[1])
            eye_spacing = cdist(
                kps["eye_l"]["x"], kps["eye_l"]["y"], kps["eye_r"]["x"], kps["eye_r"]["y"])

            nose_ratio = nose_elevation / eye_spacing

        #print("\nNose Angle Ratio\t{:.1f}".format(nose_ratio))

        if(kps["shldr_l"]["w"] > 0.4 and kps["shldr_r"]["w"] > 0.4 and kps["nose"]["w"] > 0.4 and kps["eye_l"]["w"] > 0.4 and kps["eye_r"]["w"] > 0.4):

            shoulder_spacing = cdist(
                kps["shldr_l"]["x"], kps["shldr_l"]["y"], kps["shldr_r"]["x"], kps["shldr_r"]["y"])
            shoulder_nose_left = cdist(
                kps["shldr_l"]["x"], kps["shldr_l"]["y"], kps["nose"]["x"], kps["nose"]["y"])
            shoulder_nose_right = cdist(
                kps["shldr_r"]["x"], kps["shldr_r"]["y"], kps["nose"]["x"], kps["nose"]["y"])

            nose_shoulder_perp = tri_height(
                shoulder_nose_left, shoulder_spacing, shoulder_nose_right) / eye_spacing

        #print("Nose Perp Angle Ratio\t{:.1f}".format(nose_shoulder_perp))

        if(kps["shldr_l"]["w"] > 0.4 and kps["shldr_r"]["w"] > 0.4 and kps["eye_l"]["w"] > 0.4 and kps["eye_r"]["w"] > 0.4):

            eye_slope = math.degrees(math.atan(
                (kps["eye_l"]["y"] - kps["eye_r"]["y"])/(kps["eye_l"]["x"] - kps["eye_r"]["x"])))
            shldr_slope = math.degrees(math.atan(
                (kps["shldr_l"]["y"] - kps["shldr_r"]["y"])/(kps["shldr_l"]["x"] - kps["shldr_r"]["x"])))

            eye_shldr_angle = eye_slope - shldr_slope

        #print("Eye Shldr Angle\t\t{:.1f}".format(eye_shldr_angle))

        if(kps["shldr_l"]["w"] > 0.4 and kps["shldr_r"]["w"] > 0.4 and kps["elbw_l"]["w"] > 0.4):

            arm_left = cdist(kps["shldr_l"]["x"], kps["shldr_l"]
                             ["y"], kps["elbw_l"]["x"], kps["elbw_l"]["y"])
            diag_left = cdist(kps["elbw_l"]["x"], kps["elbw_l"]
                              ["y"], kps["shldr_r"]["x"], kps["shldr_r"]["y"])

            arm_angle_left = cos_angle(arm_left, shoulder_spacing, diag_left)

        if(kps["shldr_l"]["w"] > 0.4 and kps["shldr_r"]["w"] > 0.4 and kps["elbw_r"]["w"] > 0.4):

            arm_right = cdist(kps["shldr_r"]["x"], kps["shldr_r"]
                              ["y"], kps["elbw_r"]["x"], kps["elbw_r"]["y"])
            diag_right = cdist(kps["elbw_r"]["x"], kps["elbw_r"]
                               ["y"], kps["shldr_l"]["x"], kps["shldr_l"]["y"])

            arm_angle_right = cos_angle(
                arm_right, shoulder_spacing, diag_right)

        # print("Left Arm Angle\t\t{:.1f}".format(arm_angle_left))
        # print("Right Arm Angle\t\t{:.1f}".format(arm_angle_right))

        if(kps["eye_l"]["w"] > 0.4 and kps["ear_l"]["w"]):

            ear_eye_left = math.degrees(math.atan(
                (kps["eye_l"]["y"] - kps["ear_l"]["y"])/(kps["eye_l"]["x"] - kps["ear_l"]["x"])))

        if(kps["eye_r"]["w"] > 0.4 and kps["ear_r"]["w"]):

            ear_eye_right = math.degrees(math.atan(
                (kps["eye_r"]["y"] - kps["ear_r"]["y"])/(kps["ear_r"]["x"] - kps["eye_r"]["x"])))
        #
        # print("Left E-E Angle\t\t{:.1f}".format(ear_eye_left))
        # print("Right E-E Angle\t\t{:.1f}".format(ear_eye_right))

        # Generate keypoint visualisation
        tmpimg = cfg.vis_keypoints(tmpimg, tmpkps)

        # Write jpg for local web interface
        cv2.imwrite(osp.join(cfg.vis_dir, 'live_tmp.jpg'), tmpimg)
        os.replace(osp.join(cfg.vis_dir, 'live_tmp.jpg'), osp.join(cfg.vis_dir, 'live.jpg'))


        # Return the posture features
        data = {'nose_ratio': nose_ratio,
                'nose_shoulder_perp': nose_shoulder_perp,
                'eye_shldr_angle': eye_shldr_angle,
                'arm_angle_left': arm_angle_left,
                'arm_angle_right': arm_angle_right,
                'ear_eye_left': ear_eye_left,
                'ear_eye_right': ear_eye_right}

        vec = np.array([nose_ratio, nose_shoulder_perp, eye_shldr_angle, arm_angle_left, arm_angle_right, ear_eye_left, ear_eye_right])
        nans = np.isnan(vec)
        vec[nans] = 99

    return data, kps, vec


# Geometry helpers for feature calcs

# Return cosine law angle
def cos_angle(a, b, c):

    ratio = (a**2 + b**2 - c**2)/(2*a*b)
    ratio = max(ratio, -1.0)
    ratio = min(ratio, 1.0)

    return math.degrees(math.acos(ratio))

# Perpendicular height of a triangle
def tri_height(a, b, c):

    return 0.5*(((a + b + c)*(b + c - a)*(a + b - c)*(a - b + c))**0.5)/b

# Line segment midpoint
def mid(x1, y1, x2, y2):

    return (x1/2 + x2/2, y1/2 + y2/2)

# Distance between points
def cdist(x1, y1, x2, y2):

    return ((x1 - x2)**2 + (y1 - y2)**2)**0.5


def test(test_model, device, frequency):

    # Load the posture classification model
    classifier = joblib.load('../analyse/saved_model.pkl')

    # Load the pose detection model
    tester = Tester(Model(), cfg)
    tester.load_weights(test_model)

    # Set up the video capture device
    cap = cv2.VideoCapture(device)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)

    qos = int(os.environ['MQTT_QOS'])



    global USER, SENSITIVITY


    # Filter structure initialisation
    last = 0
    vec_old = np.zeros(7)
    filter_vec = np.zeros(7)
    result_dict = {}

    # Lag filter, 5 frame tau
    alpha = 1-math.exp(-0.2)

    # Rolling average classification, 20s window
    result_bad = [0,0,0,0,0]
    result_bad_avg = 0

    # Average fps
    loops = 0
    last_fps = 0
    fps = 0

    # Main loop
    while True:
        # Grab an image
        ret, img = cap.read()

        if img is None:
            continue

        # Run the pose detection algorithm and calculate posture features
        data, kps, vec = test_net(tester, img)

        # Lag filter the feature vector
        if(np.nonzero(vec)):
            filter_vec = vec*(alpha) + vec_old*(1-alpha)
            vec_old = vec

        # Perform posture classification and store/save results & settings
        if(time.time() > last_fps + 4):

            result = classifier.predict_proba([filter_vec])[0]

            result_bad.pop(0)
            result_bad.append(result[0])

            result_bad_avg = np.sum(result_bad)/5

            print("Classification Avg: ", result_bad_avg)

            result_class = 'bad'

            if(result_bad_avg < SENSITIVITY):
                print('Good')
                result_class = 'good'
            else:
                print('Bad')

            result_dict = {'bad': result[0], 'good': result[1], 'avg_bad': result_bad_avg, 
                'class': result_class, 'fps': fps, 'sensitivity': SENSITIVITY, 'user': USER}

            with open(osp.join(cfg.vis_dir, 'live_temp.json'), 'w') as outfile:
                json.dump(result_dict, outfile)
            os.replace(osp.join(cfg.vis_dir, 'live_temp.json'), osp.join(cfg.vis_dir, 'live.json'))

            try:
                with open(osp.join(cfg.output_dir, 'jetson_config.json'), 'r') as infile:
                    settings = json.load(infile)
                    USER = settings["user"]
                    SENSITIVITY = 1 - float(settings["sensitivity"])
            except:
                print("No user settings file")

            interval = time.time() - last_fps
            fps = loops/interval

            print("FPS: ", fps)

            loops = 0
            last_fps = time.time()

        # Upload current results to the cloud via local MQTT broker
        if(time.time() > last + frequency):

            msg = dict()
            #payload_kps = json.dumps(kps)
            msg['keypoints'] = kps
            #payload_feat = json.dumps(data)
            msg['features'] = data


            #payload_result = json.dumps(result_dict)
            msg['class'] = result_dict

            msg['user'] = USER

            payload = json.dumps(msg)
            print(payload)
            # client.publish('jetson/webcam/posture/keypoints', payload_kps, qos)
            # client.publish('jetson/webcam/posture/features', payload_feat, qos)
            #client.publish('jetson/webcam/posture/class', payload_result, qos)
            global client
            client.publish('jetson/webcam/posture', payload, qos)
            print("PUBLISHED")
            last = time.time()

        loops = loops + 1

if __name__ == '__main__':
    def parse_args():
        parser = argparse.ArgumentParser()

        parser.add_argument('--test_epoch', type=str, dest='test_epoch')
        parser.add_argument('--device', type=int, dest='device')
        parser.add_argument('--frequency', type=int, dest='frequency') # seconds
        args = parser.parse_args()

        assert args.test_epoch, 'Test epoch is required.'
        assert args.device, 'Device number is required.'
        assert args.frequency, 'Frequency is required.'
        return args

    global args
    args = parse_args()
    test(int(args.test_epoch), int(args.device), int(args.frequency))
