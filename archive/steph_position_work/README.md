# w251-posture

## Jetson

To run any of the scripts, you must first build the Docker image:

```bash
# on the jetson
$ cd TF-SimpleHumanPose
$ docker build -t posture .
```

You must also allow Docker to access X Windows (for all scripts except jetson.py):
```bash
$ xhost +
```

Once the image is built you can run the following tasks. They should all be executed from the Jetson and in the `TF-SimpleHumanPose` directory.

### Data Collection

**Prerequisites**
* `output/model_dump` directory contains the model files
* set `$DEVICE` to the device id of the camera to use. On the Jetson 0 is the onboard camera, and 1 is the external camera.

```bash
$ docker run --privileged \
 -v `pwd`/output:/usr/src/app/output \
 -v /tmp/.X11-unix:/tmp/.X11-unix \
 -it \
 -e DISPLAY \
 posture python3 single_collect_sm.py --test_epoch 140 --device $DEVICE #Steph Edit
```

This will prompt the user with a picture every 5 minutes and the rendered keypoints. When the user closes the window it will prompt them either G or B for good or bad posture and save the data to a set of files.


------------------------------------------


# DNN Posture Detector

Shane Andrade, Rachael Burns, Thomas Drage

## Introduction

This code utilises a pose detection model, which is the **[TensorFlow](https://www.tensorflow.org)** implementation of **[Simple Baselines for Human Pose Estimation and Tracking (ECCV 2018)](https://arxiv.org/abs/1804.06208)** of MSRA for **2D multi-person pose estimation** from a single RGB image, which is originally available [here](https://github.com/mks0601/TF-SimpleHumanPose).

Camera captured data uses this model to return pose keypoints, from which the following postural features are calculated:

- nose_ratio	
- nose_shoulder_perp
- eye_shldr_angle	
- arm_angle_left	
- arm_angle_right	
- ear_eye_left	
- ear_eye_right

A random forest classifier is then used to determine whether a detected posture is "good", e.g. spine neutral or "bad" (slouching etc.). 
The classification and they keypoints are then sent to an MQTT broker.

## Dependencies
* [TensorFlow](https://www.tensorflow.org/)
* [CUDA](https://developer.nvidia.com/cuda-downloads)
* [cuDNN](https://developer.nvidia.com/cudnn)
* Python 3, pip3

## Directory

### Root
The `${POSE_ROOT}` is described as below.
```
${POSE_ROOT}
|-- lib
|-- analyse
|-- main
`-- output
```
* `lib` contains kernel codes for 2d multi-person pose estimation system (from TF-SimpleHumanPose source repo).
* `analyse` contains a Jupyter notebook for reading the posture feature data and training the posture classifier.
* `main` contains high-level code for collecting pose/posture feature data, testing the classification and running the edge to cloud pipeline.
* `output` contains log, trained models, visualized outputs, pose/posture feature data files.
* `localweb` contains a Flask web app for viewing the live pose detection and classification results as well as configuring the sensitivity threshold for posture classification and the current username.

Key scripts in *main* are:
- single.py - Script to test pose model using live camera input.
- single_collect.py - Data collection script for development of the posture classifier, takes an image, computes features, asks for classification and stores at set time interval.
- single_predict.py - Script to test posture classifier using live camera input.
- jetson.py - Main project script, captures camera data, performs pose detection inference, computes features, filters output, performs posture classification, interacts with local web interface, publishes data to cloud via MQTT.

Other scripts are:
- config.py - Pose detection model setup, from TTF-SimpleHumanPose source repo.
- dataset.py - COCO dataset model parameters, from TF-SimpleHumanPose source repo.
- model.py - Pose detection model class file, from TF-SimpleHumanPose source repo.
- gen_batch_single.py - A version adapted for this project of the original from the TF-SimpleHumanPose, which prepares image data for inference.

### Output
You need to follow the directory structure of the `output` folder as below.
```
${POSE_ROOT}
|-- output
|-- |-- model_dump
`-- |-- vis
```
* Creating `output` folder as soft link form is recommended instead of folder form because it would take large storage capacity.
* `model_dump` folder contains saved checkpoints for each epoch, which are [downloaded pre-trained](https://cv.snu.ac.kr/research/TF-SimpleHumanPose/COCO/model/256x192_resnet50_coco.zip) on the MSCOCO 2017 dataset.
* `vis` folder contains visualized results.
* You can change default directory structure of `output` by modifying `main/config.py`.

## Running TF-SimpleHumanPose
### Start
* Run `pip install -r jetson-requirements.txt` to install required modules.
* Run `cd ${POSE_ROOT}/lib` and `make` to build NMS modules.


## Acknowledgements
This repo is largely modified from the [TF Implementation of Simple Baselines for Human Pose Estimation and Tracking](https://github.com/mks0601/TF-SimpleHumanPose) which itself is evolved from [TensorFlow repo of CPN](https://github.com/chenyilun95/tf-cpn) and [PyTorch repo of Simple](https://github.com/Microsoft/human-pose-estimation.pytorch).

## Reference
[1] Xiao, Bin, Haiping Wu, and Yichen Wei. "Simple Baselines for Human Pose Estimation and Tracking". ECCV 2018.
