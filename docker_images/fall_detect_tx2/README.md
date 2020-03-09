# w251-posture

## Jetson

### Running fall_detect_tx2 Docker container

#### Create directory structure for persistence
To run the fall_detect_tx2 image you'll first need to create the directory structure under `/data/fall_detect/` (which is in this zip file KEVIN TODO):
- `models/`
    - `COCO/`: with the checkpoint files (unzipped)
- `w251_project/`: the project repository cloned from https://github.com/tyu0912/w251_project
- `videos/`: to store training videos. TODO KEVIN TODO TENNISON Update this to IBM Cloud Store
- `video_processing_output/`: For all the output from video_preprocessor.py
    - `video_preprocessing/`: The coordinates TSV will be in this directory after processing is complete. TODO KEVIN
    - `frames/`: Frames with skeletons for sanity check.  This dir may get large.
- `camera_capture_output/`: For all the output from camera_capture.py


#### Run the container
With the correct data structure on your TX2

```bash
$ cd /data/fall_detect/w251_project/docker_images/fall_detect_tx2/
$ ./docker_run.sh
# Script includes xhost +
```
Or to use the image stored on Docker 
```bash
$ ./remote_docker_run.sh KEVIN TODO
```

### Running Video Preprocessor
This script is used to create training data from videos.

Put your training videos in `/data/fall_detect/videos/` on your TX2 (or `/usr/src/app/videos` inside the container)

```
$ export DEVICE=1 # If your camera is mounted as /dev/video1
python3 video_preprocessor.py --test_epoch 140 --device $DEVICE --video_dir ../videos
# TODO Remove need for --device arg
```
*Note:* If you receive an OOM, rerun the script.  I don't know why, but this has worked for me twice.  

The results will be output in to `/data/fall_detect/video_processing_output` on your TX2.


### Running Camera Capture
This script is used to capture and detect fall using your camera on your TX2.

```
$ export DEVICE=1 # If your camera is mounted as /dev/video1
python3 camera_capture.py --test_epoch 140 --device $DEVICE
```

The results will be output in to `/data/fall_detect/camera_capture_output` on your TX2.

Next:
* Add directory parameter to read in files
* Improve coordinate saving

