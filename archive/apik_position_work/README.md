# w251-posture

## Jetson

To run any of the scripts, you must first build the Docker image:

From TY
make a folder called: `/posenet`

```bash
# on the jetson
$ cd TF-SimpleHumanPose
$ docker build -t posture .
```

You must also allow Docker to access X Windows (for all scripts except jetson.py):
```bash
$ xhost +
```

## Apik Additions

After building, run the following:

```
docker run --privileged  -v `pwd`/output:/usr/src/app/output  -v `pwd`/videos:/usr/src/app/main/videos -v `pwd`:/usr/src/app/ -v `pwd`/frames:/usr/src/app/main/frames  -v /tmp/.X11-unix:/tmp/.X11-unix  -v /posenet:/posenet-out  -it  -e DISPLAY  
posture bash
```

Once this is run, you can edit whatever you want inside the docker container. In `single_collect_sm.py`, make sure `test()` is commented out and make sure `capture_frames` is uncommented. Running `capture_frames` will read in a video file and save the frames, as well as the frame with the lines drawn on it, in `/frames/` directory.

```
python3 single_collect_sm.py --test_epoch 140 --device $DEVICE --video_dir videos
```

Next:
* Add directory parameter to read in files
* Improve coordinate saving

