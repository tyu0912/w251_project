xhost +
docker run --privileged \
	-v /data/fall_detect/output/:/usr/src/app/output \
	-v /data/fall_detect/videos:/usr/src/app/main/videos \
	-v /data/fall_detect/frames:/usr/src/app/main/frames \
	-v /tmp/.X11-unix:/tmp/.X11-unix \
	-v /data/fall_detect/video_preprocessing/ \
	-v /data/fall_detect/models/COCO:/usr/src/app/output/model_dump/COCO/ \
	-it \
      	-e DISPLAY \
       	fall_detect_tx2 \
       	bash
