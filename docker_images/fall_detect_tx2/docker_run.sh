xhost +
docker run --privileged \
	-v /data/fall_detect/video_processing_output/:/usr/src/app/video_processing_output \
	-v /data/fall_detect/videos:/usr/src/app/videos \
	-v /data/fall_detect/frames:/usr/src/app/frames \
	-v /tmp/.X11-unix:/tmp/.X11-unix \
	-v /data/fall_detect/video_preprocessing/ \
	-v /data/fall_detect/models/COCO:/usr/src/app/output/model_dump/COCO/ \
	-v /data/fall_detect/w251_project/docker_images/fall_detect_tx2/main:/usr/src/app/main \
	-it \
      	-e DISPLAY \
       	fall_detect_tx2 \
       	bash
