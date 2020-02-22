# docker build -t orangelabschina/pose_estimation_arm64:cpu-v2.0 .

FROM orangelabschina/pose_estimation_arm64:cpu-manylibs

COPY . /workplace/tf-openpose/

WORKDIR /workplace/tf-openpose/

EXPOSE 8080 

#ENTRYPOINT ["python3", "run_stream_flaskvideo.py", "--model=mobilenet_thin", "--resize=208x176", "--image=./images/p1.jpg"]
ENTRYPOINT ["python3", "run_webcam.py", "--camera=0"]
