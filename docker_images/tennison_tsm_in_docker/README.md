## To run the model

docker built -t tsm_container .

#docker run --rm --privileged --env DISPLAY=$DISPLAY -ti tsm_container bash

sudo docker run --privileged -v $(pwd)/dev_files:/temporal-shift-module/dev_files -v /dev/bus/usb:/dev/bus/usb -v /tmp:/tmp -e QT_X11_NO_MITSHM=1 -e DISPLAY=$DISPLAY -it tsm_test1 bash
