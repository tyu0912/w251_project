## To run the model

docker built -t tsm_container .

docker run --rm --privileged --env DISPLAY=$DISPLAY -ti tsm_container bash
