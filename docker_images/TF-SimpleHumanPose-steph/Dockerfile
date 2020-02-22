FROM w251/cuda:dev-tx2-4.2_b158
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update
RUN apt-get install -y libhdf5-serial-dev hdf5-tools \
    libhdf5-dev zlib1g-dev zip libjpeg8-dev python3 \
    python3-pip python3-opencv libfreetype6-dev xterm \
    python3-tk python3-scipy nano


WORKDIR /usr/src/app

RUN pip3 install --pre --no-cache-dir \
    --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v42 \
    numpy==1.17.0 \
    Cython==0.29.13 \
    tensorflow-gpu==1.13.1+nv19.5
    

RUN pip3 install --pre --no-cache-dir \
    --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v42 \
    pandas \
    matplotlib

COPY lib ./lib/
RUN make -C lib/

COPY jetson-requirements.txt ./
RUN pip3 install --pre --no-cache-dir -r jetson-requirements.txt
COPY . ./

WORKDIR /usr/src/app/main
ENV NO_AT_BRIDGE=1 \
    MQTT_BROKER=broker \
    MQTT_QOS=0

RUN apt-get install -y mosquitto-clients speech-dispatcher

EXPOSE 1883

CMD ["python3", "jetson.py", "--test_epoch", "140", "--device", "1", "--frequency", "30"]
