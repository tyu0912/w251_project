#!/usr/bin/env python

################
#
# Flask web app for Posture Detector local configuration interface.
# Thomas Drage, August 2019
#
################


from flask import Flask, render_template, Response, request
from time import sleep
import json

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

# Generate video data by reading jpg file
def gen():
    while True:
        frame = open("../output/vis/COCO/live.jpg", 'rb').read()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        sleep(0.1)

# Serve video stream
@app.route('/video_feed')
def video_feed():
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# Serve classification results JSON
@app.route('/results')
def results():
    f = open("../output/vis/COCO/live.json", 'r').read()
    return Response(f, mimetype='application/json')

# Save settings to JSON
@app.route('/set', methods=['POST'])
def set():
        print(request.json)
        f = open("../output/jetson_config.json", 'w')
        json.dump(request.json, f)
        f.close()

        return Response()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80, debug=True)
