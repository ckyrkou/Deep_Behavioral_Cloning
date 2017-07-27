import argparse
import base64
import json

import numpy as np
import socketio
import eventlet
import eventlet.wsgi
import time
from PIL import Image
from PIL import ImageOps
from flask import Flask, render_template
from io import BytesIO

from keras.models import load_model

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array

import time

# Fix error with Keras and TensorFlow
import tensorflow as tf
tf.python.control_flow_ops = tf
import cv2

sio = socketio.Server()
app = Flask(__name__)
model = None
prev_image_array = None

@sio.on('telemetry')

    
def telemetry(sid, data):
    global record,filename,out
    # The current steering angle of the car
    steering_angle = data["steering_angle"]
    # The current throttle of the car
    throttle = data["throttle"]
    # The current speed of the car
    speed = data["speed"]
    # The current image from the center camera of the car
    imgString = data["image"]
    image = Image.open(BytesIO(base64.b64decode(imgString)))
    image_array = np.asarray(image)
    transformed_image_array = cv2.resize(image_array,(80,40))[None, :, :, :]

    # This model currently assumes that the features of the model are just the images. Feel free to change this.
    steering_angle = float(model.predict(transformed_image_array, batch_size=1))
    
    if(record == 1):
        cv2.putText(image_array,"Steering Angle: "+str(steering_angle),(10,20),cv2.FONT_HERSHEY_SIMPLEX,0.35,(200,255,0),1)
        out.write(cv2.cvtColor(image_array,cv2.COLOR_BGR2RGB))
        cv2.imshow('Video',cv2.cvtColor(image_array,cv2.COLOR_BGR2RGB))
        
    
    # The driving model currently just outputs a constant throttle. Feel free to edit this.
    throttle = 0.2
	
	#MANAGING THROTTLE PEDAL
	#throttle = max(0.1, -0.15/0.05 * abs(steering_angle) + 0.35)
	#Managing the throttle pedal to control the speed according to the steering angle!
	#If you need to control the throttle pedal to get a constant and controlled speed you could use the following proportional control loop:

	#throttle = (DESIRED_SPEED-float(speed))*Kp

	#You can define a constant value for the desired speed and Kp is the proportionality constant also known as "loop gain". For Kp I've used 0.5 with good results. :)

	#Here you can find more information about control loops:

	#https://en.wikipedia.org/wiki/PID_controller

    print(steering_angle, throttle)
    send_control(steering_angle, throttle)
    k=cv2.waitKey(33)
    if(k & 0xFF ==ord('q')):
        out.release()
        record = 0
        cv2.destroyAllWindows()
        print('Recording Ended')

@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit("steer", data={
    'steering_angle': steering_angle.__str__(),
    'throttle': throttle.__str__()
    }, skip_sid=True)

record = 0
if(record == 1):
    filename = "./capture_videos/"+time.strftime("%H:%M:%S")+".avi"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename,fourcc,25,(320,160))


if(record == 1):
    print('Recording Enabled')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument('model', type=str,
    help='Path to model definition h5. Model should be on the same path.')
    args = parser.parse_args()

    model = load_model(args.model)
        
    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
    
