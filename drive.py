import argparse
import base64
from io import BytesIO
import numpy as np
from flask_socketio import SocketIO
from PIL import Image
from flask import Flask
from keras.models import load_model
import utils

app = Flask(__name__)
sio = SocketIO(app)

model = None
prev_image_array = None

MAX_SPEED = 25
MIN_SPEED = 10

speed_limit = MAX_SPEED

class PIController:
    def __init__(self, Kp, Ki):
        self.Kp = Kp
        self.Ki = Ki
        self.set_point = 0.
        self.error = 0.
        self.integral = 0.

    def set_desired(self, desired):
        self.set_point = desired

    def update(self, measurement):
        # proportional error
        self.error = self.set_point - measurement

        # integral error
        self.integral += self.error

        return self.Kp * self.error + self.Ki * self.integral
    
controller = PIController(0.1, 0.002)
controller.set_desired(20)
    



@sio.on('telemetry')
def telemetry(data):
    if data:
        steering_angle = float(data["steering_angle"])
        throttle = float(data["throttle"])
        speed = float(data["speed"])
        image = Image.open(BytesIO(base64.b64decode(data["image"])))
        try:
            image = np.asarray(image)
            image = utils.preprocess(image)
            image = np.array([image])

            steering_angle = float(model.predict(image, batch_size=1))

            global speed_limit
            if speed > speed_limit:
                speed_limit = MIN_SPEED
            else:
                speed_limit = MAX_SPEED
            throttle = controller.update(float(speed))

            print('{} {} {}'.format(steering_angle, throttle, speed))
            send_control(steering_angle, throttle)
        except Exception as e:
            print(e)

    else:        
        sio.emit('manual', data={}, skip_sid=True)


@sio.on('connect')
def connect():
    print("connected")
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit(
        "steer",
        data={
            'steering_angle': steering_angle.__str__(),
            'throttle': throttle.__str__()
        },
        skip_sid=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument('-m', help='path to model h5 file', dest='model', type=str, default='model/model-025.h5')
    args = parser.parse_args()                 

    model = load_model(args.model)

    sio.run(app, host='127.0.0.1', port=4567)
