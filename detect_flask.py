from flask import Flask, render_template, Response
import cv2
from time import sleep
import os
import random
import requests
import json

app = Flask(__name__)

@app.route("/")
def button():
    return render_template("button.html")  # Presents a HTML page with a button to take a picture

@app.route("/takepic")
def takepic():
    currentdir = os.getcwd()
    randomnumber = random.randint(1, 100)  # A random number is created for a query string used when presenting the picture taken, this is to avoid web browser caching of the image.

    # Capture image using the Surface Laptop camera
    cam = cv2.VideoCapture(0)
    ret, frame = cam.read()
    if ret:
        image_path = os.path.join(currentdir, "static/image.jpg")
        cv2.imwrite(image_path, frame)
    cam.release()

    url = os.environ["FACE_ENDPOINT"]  # Replace with the Azure Cognitive Services endpoint for the Face API
    key = os.environ["FACE_APIKEY"]  # Azure Cognitive Services key
    image_data = open(image_path, "rb").read()
    headers = {
        "Ocp-Apim-Subscription-Key": key,
        "Content-Type": "application/octet-stream",
    }
    params = {
        "returnFaceId": "false",
        "returnFaceLandmarks": "false",
        "returnFaceAttributes": "age,gender,headPose,smile,facialHair,glasses,emotion,hair,makeup,occlusion,accessories,blur,exposure,noise",
    }
    r = requests.post(url, headers=headers, params=params, data=image_data)  # Submit to Azure Cognitive Services Face API
    face_data = r.json()[0]["faceAttributes"]
    age = face_data["age"]
    gender = face_data["gender"]
    haircolor = face_data["hair"]["hairColor"][0]["color"]
    emotions = face_data["emotion"]

    return render_template("FaceAnalysis.html", age=age, gender=gender, haircolor=haircolor, emotions=emotions, number=randomnumber)

def gen_frames():
    camera = cv2.VideoCapture(0)  # Use the first webcam
    while True:
        success, frame = camera.read()  # Read the camera frame
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/livestream')
def livestream():
    return render_template('button.html')

if __name__ == "__main__":
    app.run(port=80, host="0.0.0.0")
