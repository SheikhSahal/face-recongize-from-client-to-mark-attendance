import face_recognition
import numpy as np
from datetime import datetime
import os
import cv2
from flask import Flask, render_template, Response, request, redirect, url_for, jsonify
import csv
import dlib
import base64

app = Flask(__name__)

# Global variables
path = "ImagesAttendance"
only_name = "only_name"
attend_csv_path = "Attendance.csv"
screenshots_path = "screen/"
images = []
image_names = []
encodeListKnown = []
attend_dict = {}

# Initialize the dlib face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Ensure screenshots directory exists
if not os.path.exists(screenshots_path):
    os.makedirs(screenshots_path)

# Clean known images folder
def clean():
    for f in os.listdir(only_name):
        os.remove(os.path.join(only_name, f))

# Access images in folder
def access():
    global images, image_names
    mylist = os.listdir(path)
    for cl in mylist:
        curImg = cv2.imread(f'{path}/{cl}')
        images.append(curImg)
        image_names.append(os.path.splitext(cl)[0])
    print(image_names)

# Return the 128-dimension face encoding for each face in the image
def find_encodings(images):
    encodeList = []
    for image in images:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(image)
        if encode:
            encodeList.append(encode[0])
    return encodeList

# Save the screenshot when attendance is marked
def save_screenshot(img, name):
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    screenshot_name = f"{name}_{timestamp}.jpg"
    cv2.imwrite(os.path.join(screenshots_path, screenshot_name), img)

# Mark attendance in CSV file
def markAttendance(name, img):
    print(name, "attended")
    now = datetime.now()
    dtString = now.strftime("%I:%M %p")
    
    if name not in attend_dict:
        attend_dict[name] = [dtString, ""]
    else:
        attend_dict[name][1] = dtString

    save_screenshot(img, name)  # Save the screenshot

    with open(attend_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["NAME", "ENTRY", "EXIT", "TIME_SPENT_IN_MIN"])
        for name, times in attend_dict.items():
            if name not in ["NAME", "UNKNOWN"]:
                entry_time = times[0]
                exit_time = times[1]
                time_spent = ""
                if entry_time and exit_time:
                    entry_hour, entry_minute = map(int, entry_time[:-3].split(':'))
                    exit_hour, exit_minute = map(int, exit_time[:-3].split(':'))
                    if "PM" in entry_time and "AM" in exit_time:
                        time_spent = ((12 - entry_hour) * 60 + entry_minute) + (exit_hour * 60 + exit_minute)
                    else:
                        time_spent = (exit_hour * 60 + exit_minute) - (entry_hour * 60 + entry_minute)
                writer.writerow([name, entry_time, exit_time, time_spent])

# Decode base64 image
def decode_base64_image(data):
    data = data.split(',')[1]
    img_data = base64.b64decode(data)
    np_arr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return img

# Route to upload image
@app.route('/upload_image', methods=['POST'])
def upload_image():
    data = request.get_json()
    image_data = data['image']
    img = decode_base64_image(image_data)
    
    # Perform face recognition
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    name = "UNKNOWN"
    if encodesCurFrame:
        encodeFace = encodesCurFrame[0]
        min_dist = float("inf")
        for known_encode, known_name in zip(encodeListKnown, image_names):
            face_distance = face_recognition.face_distance([known_encode], encodeFace)
            if face_distance < min_dist and face_distance < 0.5:  # Adjust threshold as necessary
                min_dist = face_distance
                name = known_name.upper()

        if name != "UNKNOWN":
            markAttendance(name, img)
    
    return jsonify({"name": name})

# Routes
@app.route('/')
def index():
    return render_template('index.html')

if __name__ == "__main__":
    clean()
    access()
    encodeListKnown = find_encodings(images)
    print("Encoding Completed..")
    app.run(debug=True, port=5001)  # Change port if necessary
