from flask import Flask, render_template, Response, jsonify, request
import cv2
import numpy as np

app = Flask(__name__)
camera = cv2.VideoCapture(0)

# Load models
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
age_model = cv2.dnn.readNetFromCaffe("age_deploy.prototxt", "age_net.caffemodel")
gender_model = cv2.dnn.readNetFromCaffe("gender_deploy.prototxt", "gender_net.caffemodel")

AGE_RANGES = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
GENDERS = ['Male', 'Female']

# Human detection state
human_detected = False

def get_skin_color(image, box):
    x, y, w, h = box
    face = image[y:y+h, x:x+w]
    avg_color_per_row = np.average(face, axis=0)
    avg_color = np.average(avg_color_per_row, axis=0)
    return 'Fair' if avg_color[2] > 130 else 'Dark'

def get_hair_color(image, box):
    x, y, w, h = box
    top = image[y:y+int(h/4), x:x+w]
    avg = np.mean(top, axis=(0, 1))
    if avg[0] > 100 and avg[2] > 100:
        return "Blonde"
    elif avg[2] > avg[0] and avg[2] > avg[1]:
        return "Red/Brown"
    else:
        return "Black"

def generate_frames():
    global human_detected
    while True:
        success, frame = camera.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.2, 5)

        # Update human detection state
        human_detected = len(faces) > 0

        for (x, y, w, h) in faces:
            face_img = frame[y:y+h, x:x+w].copy()
            blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227),
                                         (78.4263, 87.7689, 114.8958), swapRB=False)

            # Predict gender
            gender_model.setInput(blob)
            gender = GENDERS[gender_model.forward().argmax()]

            # Predict age
            age_model.setInput(blob)
            age = AGE_RANGES[age_model.forward().argmax()]

            skin = get_skin_color(frame, (x, y, w, h))
            hair = get_hair_color(frame, (x, y, w, h))

            label = f"{gender}, {age}, {skin} Skin, {hair} Hair"
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Encode to JPEG
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Send frame
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' +
               frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start')
def start():
    global camera
    if not camera.isOpened():
        camera.open(0)
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stop')
def stop():
    global camera
    camera.release()
    return "Camera Off"

@app.route('/refresh')
def refresh():
    global camera
    camera.release()
    camera.open(0)
    return "Camera Refreshed"

@app.route('/status')
def status():
    return jsonify({"human": human_detected})

import base64
import io
from PIL import Image

@app.route('/detect-image', methods=['POST'])
def detect_image():
    try:
        # Read image from POST request
        data = request.files['image']
        image = Image.open(data.stream).convert('RGB')
        image_np = np.array(image)
        image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.2, 5)

        is_human = len(faces) > 0
        return jsonify({'human': is_human})
    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True)
