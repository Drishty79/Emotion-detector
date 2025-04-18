from flask import Flask, render_template, Response, jsonify, url_for
import cv2
import tensorflow as tf
import numpy as np
from keras.preprocessing.image import img_to_array
import os

app = Flask(__name__)

# Load the pre-trained emotion detection model
model_path = "models/emotion_model.h5"
model = tf.keras.models.load_model(model_path)

# Emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Open webcam
cap = cv2.VideoCapture(0)


def detect_emotion_from_face(face_img):
    """Detect emotion from a face image."""
    resized = cv2.resize(face_img, (48, 48))
    img_array = img_to_array(resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    emotion_index = np.argmax(prediction)
    return emotion_labels[emotion_index]


def generate_frames():
    """Generate frames for live webcam stream with emotion detection."""
    while True:
        success, frame = cap.read()
        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        if len(faces) == 0:
            # No face detected
            cv2.putText(frame, "No face detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 255), 2, cv2.LINE_AA)
        else:
            for (x, y, w, h) in faces:
                roi_gray = gray[y:y + h, x:x + w]
                emotion = detect_emotion_from_face(roi_gray)

                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 255, 0), 2, cv2.LINE_AA)

        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/detect_emotion', methods=['POST'])
def detect_emotion_api():
    """API to detect emotion for a face in the frame and return the associated song path."""
    success, frame = cap.read()
    if not success:
        return jsonify({'error': 'Failed to capture frame'})

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    if len(faces) == 0:
        return jsonify({'emotion': 'None', 'song': ''})

    # Use the first detected face
    (x, y, w, h) = faces[0]
    roi_gray = gray[y:y + h, x:x + w]
    emotion = detect_emotion_from_face(roi_gray)

    emotion_songs = {
        "Happy": "static/songs/happy_song.mp3",
        "Sad": "static/songs/sad_song.mp3",
        "Angry": "static/songs/angry_song.mp3",
        "Surprise": "static/songs/surprise_song.mp3",
        "Neutral": "static/songs/neutral_song.mp3",
        "Fear": "static/songs/fear_song.mp3",
        "Disgust": "static/songs/disgust_song.mp3"
    }

    song_path = emotion_songs.get(emotion, "")
    if song_path:
        song_path = url_for('static', filename=song_path.split('static/')[-1])

    return jsonify({'emotion': emotion, 'song': song_path})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
