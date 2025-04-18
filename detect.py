import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load trained emotion model
model = load_model("models/emotion_model.h5")

# Emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Load face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        # No face found — show message
        cv2.putText(frame, "No face detected", (30, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 2)
    else:
        for (x, y, w, h) in faces:
            # Extract face region
            roi_gray = gray[y:y+h, x:x+w]
            roi_gray = cv2.resize(roi_gray, (48, 48))
            roi_gray = roi_gray.astype("float") / 255.0
            roi_gray = np.reshape(roi_gray, (1, 48, 48, 1))

            # Predict emotion
            prediction = model.predict(roi_gray)
            emotion = emotion_labels[np.argmax(prediction)]

            # Draw rectangle and show emotion
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, f"Emotion: {emotion}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (0, 255, 0), 2)

    # Show the result
    cv2.imshow("Emotion Detection", frame)

    # Break loop on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
