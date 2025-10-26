import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Hide TensorFlow info messages

import cv2
import numpy as np
import threading
from playsound import playsound
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model("mask_detector.h5")  # Your trained H5 model

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start webcam
cap = cv2.VideoCapture(0)

# Function to play alert sound in a separate thread
def play_alert_sound():
    threading.Thread(target=playsound, args=("alert.mp3",), daemon=True).start()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        face_resized = cv2.resize(face, (150, 150))  # Must match model input size
        face_normalized = face_resized / 255.0
        face_reshaped = np.reshape(face_normalized, (1, 150, 150, 3))

        # Predict using the model
        result = model.predict(face_reshaped)[0][0]
        label = "Mask" if result < 0.5 else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

        # Play alert if no mask
        if label == "No Mask":
            play_alert_sound()

        # Draw rectangle and label
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, label, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow('Face Mask Detection', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
