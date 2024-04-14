import cv2
import os
import numpy as np
from datetime import datetime
from tensorflow.keras.models import load_model

# Load the saved pre-trained model
model = load_model('saved_model/s_model.h5')

# Load prebuilt classifier for Frontal Face detection
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)

# Font style
font = cv2.FONT_HERSHEY_SIMPLEX

# Create or append to a log file
def assure_path_exists(path):
    if not path:
        print("Invalid path specified.")
        return

    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        try:
            os.makedirs(dir)
        except OSError as e:
            print(f"Error creating directory: {e}")

# Example usage
def get_log_file_path(name):
    return f'identification_log_{name}.txt'

def write_log(log_file, message):
    with open(log_file, 'a') as file:
        file.write(message)

def preprocess_image(image):
    rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    resized_image = cv2.resize(rgb_image, (224, 224))
    return resized_image.astype(np.float32) / 255.0

# Video file path
video_file = 'test_video.mp4'

# Open the video file
cap = cv2.VideoCapture(video_file)

# Dictionary to store last log time for each individual
last_log_time = {}

while(cap.isOpened()):
    ret, frame = cap.read()
    if not ret:
        break

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, 1.2, 5)  # default

    for (x, y, w, h) in faces:
        face_img = preprocess_image(gray[y:y+h, x:x+w])

        face_img = np.expand_dims(face_img, axis=0)
        face_img = np.expand_dims(face_img, axis=-1)

        prediction = model.predict(face_img)
        class_id = np.argmax(prediction)
        confidence = prediction[0, class_id]

        names = ["Ishan", "Amogh", "Niraj", "Harshada", "Unknown"]
        recognized_name = names[class_id]

        log_file_path = get_log_file_path(recognized_name)

        # Check if log file for this individual exists, if not create it
        if not os.path.exists(log_file_path):
            assure_path_exists(log_file_path)

        # Check if log should be written
        if recognized_name not in last_log_time or (datetime.now() - last_log_time[recognized_name]).seconds > 10:
            log_entry = f"{timestamp} - Video: {video_file} - Recognized: {recognized_name} with confidence {confidence:.2f}\n"
            write_log(log_file_path, log_entry)
            last_log_time[recognized_name] = datetime.now()

        cv2.rectangle(frame, (x-22, y-90), (x+w+22, y-22), (0, 255, 0), -1)
        display_text = f"{recognized_name} ({confidence:.2f})"
        cv2.putText(frame, display_text, (x, y-40), font, 1, (255, 255, 255), 3)

    cv2.imshow('Frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
