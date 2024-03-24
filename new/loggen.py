import cv2
import os
import numpy as np
from tensorflow.keras.models import load_model
from datetime import datetime

# Method for checking existence of path i.e the directory


# Load the saved pre-trained model
model = load_model('saved_model/s_model.h5')

# Load prebuilt classifier for Frontal Face detection
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)

# Font style
font = cv2.FONT_HERSHEY_SIMPLEX

# Initialize and start the video frame capture from webcam
cam = cv2.VideoCapture(0)

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
log_file_path = 'identification_log.txt'
log_file_path = os.path.abspath(log_file_path)
assure_path_exists(log_file_path)


# Looping starts here
while True:
    # Read the video frame
    ret, im = cam.read()

    # Convert the captured frame into grayscale
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    # Getting all faces from the video frame
    faces = faceCascade.detectMultiScale(gray, 1.2, 5)  # default

    # For each face in faces, we will start predicting using the pre-trained model
    for (x, y, w, h) in faces:
        # Preprocess the face image before making a prediction
        face_img = gray[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, (224, 224))  # Assuming the model expects this size
        face_img = np.expand_dims(face_img, axis=-1)  # Add batch dimension
        face_img = np.expand_dims(face_img, axis=0)  # Add channel dimension

        # Predict using the loaded model
        prediction = model.predict(face_img)

        # Set the name and confidence level according to the prediction
        class_id = np.argmax(prediction)
        confidence = prediction[0, class_id]
        names = ["Ishan", "Amogh", "Niraj", "Harshada", "Unknown"]
        recognized_name = names[class_id]

        # Log the result with timestamp to a text file
        with open(log_file_path, 'a') as log_file:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_entry = f"{timestamp} - Recognized: {recognized_name} with confidence {confidence:.2f}\n"
            log_file.write(log_entry)

        # Set rectangle around face and name of the person
        cv2.rectangle(im, (x-22, y-90), (x+w+22, y-22), (0, 255, 0), -1)
        display_text = f"{recognized_name} ({confidence:.2f})"
        cv2.putText(im, display_text, (x, y-40), font, 1, (255, 255, 255), 3)

    # Display the video frame with the bounded rectangle
    cv2.imshow('im', im)

    # Press q to close the program
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Terminate video
cam.release()

# Close all windows
cv2.destroyAllWindows()
