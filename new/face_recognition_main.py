import cv2
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input

# Method for checking existence of path i.e the directory
def assure_path_exists(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)

# Load the saved pre-trained model
model = load_model('saved_model/s_model.h5')

# Load prebuilt classifier for Frontal Face detection
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)

# Font style
font = cv2.FONT_HERSHEY_SIMPLEX

# Initialize and start the video frame capture from webcam
cam = cv2.VideoCapture(0)

# Initialize variables to keep track of total faces and confidence sum
total_faces = 0
confidence_sum = 0

# Looping starts here
while True:
    # Read the video frame
    ret, im = cam.read()

    # Convert the captured frame into grayscale
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    # Getting all faces from the video frame
    faces = faceCascade.detectMultiScale(gray, 1.2, 5)  # default

    # Reset total faces and confidence sum for each frame
    total_faces = 0
    confidence_sum = 0

    # For each face in faces, we will start predicting using the pre-trained model
    for (x, y, w, h) in faces:
        # Create rectangle around the face
        cv2.rectangle(im, (x-20, y-20), (x+w+20, y+h+20), (0, 255, 0), 4)

        # Preprocess the face image before making a prediction
        face_img = gray[y:y+h, x:x+w]
        face_img_rgb = cv2.cvtColor(face_img, cv2.COLOR_GRAY2RGB)  # Convert to RGB
        face_img_rgb = cv2.resize(face_img_rgb, (224, 224))
        face_img_rgb = image.img_to_array(face_img_rgb)
        face_img_rgb = np.expand_dims(face_img_rgb, axis=0)
        face_img_rgb = preprocess_input(face_img_rgb)

        # Predict using the loaded model
        prediction = model.predict(face_img_rgb)

        # Set the name and confidence level according to the prediction
        class_id = np.argmax(prediction)
        confidence = prediction[0, class_id]
        accuracy_percentage = round(confidence * 100, 2)
        names = ["Ishan", "Amogh", "Niraj", "Harshada", "Unknown"]
        recognized_name = names[class_id]
        display_text = f"{recognized_name} ({accuracy_percentage}%)"

        # Set rectangle around face and name of the person
        cv2.rectangle(im, (x-22, y-90), (x+w+22, y-22), (0, 255, 0), -1)
        cv2.putText(im, display_text, (x, y-40), font, 1, (255, 255, 255), 3)

        # Increment total faces and add confidence to sum
        total_faces += 1
        confidence_sum += confidence

    # Calculate average confidence level
    average_confidence = 0
    if total_faces > 0:
        average_confidence = confidence_sum / total_faces

    # Display average confidence level
    cv2.putText(im, f"Average Confidence: {round(average_confidence * 100, 2)}%", (20, 40), font, 1, (255, 255, 255), 3)

    # Display the video frame with the bounded rectangle
    cv2.imshow('im', im)

    # Press q to close the program
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Terminate video
cam.release()

# Close all windows
cv2.destroyAllWindows()
