
import cv2
import os

#Method for checking existence of path i.e the directory

def assure_path_exists(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)

# Starting the web cam by invoking the VideoCapture method
vid_cam = cv2.VideoCapture(0)

# For detecting the faces in each frame we will use Haarcascade Frontal Face default classifier of OpenCV
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Set unique id for each individual person
face_id = 2

# Variable for co unting the no. of images
count = 0

#checking existence of path
assure_path_exists("training_data/")

# Looping starts here
while(True):

    # Capturing each video frame from the webcam
    _, image_frame = vid_cam.read()

    # Converting each frame to grayscale image
    gray = cv2.cvtColor(image_frame, cv2.COLOR_BGR2GRAY)

    # Detecting different faces
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    # Looping through all the detected faces in the frame
    for (x,y,w,h) in faces:

        # Crop the image frame into rectangle
        cv2.rectangle(image_frame, (x,y), (x+w,y+h), (255,0,0), 2)
        
        # Increasing the no. of images by 1 since frame we captured
        count += 1

        # Saving the captured image into the training_data folder
        cv2.imwrite("training_data/Person." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])

        # Displaying the frame with rectangular bounded box
        cv2.imshow('frame', image_frame)

    # press 'q' for at least 100ms to stop this capturing process
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

    #We are taking 100 images for each person for the training data
    # If image taken reach 100, stop taking video
    elif count>1000:
        break

# Terminate video
vid_cam.release()

# Terminate all started windows
cv2.destroyAllWindows()










##
import cv2
import os
import numpy as np

def assure_path_exists(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)

def collect_training_data(face_id):
    vid_cam = cv2.VideoCapture(0)
    face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    count = 0

    assure_path_exists(f"training_data/{face_id}/")

    while True:
        _, image_frame = vid_cam.read()
        gray = cv2.cvtColor(image_frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            # Augment data with random flips
            if np.random.choice([True, False]):
                gray = cv2.flip(gray, 1)

            # Augment data with random rotations (between -15 and 15 degrees)
            rotation_angle = np.random.uniform(-15, 15)
            rotation_matrix = cv2.getRotationMatrix2D((x + w / 2, y + h / 2), rotation_angle, 1.0)
            gray = cv2.warpAffine(gray, rotation_matrix, (gray.shape[1], gray.shape[0]))

            # Augment data with random changes in brightness (scaling factor between 0.5 and 1.5)
            brightness_scale = np.random.uniform(0.5, 1.5)
            gray = cv2.multiply(gray, brightness_scale)

            # Augment data with random scaling (scaling factor between 0.8 and 1.2)
            scale_factor = np.random.uniform(0.8, 1.2)
            gray = cv2.resize(gray, None, fx=scale_factor, fy=scale_factor)

            cv2.rectangle(image_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            count += 1
            cv2.imwrite(f"training_data/{face_id}/person_{face_id}_{count}.jpg", gray[y:y + h, x:x + w])
            cv2.imshow('frame', image_frame)

        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
        elif count >= 20:
            break

    vid_cam.release()
    cv2.destroyAllWindows()

# Change the face_id accordingly for each person
collect_training_data(face_id=1)

