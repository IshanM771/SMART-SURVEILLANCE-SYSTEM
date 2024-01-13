import cv2
import os
import numpy as np

def assure_path_exists(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)

def train_deep_learning_model():
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    assure_path_exists('saved_model_deep_learning/')

    faces = []
    ids = []

    for root, dirs, files in os.walk("training_data"):
        for dir in dirs:
            subject_path = os.path.join(root, dir)
            label = int(dir)
            for filename in os.listdir(subject_path):
                img_path = os.path.join(subject_path, filename)
                img = cv2.imread(img_path)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces.append(gray)
                ids.append(label)

    ids = np.array(ids)

    model = cv2.face.LBPHFaceRecognizer_create()
    model.train(faces, ids)
    model.save('saved_model_deep_learning/deep_model.yml')

train_deep_learning_model()
