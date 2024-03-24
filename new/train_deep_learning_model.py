import cv2
import os
import numpy as np
from mtcnn import MTCNN
from keras_facenet import FaceNet
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
import joblib

def assure_path_exists(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)

def train_deep_learning_model():
    detector = MTCNN()
    facenet = FaceNet()
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
                results = detector.detect_faces(img)

                if results:
                    x1, y1, width, height = results[0]['box']
                    x2, y2 = x1 + width, y1 + height
                    face = img[y1:y2, x1:x2]
                    face = cv2.resize(face, (160, 160))
                    embedding = facenet.embeddings(face)

                    # Flatten to remove extra dimension
                    flattened_embedding = embedding.flatten()

                    # Reshape to make it compatible with the model
                    reshaped_embedding = flattened_embedding.reshape(1, 160, 160, 1)

                    faces.append(reshaped_embedding)
                    ids.append(label)

    ids = np.array(ids)

    # Convert faces to a numpy array
    faces = np.vstack(faces)

    # Reduce the dimensionality of the face embeddings
    pca = PCA(n_components=128)
    faces = pca.fit_transform(faces)

    # Use a Random Forest classifier for training
    model = RandomForestClassifier(n_estimators=100)
    model.fit(faces, ids)

    # Save the trained model
    joblib.dump(model, 'saved_model_deep_learning/deep_model.pkl')

train_deep_learning_model()
