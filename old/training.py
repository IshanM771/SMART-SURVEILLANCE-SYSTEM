import cv2
import os
import numpy as np
from PIL import Image
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# Method for checking existence of path i.e the directory
def assure_path_exists(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)

# We will be using a pre-trained ResNet50 model for feature extraction
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')  # Assuming binary classification
])

# For detecting the faces in each frame we will use Haarcascade Frontal Face default classifier of OpenCV
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Method for loading and preprocessing images
def load_images_and_labels(path):
    images, labels = [], []
    image_paths = [os.path.join(path, f) for f in os.listdir(path)]
    
    for image_path in image_paths:
        PIL_img = Image.open(image_path).convert('RGB')
        img_array = np.array(PIL_img.resize((224, 224))) / 255.0  # Resize to match ResNet input size
        label = int(os.path.split(image_path)[-1].split(".")[1])
        images.append(img_array)
        labels.append(label)
    
    return np.array(images), np.array(labels)

# Getting the faces and IDs
images, labels = load_images_and_labels('training_data')

# Split the data into training and validation sets
train_images, val_images, train_labels, val_labels = train_test_split(images, labels, test_size=0.2, random_state=42)

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model with more epochs
model.fit(train_images, train_labels, epochs=20, batch_size=32, validation_data=(val_images, val_labels))

# Save the model into s_model.h5
assure_path_exists('saved_model/')
model.save('saved_model/s_model.h5')
