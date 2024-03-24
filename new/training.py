import cv2
import os
import numpy as np
from PIL import Image
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from mtcnn import MTCNN  # Install this library using: pip install mtcnn

def assure_path_exists(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)

# Create a dictionary to map names to unique integer labels
name_to_label = {0:"Ishan", 1:"inzy"}

def train_model():
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),  # Increased complexity
        layers.Dropout(0.5),
        layers.Dense(len(name_to_label), activation='softmax')  # Adjust the output layer to match the number of classes
    ])

    detector = MTCNN()

    def load_images_and_labels(path):
        images, labels = [], []
        image_paths = [os.path.join(path, f) for f in os.listdir(path)]
        
        for image_path in image_paths:
            PIL_img = Image.open(image_path).convert('RGB')
            img_array = np.array(PIL_img.resize((224, 224))) / 255.0  # Resize to match ResNet input size
            name = os.path.split(image_path)[-1].split(".")[1]
            if name not in name_to_label:
                name_to_label[name] = len(name_to_label)  # Assign a new label to the new name
            label = name_to_label[name]
            images.append(img_array)
            labels.append(label)
        
        return np.array(images), np.array(labels)

    def get_faces(image_path):
        image = cv2.imread(image_path)
        result = detector.detect_faces(image)
        faces = []
        for face in result:
            x, y, w, h = face['box']
            faces.append(image[y:y+h, x:x+w])
        return faces

    def load_and_preprocess_faces(path):
        images, labels = [], []
        image_paths = [os.path.join(path, f) for f in os.listdir(path)]
        
        for image_path in image_paths:
            faces = get_faces(image_path)
            for face in faces:
                PIL_img = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
                img_array = np.array(PIL_img.resize((224, 224))) / 255.0
                name = os.path.split(image_path)[-1].split(".")[1]
                if name not in name_to_label:
                    name_to_label[name] = len(name_to_label)  # Assign a new label to the new name
                label = name_to_label[name]
                images.append(img_array)
                labels.append(label)
        
        return np.array(images), np.array(labels)

    images, labels = load_and_preprocess_faces('training_data')
    
    train_images, val_images, train_labels, val_labels = train_test_split(images, labels, test_size=int(0.2 * len(images)), random_state=42)

    # Data augmentation
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True)
    datagen.fit(train_images)

    # Early stopping & checkpointing the best model
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
    mc = ModelCheckpoint('saved_model/best_model.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(datagen.flow(train_images, train_labels, batch_size=32),
              validation_data=(val_images, val_labels),
              steps_per_epoch=len(train_images) / 32,
              epochs=50,  # Increased epochs
              callbacks=[es, mc])

    # Save the model
    model.save('saved_model/s_model.h5')
