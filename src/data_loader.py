import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array

def load_data(image_dir, target_size=(224,224)):
    images, labels = [], []
    classes = os.listdir(image_dir)
    for label in classes:
        folder = os.path.join(image_dir, label)
        for fname in os.listdir(folder):
            path = os.path.join(folder, fname)
            img = load_img(path, target_size=target_size, color_mode='rgb')
            arr = img_to_array(img) / 255.0  # normalize
            images.append(arr)
            labels.append(label)
    return np.array(images), np.array(labels)
