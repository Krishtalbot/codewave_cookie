import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array

def load_images(damaged_dir, augmented_damaged_dir, augmented_undamaged_dir, img_size=(128, 128)):
    # Load damaged images
    damaged_images = []
    for filename in os.listdir(damaged_dir):
        img_path = os.path.join(damaged_dir, filename)
        img = load_img(img_path, target_size=img_size)
        img_array = img_to_array(img)
        damaged_images.append(img_array)

    # Load augmented damaged images
    for filename in os.listdir(augmented_damaged_dir):
        img_path = os.path.join(augmented_damaged_dir, filename)
        img = load_img(img_path, target_size=img_size)
        img_array = img_to_array(img)
        damaged_images.append(img_array)

    # Load augmented undamaged images
    undamaged_images = []
    for filename in os.listdir(augmented_undamaged_dir):
        img_path = os.path.join(augmented_undamaged_dir, filename)
        img = load_img(img_path, target_size=img_size)
        img_array = img_to_array(img)
        undamaged_images.append(img_array)

    return np.array(damaged_images) / 255.0, np.array(undamaged_images) / 255.0
