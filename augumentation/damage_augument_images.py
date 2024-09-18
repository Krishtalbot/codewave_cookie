import os
import cv2
from random import randint

# Paths to augmented images and where to save the damaged versions
augmented_dir = 'data/augmented_undamaged'
output_dir = 'data/augmented_damaged'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Function to apply blur or clipping
def damage_image(img):
    if randint(0, 1) == 0:
        # Apply Gaussian Blur
        return cv2.GaussianBlur(img, (5, 5), 0)
    else:
        # Apply random clipping
        h, w, _ = img.shape
        x = randint(0, w//2)
        y = randint(0, h//2)
        img[y:h-y, x:w-x] = 0  # Black out a random region (clipping)
        return img

# Process each augmented undamaged image
for filename in os.listdir(augmented_dir):
    img_path = os.path.join(augmented_dir, filename)
    img = cv2.imread(img_path)
    
    # Create damaged version
    damaged_img = damage_image(img)
    
    # Save the damaged image
    output_path = os.path.join(output_dir, filename.replace('undamaged', 'damaged'))
    cv2.imwrite(output_path, damaged_img)
