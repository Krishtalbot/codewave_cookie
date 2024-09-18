import cv2
import numpy as np
import random
import imageio.v2 as imageio

# Load the original image
image_path = r'C:\Users\krish\OneDrive\Desktop\Dev\projects\codewave_cookie\data\undamaged\undamaged.jpg'
image = imageio.imread(image_path)

def add_mask(image):
    # Remove a small, irregular part of the image (simulate chipping)
    masked_img = image.copy()
    rows, cols, _ = masked_img.shape
    
    # Define random coordinates for an irregular "chipped" part
    x = random.randint(0, cols - cols // 4)
    y = random.randint(0, rows - rows // 4)
    w = random.randint(cols // 8, cols // 4)
    h = random.randint(rows // 8, rows // 4)
    
    # Apply the "chipped" section as a whited-out part
    masked_img[y:y+h, x:x+w] = [255, 255, 255]  # Set part of the image to white
    
    return masked_img

def add_noise(image, noise_level=25):
    # Add random noise to the image
    noisy_img = image.copy()
    noise = np.random.randint(-noise_level, noise_level, noisy_img.shape, dtype='int16')
    noisy_img = noisy_img.astype('int16') + noise
    noisy_img = np.clip(noisy_img, 0, 255).astype('uint8')  # Ensure the values are within valid pixel range
    return noisy_img

def add_blur(image, num_regions=3):
    # Apply random blurring to multiple parts of the image to simulate wear
    blur_img = image.copy()
    rows, cols, _ = blur_img.shape
    
    for _ in range(random.randint(1, num_regions)):  # Random number of blur regions
        # Select a random region to blur
        x = random.randint(0, cols - cols // 4)
        y = random.randint(0, rows - rows // 4)
        w = random.randint(cols // 8, cols // 4)
        h = random.randint(rows // 8, rows // 4)
        
        # Extract the region of interest (ROI) to blur
        roi = blur_img[y:y+h, x:x+w]
        
        # Apply Gaussian blur to this region
        ksize = random.choice([(5, 5), (7, 7), (9, 9)])  # Larger kernel for stronger blur
        roi_blurred = cv2.GaussianBlur(roi, ksize, 0)
        
        # Put the blurred ROI back into the image
        blur_img[y:y+h, x:x+w] = roi_blurred
    
    return blur_img

def add_mask_and_blur(image):
    # Apply both masking and blurring to the image
    img = image.copy()
    
    # First, apply the mask (whited-out region)
    img = add_mask(img)
    
    # Then, apply the blur to different parts of the image
    img = add_blur(img)
    
    return img

# Generate synthetic images with blurring, masking, noise, or both effects
synthetic_images = []
for _ in range(100):  # Create 10 synthetic images
    effect = random.random()
    
    if effect < 0.33:
        synthetic_img = add_mask(image)  # Apply mask
    elif effect < 0.66:
        synthetic_img = add_blur(image)  # Apply blur
    else:
        synthetic_img = add_mask_and_blur(image)  # Apply both mask and blur
    
    # Optionally, add noise to some images
    if random.random() > 0.5:
        synthetic_img = add_noise(synthetic_img)
    
    synthetic_images.append(synthetic_img)

# Save synthetic images
for i, img in enumerate(synthetic_images):
    imageio.imwrite(f'C:/Users/krish/OneDrive/Desktop/Dev/projects/codewave_cookie/data/damaged/damaged{i+1}.jpg', img)