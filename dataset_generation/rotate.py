import cv2
import numpy as np
import random
import imageio.v2 as imageio


image_path = r'C:\Users\krish\OneDrive\Desktop\Dev\projects\codewave_cookie\data\undamaged\undamaged.jpg'
image = imageio.imread(image_path)

def rotate_image(image, angle):
    rows, cols, _ = image.shape
    M = cv2.getRotationMatrix2D((cols // 2, rows // 2), angle, 1)
    
  
    return cv2.warpAffine(image, M, (cols, rows), borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))


synthetic_images = []
for _ in range(10):  
   
    angle = random.uniform(-45, 45)
    synthetic_img = rotate_image(image, angle)
    
    synthetic_images.append(synthetic_img)


for i, img in enumerate(synthetic_images):
    imageio.imwrite(f'C:/Users/krish/OneDrive/Desktop/Dev/projects/codewave_cookie/data/augumented_undamaged/undamaged_aug{i+1}.jpg', img)

