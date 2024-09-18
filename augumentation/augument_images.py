import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array

# Path to the undamaged image
undamaged_dir = 'data/undamaged'
output_dir = 'data/augmented_undamaged'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Load the undamaged image
img_path = os.path.join(undamaged_dir, 'undamaged.jpg')
img = load_img(img_path, target_size=(128, 128))
img_array = img_to_array(img)
img_array = img_array.reshape((1,) + img_array.shape)

# Create an ImageDataGenerator for augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Generate and save augmented images
i = 0
for batch in datagen.flow(img_array, batch_size=1, save_to_dir=output_dir, save_prefix='undamaged_aug', save_format='jpg'):
    i += 1
    if i >= 50:  # Generate 50 augmented images
        break
