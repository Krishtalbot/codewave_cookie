from model import build_autoencoder
from data_loader import load_images

# Load dataset
damaged_images, undamaged_images = load_images('data/damaged', 'data/augmented_damaged', 'data/augmented_undamaged')

# Check sizes
assert len(damaged_images) == len(undamaged_images), "Mismatch in number of damaged and undamaged images!"

# Build and compile the model
autoencoder = build_autoencoder((128, 128, 3))
autoencoder.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
autoencoder.fit(damaged_images, undamaged_images, epochs=50, batch_size=8, validation_split=0.1)

# Save the trained model
autoencoder.save('checkpoints/coin_restoration_autoencoder.h5')
from model import build_autoencoder
from data_loader import load_images

# Load dataset
damaged_images, undamaged_images = load_images('data/damaged', 'data/augmented_damaged', 'data/augmented_undamaged')

# Check sizes
assert len(damaged_images) == len(undamaged_images), "Mismatch in number of damaged and undamaged images!"

# Build and compile the model
autoencoder = build_autoencoder((128, 128, 3))
autoencoder.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
autoencoder.fit(damaged_images, undamaged_images, epochs=50, batch_size=8, validation_split=0.1)

# Save the trained model
autoencoder.save('checkpoints/coin_restoration_autoencoder.h5')
