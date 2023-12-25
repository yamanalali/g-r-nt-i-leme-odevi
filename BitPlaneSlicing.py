import cv2
import numpy as np
import matplotlib.pyplot as plt

def bit_plane_slice(image, bit_position):
    # Extract the specified bit plane from each pixel
    bit_plane = (image >> bit_position) & 1
    # Convert the bit plane to a 0-255 grayscale image
    bit_plane *= 255
    return bit_plane.astype(np.uint8)

# Load an example image
image_path = 'Photos/dog.jpg'
original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Get the height and width of the image
height, width = original_image.shape

# Number of bits per pixel (assuming 8-bit image)
num_bits = 8

# Create subplots to display the original image and its bit planes
plt.figure(figsize=(12, 6))

# Display the original image
plt.subplot(3, 3, 1)
plt.imshow(original_image, cmap='gray')
plt.title('Original Image')

# Display each bit plane
for i in range(num_bits):
    plt.subplot(3, 3, i + 2)
    plt.imshow(bit_plane_slice(original_image, i), cmap='gray')
    plt.title(f'Bit Plane {i}')

plt.tight_layout()
plt.show()
