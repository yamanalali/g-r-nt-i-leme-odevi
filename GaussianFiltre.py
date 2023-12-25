import cv2
import numpy as np
import matplotlib.pyplot as plt

def apply_gaussian_filter(image, kernel_size, sigma):
    # Apply the Gaussian filter using OpenCV's GaussianBlur function
    filtered_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)

    return filtered_image

# Load an example image
image_path = 'Photos/dog.jpg'
original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Display the original image
plt.imshow(original_image, cmap='gray')
plt.title('Original Image')
plt.show()

# Apply a Gaussian filter with a specified kernel size and sigma
kernel_size = 5  # You can adjust this value to control the filter size
sigma = 1.0  # You can adjust this value to control the standard deviation
filtered_image = apply_gaussian_filter(original_image, kernel_size, sigma)

# Display the filtered image
plt.imshow(filtered_image, cmap='gray')
plt.title(f'Gaussian Filtered Image (Kernel Size: {kernel_size}, Sigma: {sigma})')
plt.show()
