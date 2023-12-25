import cv2
import numpy as np
import matplotlib.pyplot as plt

def adjust_gamma(image, gamma):
    # Apply gamma correction using the formula: new_pixel = old_pixel ** (1/gamma)
    gamma_corrected = np.clip(image ** (1/gamma), 0, 255).astype(np.uint8)
    return gamma_corrected

# Load an example image
image_path = 'Photos/dog.jpg'
original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Display the original image
plt.imshow(original_image, cmap='gray')
plt.title('Original Image')
plt.show()

# Adjust gamma (increase or decrease brightness)
gamma_value = 1.5  # You can adjust this value to control the gamma correction
gamma_corrected_image = adjust_gamma(original_image, gamma_value)

# Display the gamma-corrected image
plt.imshow(gamma_corrected_image, cmap='gray')
plt.title(f'Gamma Corrected Image (Gamma: {gamma_value})')
plt.show()
