import cv2
import numpy as np
import matplotlib.pyplot as plt

def adjust_contrast(image, alpha):
    # Perform contrast adjustment using the formula: new_pixel = alpha * original_pixel
    adjusted_image = np.clip(alpha * image, 0, 255).astype(np.uint8)
    return adjusted_image

# Load an example image
image_path = 'Photos/dog.jpg'
original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Display the original image
plt.imshow(original_image, cmap='gray')
plt.title('Original Image')
plt.show()

# Adjust contrast (increase contrast by scaling pixel intensities)
contrast_factor = 1.5  # You can adjust this value to control the contrast
contrast_adjusted_image = adjust_contrast(original_image, contrast_factor)

# Display the adjusted image
plt.imshow(contrast_adjusted_image, cmap='gray')
plt.title(f'Contrast Adjusted Image (Factor: {contrast_factor})')
plt.show()
