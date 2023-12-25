import cv2
import numpy as np
import matplotlib.pyplot as plt

def apply_gaussian_blur(image, blur_kernel_size):
    # Apply Gaussian blur to the image
    blurred_image = cv2.GaussianBlur(image, (blur_kernel_size, blur_kernel_size), 0)
    return blurred_image

def apply_sharpening(image, sharpening_factor):
    # Create a kernel for sharpening
    sharpening_kernel = np.array([[-1, -1, -1],
                                  [-1,  9, -1],
                                  [-1, -1, -1]])

    # Apply the sharpening kernel to the image
    sharpened_image = cv2.filter2D(image, -1, sharpening_factor * sharpening_kernel)
    return sharpened_image

# Load an example image
image_path = 'Photos/dog.jpg'
original_image = cv2.imread(image_path)

# Display the original image
plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.show()

# Apply Gaussian blur for smoothing
blur_kernel_size = 5  # You can adjust this value to control the blur
smoothed_image = apply_gaussian_blur(original_image, blur_kernel_size)

# Display the smoothed image
plt.imshow(cv2.cvtColor(smoothed_image, cv2.COLOR_BGR2RGB))
plt.title(f'Smoothed Image (Blur Kernel Size: {blur_kernel_size})')
plt.show()

# Apply sharpening to the smoothed image
sharpening_factor = 1.5  # You can adjust this value to control the sharpening
sharpened_image = apply_sharpening(smoothed_image, sharpening_factor)

# Display the sharpened image
plt.imshow(cv2.cvtColor(sharpened_image, cv2.COLOR_BGR2RGB))
plt.title(f'Sharpened Image (Sharpening Factor: {sharpening_factor})')
plt.show()
