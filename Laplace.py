import cv2
import numpy as np
import matplotlib.pyplot as plt

def apply_blur_and_laplacian(image, blur_kernel_size):
    # Apply Gaussian blur to the image
    blurred_image = cv2.GaussianBlur(image, (blur_kernel_size, blur_kernel_size), 0)

    # Apply Laplacian filter to the blurred image
    laplacian = cv2.Laplacian(blurred_image, cv2.CV_64F)

    # Normalize the Laplacian to the range [0, 255]
    laplacian = cv2.normalize(laplacian, None, 0, 255, cv2.NORM_MINMAX)

    return laplacian.astype(np.uint8)

# Load an example image
image_path = 'Photos/dog.jpg'
original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Display the original image
plt.imshow(original_image, cmap='gray')
plt.title('Original Image')
plt.show()

# Apply blurring operation with Laplacian filter
blur_kernel_size = 5  # You can adjust this value to control the blur
laplacian_filtered_image = apply_blur_and_laplacian(original_image, blur_kernel_size)

# Display the resulting image
plt.imshow(laplacian_filtered_image, cmap='gray')
plt.title(f'Blurring with Laplacian (Blur Kernel Size: {blur_kernel_size})')
plt.show()
