import cv2
import numpy as np
import matplotlib.pyplot as plt

def contraharmonic_mean_filter(image, kernel_size, Q):
    # Padding to handle border pixels
    padded_image = cv2.copyMakeBorder(image, kernel_size // 2, kernel_size // 2, kernel_size // 2, kernel_size // 2, cv2.BORDER_CONSTANT)

    # Apply Contraharmonic Mean filter
    filtered_image = np.zeros_like(image, dtype=np.float32)
    for i in range(kernel_size // 2, padded_image.shape[0] - kernel_size // 2):
        for j in range(kernel_size // 2, padded_image.shape[1] - kernel_size // 2):
            region = padded_image[i - kernel_size // 2:i + kernel_size // 2 + 1, j - kernel_size // 2:j + kernel_size // 2 + 1]
            numerator = np.sum(region ** (Q + 1))
            denominator = np.sum(region ** Q)
            filtered_image[i - kernel_size // 2, j - kernel_size // 2] = numerator / denominator if denominator != 0 else 0

    return filtered_image.astype(np.uint8)

# Load an example image
image_path = 'Photos/dog.jpg'
original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Display the original image
plt.imshow(original_image, cmap='gray')
plt.title('Original Image')
plt.show()

# Apply Contraharmonic Mean filter
kernel_size = 3  # You can adjust this value to control the filter size
Q = 1.5  # You can adjust this value to control the order of the filter
filtered_image = contraharmonic_mean_filter(original_image, kernel_size, Q)

# Display the filtered image
plt.imshow(filtered_image, cmap='gray')
plt.title(f'Contraharmonic Mean Filter (Kernel Size: {kernel_size}, Q: {Q})')
plt.show()
