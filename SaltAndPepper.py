import cv2
import numpy as np
import matplotlib.pyplot as plt

def add_salt_and_pepper_noise(image, salt_prob, pepper_prob):
    noisy_image = np.copy(image)

    # Add salt noise
    salt_mask = np.random.rand(*image.shape[:2])
    noisy_image[salt_mask < salt_prob] = 255

    # Add pepper noise
    pepper_mask = np.random.rand(*image.shape[:2])
    noisy_image[pepper_mask < pepper_prob] = 0

    return noisy_image

image_path = 'Photos/dog.jpg'
original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Display the original image
plt.imshow(original_image, cmap='gray')
plt.title('Original Image')
plt.show()

# Add salt and pepper noise
salt_probability = 0.02  # You can adjust this value to control the amount of salt
pepper_probability = 0.03  # You can adjust this value to control the amount of pepper
noisy_image = add_salt_and_pepper_noise(original_image, salt_probability, pepper_probability)

plt.imshow(noisy_image, cmap='gray')
plt.title('Image with Salt and Pepper Noise')
plt.show()
