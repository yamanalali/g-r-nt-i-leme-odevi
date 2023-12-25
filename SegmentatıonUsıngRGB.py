import cv2
import numpy as np
import matplotlib.pyplot as plt

def rgb_mean_segmentation(image, threshold):
    # Calculate mean RGB values for each region
    mean_values = np.mean(image, axis=(0, 1))

    # Create a binary mask based on the threshold
    mask = np.all(image > mean_values * threshold, axis=-1)

    # Apply the mask to the original image
    segmented_image = np.zeros_like(image)
    segmented_image[mask] = image[mask]

    return segmented_image


image_path = 'Photos/dog.jpg'
original_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.show()
threshold = 1.1 # You can adjust this value to control the segmentation
segmented_image = rgb_mean_segmentation(original_image, threshold)

# Display the segmented image
plt.imshow(cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB))
plt.title(f'Segmented Image (Threshold: {threshold})')
plt.show()
