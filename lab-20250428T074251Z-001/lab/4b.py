# Task 4(b): Ringing Effect of Ideal Low Pass Filter
 #Task 4 - Filtering in Frequency Domain
#Task 4(b): Take a grayscale image of size 512x512, 
#add some Gaussian Noise & observe the Ringing Effect of Ideal Low Pass Filter on the image. 
#Use different radius of Ideal Low Pass Filter & display their results
#Importing the Libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Function to apply Ideal Low Pass Filter
def apply_ideal_low_pass_filter(image_fft, cutoff_frequency):
    height, width = image_fft.shape
    filter_mask = np.zeros((height, width), dtype=np.float32)

    for u in range(height):
        for v in range(width):
            D = np.sqrt((u - height / 2) ** 2 + (v - width / 2) ** 2)
            if D <= cutoff_frequency:
                filter_mask[u, v] = 1

    # Apply filter in frequency domain
    filtered_fft = image_fft * filter_mask
    filtered_image = np.fft.ifft2(np.fft.ifftshift(filtered_fft))
    return np.abs(filtered_image)

# Load a 512x512 grayscale image
image = cv2.imread('./Images/Fig0445(a) Characters Test Pattern 688x688.tif', 0)

# Resize to 512x512
image = cv2.resize(image, (512, 512))

# Add Gaussian noise
noise = np.random.normal(0, 20, image.shape)  # mean=0, std=20
noisy_image = np.clip(image + noise, 0, 255).astype(np.uint8)

plt.figure(figsize=(6, 5))
plt.imshow(noisy_image, cmap='gray')
plt.title("Noisy Image")
plt.axis('off')
plt.show()

# FFT of noisy image
image_fft = np.fft.fftshift(np.fft.fft2(noisy_image))

# Apply Ideal Low Pass Filters with different cutoff radii
plt.figure(figsize=(15, 10))
for i, radius in enumerate([10, 20, 30, 50, 70, 100]):
    filtered_image = apply_ideal_low_pass_filter(image_fft, radius)
    plt.subplot(2, 3, i+1)
    plt.imshow(filtered_image, cmap='gray')
    plt.title(f"Radius = {radius}")
    plt.axis('off')

plt.suptitle("Ringing Effect of Ideal Low Pass Filter", fontsize=16)
plt.tight_layout()
plt.show()
