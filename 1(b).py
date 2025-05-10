import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# --- Harmonic Mean Filter ---
def apply_harmonic_mean_filter(image, mask_size):
    filtered_image = np.zeros_like(image, dtype=np.float32)
    height, width = image.shape
    offset = mask_size // 2

    for r in range(height):
        for c in range(width):
            sub_matrix = image[max(r-offset,0):min(r+offset+1,height), max(c-offset,0):min(c+offset+1,width)]
            with np.errstate(divide='ignore'):
                harmonic_mean = (sub_matrix.size / np.sum(1.0 / (sub_matrix + 1e-6)))
            filtered_image[r, c] = min(255, harmonic_mean)

    return np.uint8(filtered_image)

# --- Geometric Mean Filter ---
def apply_geometric_mean_filter(image, mask_size):
    filtered_image = np.zeros_like(image, dtype=np.float32)
    height, width = image.shape
    offset = mask_size // 2

    for r in range(height):
        for c in range(width):
            sub_matrix = image[max(r-offset,0):min(r+offset+1,height), max(c-offset,0):min(c+offset+1,width)]
            sub_matrix = sub_matrix + 1e-6  # avoid log(0)
            geo_mean = np.exp(np.mean(np.log(sub_matrix)))
            filtered_image[r, c] = min(255, geo_mean)

    return np.uint8(filtered_image)

# --- Salt & Pepper Noise ---
def add_salt_pepper_noise(image, percent):
    noisy = image.copy()
    num_noisy = int(percent / 100 * image.size)
    coords = [np.random.randint(0, i, num_noisy) for i in image.shape]
    noisy[coords[0], coords[1]] = np.random.choice([0, 255], num_noisy)
    return noisy

# --- PSNR Calculation ---
def compute_psnr(original, compared):
    mse = np.mean((original.astype(np.float64) - compared.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')
    return round(20 * np.log10(255.0) - 10 * np.log10(mse), 2)

# --- Load Image ---
image_path = r"C:\Users\hp\Desktop\Image Processing Lab\ela.jpg"
character_image = cv2.imread(image_path, 0)

if character_image is None:
    raise FileNotFoundError("Image not found. Check the file path.")

character_image = cv2.resize(character_image, (512, 512))

# --- Add Noise ---
noisy_image = add_salt_pepper_noise(character_image, 1.5)
psnr_noisy = compute_psnr(character_image, noisy_image)

# --- Apply Filters ---
mask_size = 3
harmonic_filtered = apply_harmonic_mean_filter(noisy_image, mask_size)
geometric_filtered = apply_geometric_mean_filter(noisy_image, mask_size)

psnr_harmonic = compute_psnr(character_image, harmonic_filtered)
psnr_geometric = compute_psnr(character_image, geometric_filtered)

# --- Display Results ---
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.imshow(character_image, cmap='gray')
plt.title("Original Image")
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(noisy_image, cmap='gray')
plt.title(f"Noisy Image\nPSNR: {psnr_noisy} dB")
plt.axis('off')

plt.subplot(2, 2, 3)
plt.imshow(harmonic_filtered, cmap='gray')
plt.title(f"Harmonic Mean Filter\nPSNR: {psnr_harmonic} dB")
plt.axis('off')

plt.subplot(2, 2, 4)
plt.imshow(geometric_filtered, cmap='gray')
plt.title(f"Geometric Mean Filter\nPSNR: {psnr_geometric} dB")
plt.axis('off')


plt.tight_layout()
plt.show()
