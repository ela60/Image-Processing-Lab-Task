import cv2
import numpy as np
import matplotlib.pyplot as plt

# Function to apply Average Filter manually
def average_filter(image, mask_size):
    filtered_image = np.zeros_like(image, dtype=np.float32)
    height, width = image.shape
    offset = mask_size // 2
    weight = mask_size * mask_size

    for r in range(height):
        for c in range(width):
            for x in range(-offset, offset + 1):
                for y in range(-offset, offset + 1):
                    if 0 <= r + x < height and 0 <= c + y < width:
                        filtered_image[r, c] += image[r + x, c + y] / weight

    return np.uint8(np.clip(filtered_image, 0, 255))

# Function to add Salt & Pepper noise
def add_salt_pepper_noise(image, percent):
    noisy_image = image.copy()
    total_pixels = image.shape[0] * image.shape[1]
    noise_pixels = int(total_pixels * percent / 100)

    for _ in range(noise_pixels // 2):
        i = np.random.randint(0, image.shape[0])
        j = np.random.randint(0, image.shape[1])
        noisy_image[i, j] = 0  # pepper

    for _ in range(noise_pixels // 2):
        i = np.random.randint(0, image.shape[0])
        j = np.random.randint(0, image.shape[1])
        noisy_image[i, j] = 255  # salt

    return noisy_image

# Function to compute PSNR
def compute_psnr(original, processed):
    original = np.float64(original)
    processed = np.float64(processed)
    mse = np.mean((original - processed) ** 2)
    if mse == 0:
        return float('inf')
    psnr = 20 * np.log10(255.0) - 10 * np.log10(mse)
    return round(psnr, 2)

# Load the image
image_path = r"C:\Users\hp\Desktop\Image Processing Lab\Characters Test Pattern 688x688 (1).tif"
character_image = cv2.imread(image_path, 0)

if character_image is None:
    raise FileNotFoundError(f"Image not found at: {image_path}")

# Resize to 512x512
character_image = cv2.resize(character_image, (512, 512))

# Add salt & pepper noise
noisy_image = add_salt_pepper_noise(character_image, 15)

# Compute PSNR between original and noisy image
psnr_noisy = compute_psnr(character_image, noisy_image)

# Apply Average Filter with different mask sizes and compute PSNR
mask_sizes = [3, 5, 7]
filtered_results = []

for mask_size in mask_sizes:
    filtered_image = average_filter(noisy_image, mask_size)
    psnr_filtered = compute_psnr(character_image, filtered_image)
    filtered_results.append((filtered_image, mask_size, psnr_filtered))

# Plot all images in one figure
fig, axes = plt.subplots(1, 5, figsize=(20, 5))

# Original image
axes[0].imshow(character_image, cmap="gray")
axes[0].set_title("Original")
axes[0].axis("off")

# Noisy image
axes[1].imshow(noisy_image, cmap="gray")
axes[1].set_title(f"Noisy\nPSNR: {psnr_noisy} dB")
axes[1].axis("off")

# Filtered images
for idx, (img, size, psnr_val) in enumerate(filtered_results):
    axes[idx + 2].imshow(img, cmap="gray")
    axes[idx + 2].set_title(f"Avg {size}x{size}\nPSNR: {psnr_val} dB")
    axes[idx + 2].axis("off")

plt.tight_layout()
plt.show()
