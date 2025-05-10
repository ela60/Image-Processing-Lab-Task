import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.util import random_noise
from skimage.metrics import peak_signal_noise_ratio as psnr

# Load and resize image (FIXED PATH)
img = cv2.imread(r'C:\Users\hp\Desktop\Image Processing Lab\Characters Test Pattern 688x688 (1).tif', 0)
img = cv2.resize(img, (512, 512))

# Add Gaussian noise
noisy_img = random_noise(img, mode='gaussian', var=0.01)
noisy_img = np.array(255 * noisy_img, dtype=np.uint8)

# FFT of the noisy image
f = np.fft.fft2(noisy_img)
fshift = np.fft.fftshift(f)
rows, cols = img.shape
crow, ccol = rows // 2, cols // 2

# Butterworth Low Pass Filter (4th order)
def butterworth_low_pass(shape, D0, n):
    P, Q = shape
    u = np.arange(P) - P // 2
    v = np.arange(Q) - Q // 2
    U, V = np.meshgrid(u, v, indexing='ij')
    D = np.sqrt(U**2 + V**2)
    H = 1 / (1 + (D / D0)**(2 * n))
    return H

# Gaussian Low Pass Filter
def gaussian_low_pass(shape, D0):
    P, Q = shape
    u = np.arange(P) - P // 2
    v = np.arange(Q) - Q // 2
    U, V = np.meshgrid(u, v, indexing='ij')
    D = np.sqrt(U**2 + V**2)
    H = np.exp(-(D**2) / (2 * (D0**2)))
    return H

# Apply Butterworth Filter
H_butter = butterworth_low_pass((rows, cols), D0=50, n=4)
filtered_butter = np.abs(np.fft.ifft2(np.fft.ifftshift(fshift * H_butter)))

# Apply Gaussian Filter
H_gaussian = gaussian_low_pass((rows, cols), D0=50)
filtered_gaussian = np.abs(np.fft.ifft2(np.fft.ifftshift(fshift * H_gaussian)))

# PSNR Calculation
psnr_butter = psnr(img, filtered_butter)
psnr_gaussian = psnr(img, filtered_gaussian)
print(f"Butterworth LPF PSNR: {psnr_butter:.2f} dB")
print(f"Gaussian LPF PSNR: {psnr_gaussian:.2f} dB")

# Optional: Show results
plt.figure(figsize=(10, 5))
plt.subplot(1, 3, 1), plt.imshow(noisy_img, cmap='gray'), plt.title("Noisy Image"), plt.axis("off")
plt.subplot(1, 3, 2), plt.imshow(filtered_butter, cmap='gray'), plt.title("Butterworth LPF"), plt.axis("off")
plt.subplot(1, 3, 3), plt.imshow(filtered_gaussian, cmap='gray'), plt.title("Gaussian LPF"), plt.axis("off")
plt.tight_layout()
plt.show()
