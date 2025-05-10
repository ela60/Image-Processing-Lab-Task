# 🖼️ Image Processing Lab Tasks

This repository contains implementations of various spatial and frequency domain filtering techniques, applied to noisy images. Noise removal performance is evaluated using PSNR.

---

## 📚 Table of Contents

- [Overview](#overview)
- [Tasks](#tasks)
  - [Noise Addition](#1-noise-addition)
  - [Spatial Filtering](#2-spatial-domain-filtering)
  - [Frequency Filtering](#3-frequency-domain-filtering)
- [Evaluation](#evaluation)
- [Technologies Used](#technologies-used)
- [Sample Results](#sample-results)
- [Conclusion](#conclusion)
- [Usage](#usage)

---

## 📌 Overview

This project focuses on:
- Adding **salt & pepper** and **Gaussian noise** to images.
- Applying various **noise removal filters** in both spatial and frequency domains.
- Measuring the quality of filtered images using **PSNR (Peak Signal-to-Noise Ratio).**

---

## 🧪 Tasks

### 1. Noise Addition

- **Salt & Pepper Noise**: Random black and white pixels.
- **Gaussian Noise**: Normally distributed random noise.

### 2. Spatial Domain Filtering

| Filter              | Description                                      |
|---------------------|--------------------------------------------------|
| Arithmetic Mean     | Averages pixel values in a local window.         |
| Geometric Mean      | Useful for reducing multiplicative noise.        |
| Harmonic Mean       | Best for eliminating salt noise.                 |

### 3. Frequency Domain Filtering

| Filter              | Description                                      |
|---------------------|--------------------------------------------------|
| Ideal Low-Pass      | Sharp cutoff at a specific frequency.            |
| Butterworth LPF     | Smooth cutoff with configurable order.           |
| Gaussian LPF        | Smooth Gaussian filter in frequency space.       |

---

## 📏 Evaluation

- **PSNR** is used to compare the denoised image with the original.
- Formula:  
  \[
  PSNR = 10 \cdot \log_{10} \left( \frac{MAX^2}{MSE} \right)
  \]

---

## 🛠️ Technologies Used

- Python 🐍
- OpenCV
- NumPy
- Matplotlib
- scikit-image

---

## 📊 Sample Results

| Filter               | Noise Type       | PSNR (dB) |
|----------------------|------------------|-----------|
| Arithmetic Mean      | Salt & Pepper    | 25.3      |
| Geometric Mean       | Gaussian         | 26.8      |
| Harmonic Mean        | Salt & Pepper    | 28.1      |
| Gaussian LPF         | Gaussian         | 27.5      |
| Butterworth LPF (n=2)| Salt & Pepper    | 24.9      |
| Ideal LPF            | Gaussian         | 23.7      |

---

## 📌 Conclusion

- **Harmonic Mean Filter** is most effective against salt noise.
- **Geometric Mean Filter** works better for Gaussian noise.
- **Gaussian LPF** outperforms other frequency filters in balance and detail preservation.

---

## ▶️ Usage

1. Clone the repo:
   ```bash
   git clone https://github.com/yourusername/image-processing-lab.git
   cd image-processing-lab
# pip install -r requirements.txt
# python main.py
