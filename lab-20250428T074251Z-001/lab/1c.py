#Task 1 - Resolution, Histogram & Thresholding
#Task 1(c): Take a grayscale image of size 512x512, 
#illustrate the Histogram of the image & 
#make Single Threshold Segmentation observed from the histogram
#Importing the Libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt
#Function for generating Histogram
def generate_histogram(image,d):
    gray_levels_count = np.zeros(256)
    height, width = image.shape

    for r in range (width):
        for c in range(height):
            gray_levels_count[image[c, r]] += 1

    plt.subplot(2, 2, d)
    plt.bar(range(256), gray_levels_count, width = 1.0, color = "gray")
    plt.title(f"Histogram ")
    #plt.show()
#Loading the Original Image
original_images = cv2.imread("./Skeleton 750x1482.tif", cv2.IMREAD_GRAYSCALE)
plt.subplot(2, 2, 1)
plt.imshow(cv2.cvtColor(original_images, cv2.COLOR_BGR2RGB))
plt.title(f"original image")
generate_histogram(original_images, 2)
#Making Single Threshold Segmentation observed from the Histogram
threshold_intensity = 27
segmented_image = np.where(original_images < threshold_intensity, 0, 255)
segmented_image = np.uint8(segmented_image)
plt.subplot(2, 2, 3)
plt.imshow(cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB))
plt.title("The Segmented Image")

#Showing the Histogram of the Segmented Image
generate_histogram(segmented_image,4)
plt.show()