#Task 1 - Resolution, Histogram & Thresholding
#Task 1(a): Take a grayscale image of size 512x512, 
#decrease its spatial resolution by half every time 
#& observe it's change when displaying in the same window size

#Importing the Libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt
#Function for Decreasing Spatial Resolution by Half
def decrease_resolution(image):
    height, width = image.shape
    ##the floor division // rounds the result down to the nearest whole number
    #9//2 = 4 but 9/2 = 4.5
    decreased_image = np.zeros((height // 2, width // 2))

    for r in range(0, height, 2):
        for c in range(0, width, 2):
            decreased_image[r // 2, c // 2] = image[r, c]
    #uint8 = unsigned int, 8 bits    
    return np.uint8(decreased_image)
#Loading the Original Image
original_image = cv2.imread("./Rose 1024x1024.tif", cv2.IMREAD_GRAYSCALE)

original_image = cv2.resize(original_image, (512, 512))
#Decreasing the Spatial Resolution by Half

decreased_image = original_image.copy()
#figsize=(13, 13): The figsize parameter is optional, 
#and when specified, it determines the width and height 
#of the figure in inches. In this case, the width is set to 13 inches, 
#and the height is set to 13 inches.
plt.figure(figsize = (13, 13))

for k in range (1, 5):
    plt.subplot(2, 2, k)
    plt.imshow(decreased_image, cmap = 'gray')
    height, width = decreased_image.shape
    plt.title(f"{height}x{width}")
    decreased_image = decrease_resolution(decreased_image)

plt.show()