#Task 1 - Resolution, Histogram & Thresholding
#Task 1(b): Take a grayscale image of size 512x512, 
#decrease its intensity level resolution by one bit 
#up to reach its binary format & observe its change 
#when displaying in the same window size
#Importing the Libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt
#Function for Decreasing Intensity Level Resolution by 1-Bit
def decrease_resolution(image, number_of_bits):
    #Double Star or (**) is one of the Arithmetic Operator (Like +, -, *, **, /, //, %) in Python Language. 
    #It is also known as Power Operator.
    step = 255 / (2**number_of_bits - 1)
    height, width = image.shape
    decreased_image = image.copy()

    for r in range(height):
        for c in range(width):
            decreased_image[r, c] = round(image[r, c] / step) * step
        
    return decreased_image

#Loading the Original Image
original_image = cv2.imread("./Skull 374x452.tif", 0)
#Decreasing Intensity Level Resolution by 1-Bit
decreased_image = original_image.copy()
plt.figure(figsize = (13, 8))

for k in range(1, 9):
    plt.subplot(2, 4, k)
    number_of_bits = 9 - k
    decreased_image = decrease_resolution(decreased_image, number_of_bits)
    plt.imshow(cv2.cvtColor(decreased_image, cv2.COLOR_BGR2RGB))
    plt.title(f"{number_of_bits}-Bits Image")

plt.show()