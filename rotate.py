#####################################################
#
#   Instructions to run the script:
#       python codefy.py in.png input_msg.txt plane_nbr out.png
#
######################################################

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import cv2
from  skimage import filters, measure
from scipy import ndimage
from skimage.transform import (hough_line, hough_line_peaks,
                               probabilistic_hough_line)

def cost(img, theta):
    img_new = ndimage.interpolation.rotate(img, theta)
    n = img_new.shape[0]
    bin = [0] * n
    sum = 0
    for row in range(0, n):
        for col in range(0, img_new.shape[1]):
            bin[row] = bin[row] + img_new[row][col]
            
        if row > 0: 
            sum = sum + abs(bin[row-1] - bin[row])**2

    return sum
    
def compTheta(img, initTheta, finalTheta):
    resultTheta = 0
    maxCost = cost(img, initTheta)
    for theta in range(initTheta+1, finalTheta + 1 ):
        aux = cost(img, theta)
        if maxCost < aux:
            maxCost = aux
            resultTheta = theta
    return resultTheta
    
def horizontal_projection(img):
    theta = compTheta(img, 0, 180)
    if theta > 90:
        theta = theta - 180
    return theta
    
def compHough(img):
    n, theta, d = hough_line(img)
    i, j = np.unravel_index(n.argmax(), n.shape)
    if j > 90:
        j = j - 180
    return j
    
# load image 
img_file = os.getcwd() + "\\" + sys.argv[1]
img = cv2.imread(img_file)

# turning image into grayscale image
binary = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
binary[binary > 128] = 255 # white
binary[binary <= 128] = 0 # black
binary[binary == 0 ] = 1
binary[binary == 255] = 0


# binary image 
print("INFO: \tComputing degree using horizontal projection")
theta_proj = horizontal_projection(binary)
print("INFO: \tTheta found: "+ str(theta_proj))

img_out1 = ndimage.interpolation.rotate(img, theta_proj)
img_file = img_file[:-4]
cv2.imwrite(img_file+"proj.png",img_out1)
print("INFO: \tUpdated image saved at: " + img_file+"proj.png")

print("INFO: \tComputing degree using Hough Lines")
theta_hough = compHough(binary)
print("INFO: \tTheta found: "+ str(theta_hough))
img_out2 = ndimage.interpolation.rotate(img, theta_hough)

cv2.imwrite(img_file+"hough.png",img_out2)
print("INFO: \tUpdated image saved at: " + img_file+"hough.png")

