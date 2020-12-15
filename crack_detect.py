# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 16:16:26 2020

@author: Hao Cheam

Crack detection
Non data based method. Uses traditional CV.

Method:
1. grayscale
2. bilateral and gaussian filtering to denoise. median filtering to get rid of "salt-and-pepper"
    in the asphalt
3. image log (+normalize) to emphasize dark cracks 
4. canny edge detection
5. close filter to connect blobs
6. pick biggest blobs to clean up the result

Notes:
Biggest hurdle is finding a way to ignore all the background noise that is present in the
asphalt or cement. Could probably fine-tune the filter params to achieve better results.
Need better ways to handle shadows, lane paintings, etc.
"""

import cv2
import numpy as np

input_img_str = "sample_images/071.jpg"
img_input = cv2.imread(input_img_str)

# grayscale
img_gray = cv2.cvtColor(img_input, cv2.COLOR_BGR2GRAY)
cv2.imshow("Grayscale Input", img_gray)

# filtering
img_bilat = cv2.bilateralFilter(img_gray,5,75,55)
med = cv2.medianBlur(img_bilat, 3)
med = cv2.GaussianBlur(med,(3,3), 0)
# cv2.imshow("filtered", med)

# Apply logarithmic transform to emphasize dark areas
# this approach assumes cracks are dark pixels
c = 255 / np.log(1 + np.max(med)) 
img_log = c * (np.log(med + 1)) 
img_log = np.array(img_log,dtype=np.uint8)
img_log = cv2.normalize(img_log, None, 0, 255, cv2.NORM_MINMAX, dtype = cv2.CV_8U)
# cv2.imshow("log", img_log)

# canny edge detection
edges = cv2.Canny(img_log,100,250)
# cv2.imshow("canny", edges)

# close filter
kernel = np.ones((9,9), np.uint8)
closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
# cv2.imshow('open', closed)


# find biggest blobs
num_blobs = 5
im2, contours, hierarchy = cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
c = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)

for i in range(num_blobs):
    cv2.drawContours(img_gray, c, i, (0,255,0), 3)

cv2.imshow('Cracks', img_gray)














