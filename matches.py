#!/usr/local//bin/python3
import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure
import os

# Source: https://stackoverflow.com/questions/20259025/module-object-has-no-attribute-drawmatches-opencv-python
def drawMatches(img1, img2, dist, kp1, kp2):
    """
    This function takes in two images with their associated 
    keypoints, as well as a list of DMatch data structure (matches) 
    that contains which keypoints matched in which images.

    An image will be produced where a montage is shown with
    the first image followed by the second image beside it.

    Keypoints are delineated with circles, while lines are connected
    between matching keypoints.

    img1,img2 - Images
    dist      - distance obtained from orb_features
    kp1,kp2 - Detected list of keypoints through any of the OpenCV keypoint 
              detection algorithms
    """

    rows1 = img1.shape[0]
    cols1 = img1.shape[1]
    rows2 = img2.shape[0]
    cols2 = img2.shape[1]

    out = np.zeros((max([rows1,rows2]),cols1+cols2,3), dtype='uint8')

    out[:rows1,:cols1,:] = np.dstack([img1, img1, img1])

    out[:rows2,cols1:cols1+cols2,:] = np.dstack([img2, img2, img2])

    for key,value in dist.items():

        (x1,y1) = (kp1[key[0]].pt[0], kp1[key[0]].pt[1])
        (x2,y2) = (kp2[key[1]].pt[0], kp2[key[1]].pt[1])

        cv2.circle(out, (int(x1),int(y1)), 4, (255, 0, 0), 1)   
        cv2.circle(out, (int(x2)+cols1,int(y2)), 4, (255, 0, 0), 1)
        cv2.line(out, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), (255, 0, 0), 1)


    figure(figsize=(15, 6))
    plt.imshow(out, interpolation = 'bicubic')
    plt.xticks([]), plt.yticks([])  
    plt.show()