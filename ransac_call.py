#!/usr/local//bin/python3

import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure
import os
from sklearn.cluster import KMeans
import random
import copy
from orb import orb_features
import cv2
from matches import drawMatches
from ransac import ransac, drawMatches_ransac

img = cv2.imread("/Users/rs/Documents/Indiana University/2nd Semester/CV/Assignment 2/b657-a2/part2-images/book1.jpg", cv2.IMREAD_GRAYSCALE)
img1 = cv2.imread("/Users/rs/Documents/Indiana University/2nd Semester/CV/Assignment 2/b657-a2/part2-images/book2.jpg",cv2.IMREAD_GRAYSCALE)

dist, kp1, kp2, desc1, desc2 = orb_features(img, img1)

# Setting the threshold
threshold = 0.8
# subsetting the key points based on the thresholds
dist1 = {key:value for key,value in dist.items() if (value <= threshold)}

# This draws the key points between the images
drawMatches(img, img1, dist1, kp1,kp2)

final_inliers, final_proj,flag = ransac(4, dist1, kp1, kp2, 5, 0.6)

drawMatches_ransac(img, img1, final_inliers)



