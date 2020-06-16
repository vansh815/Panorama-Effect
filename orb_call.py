#!/usr/local//bin/python3

"""
This is a sample code to run orb_features and draw matches between images
"""
from orb import orb_features
import cv2
from matches import drawMatches

img1 = cv2.imread("/Users/rs/Documents/Indiana University/2nd Semester/CV/FInal project/data/pedestrian_crosswalk_sign/crosswalk0.jpg", cv2.IMREAD_GRAYSCALE)
img = cv2.imread("/Users/rs/Documents/Indiana University/2nd Semester/CV/FInal project/data/pedestrian_crosswalk_sign/crosswalk2.jpg", cv2.IMREAD_GRAYSCALE)

dist, kp1, kp2, desc1, desc2 = orb_features(img, img1)

# Setting the threshold
threshold = 0.8
# subsetting the key points based on the thresholds
dist1 = {key:value for key,value in dist.items() if (value <= threshold)}

# This draws the key points between the images
drawMatches(img, img1, dist1, kp1,kp2)
