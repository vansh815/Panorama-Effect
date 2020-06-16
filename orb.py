#!/usr/local//bin/python3

import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure
import os

def key_desc(orb, img):
    (keypoints, descriptors) = orb.detectAndCompute(img, None)
    return keypoints, descriptors

def orb_features(kp1, kp2, desc1, desc2):
    """
	:param kp1, desc1: return value of image1 from key_desc
	:param kp2, desc2: return value of image2 from key_desc
	:return: Ratio of euclidean distance, key points of first and second image, and descriptors of first and second image
	"""
    dist = {}
    for i in range(len(desc1)):
        dist_temp = {}
        for j in range(len(desc2)):
            dist_temp[(i,j)] = int(cv2.norm( desc1[i], desc2[j], cv2.NORM_L2))
        dist_temp = dict(sorted(dist_temp.items(), key = lambda x: x[1],))
        hamming_ratio = list(dist_temp.values())[0]/list(dist_temp.values())[1]
        
        dist[(list(dist_temp.keys())[0])] = hamming_ratio
    return dist

    
    
