#!/usr/local//bin/python3

import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure
import os
from sklearn.cluster import KMeans
from kmeans_bovw import clustering

# Load the images as a list from the directory
images = []
for file in os.listdir('b657-a2/part1-images/'):
    images.append('b657-a2/part1-images/'+file)
    
descriptor_list = []
descriptors_final = []

orb = cv2.ORB_create(nfeatures = 500)

for i in range(len(images)):
    img = cv2.imread(images[i], cv2.IMREAD_GRAYSCALE)
    (keypoints, descriptors) = orb.detectAndCompute(img, None)
    descriptors_final.append(descriptors)
    descriptor_list.extend(descriptors)
    
clusters_final = clustering(descriptor_list, descriptors_final, 100, images)