# -*- coding: utf-8 -*-
"""cluster.py
provides a cluster method for part1
can be used by calling cluster(args)
arguments are:

img_name : list of image names. make sure they are in the same directory.

n_cluster : number of clusters, default is 10

method : choose between; single, complete, average, kmeans, kmeans_his.
default is kmeans_his

* make sure that the images to be tested are in the
part1-images directory
also, make sure these are the only files in the directory.
"""

#!/usr/local//bin/python3

import cv2
import time
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure
import os
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import AgglomerativeClustering, KMeans
from orb import orb_features, key_desc

def build_histogram(descriptor_list, cluster_alg):
    histogram = np.zeros(len(cluster_alg.cluster_centers_))
    cluster_result =  cluster_alg.predict(descriptor_list)
    for i in cluster_result:
        histogram[i] += 1.0
    return histogram

def cluster(img_name, n_cluster=10, method='kmeans_his'):
  
  img_list = []
  desc_list = []
  desc_final = []
  threshold = 0.9

  orb = cv2.ORB_create(nfeatures=1000)
  #read the images
  n_img = len(img_name)

  for i in range(n_img):
    img_list.append(cv2.imread(img_name[i], cv2.IMREAD_GRAYSCALE))


  # make a distance matrix for evaluation
  # make a list of descriptors for k means
  img_dst = np.ones(shape=(n_img,n_img))
  img_values = {}
  for i in range(n_img):
    img_values[img_name[i]] = key_desc(orb, img_list[i])
      
  for i in range(n_img):
    kp1, desc1 = img_values[img_name[i]]
    if method == 'complete' or method == 'single' or method == 'average':
      for j in range(i+1, n_img):
        kp2, desc2 = img_values[img_name[j]]
        dist = orb_features(kp1, kp2, desc1, desc2)
        dist1 = {key:value for key,value in dist.items() if (value <= threshold)}
        img_dst[i, j] = 1/len(dist1)
        img_dst[j, i] = 1/len(dist1)
    elif method == 'kmeans' or method == 'kmeans_his':
      desc_list.extend(desc1)
      desc_final.append(desc1)

  if method == 'average':
    cluster = AgglomerativeClustering(n_clusters=n_cluster, affinity='precomputed', linkage='average').fit(img_dst)
  if method == 'single':
    cluster = AgglomerativeClustering(n_clusters=n_cluster, affinity='precomputed', linkage='single').fit(img_dst)
  if method == 'complete':
    cluster = AgglomerativeClustering(n_clusters=n_cluster, affinity='precomputed', linkage='complete').fit(img_dst)

  if method == 'kmeans' or method == 'kmeans_his':
    np.random.seed(1)
    patch_kmeans = KMeans(n_clusters=n_cluster)
    cluster = patch_kmeans.fit(desc_list)

  if method == 'kmeans_his':
    preprocessed_image = []
    for desc in desc_final:
      histogram = build_histogram(desc, patch_kmeans)
      preprocessed_image.append(histogram)

    cluster = KMeans(n_clusters = n_cluster).fit(preprocessed_image)
    
  print_list = [[] for i in range(n_cluster)]
  for i in range(n_img):
    print_list[cluster.labels_[i]].append(img_name[i])

  return print_list
  
