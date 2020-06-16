#!/usr/local//bin/python3

import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure
import os
from sklearn.cluster import KMeans

def build_histogram(descriptor_list, cluster_alg):
    histogram = np.zeros(len(cluster_alg.cluster_centers_))
    cluster_result =  cluster_alg.predict(descriptor_list)
    for i in cluster_result:
        histogram[i] += 1.0
    return histogram

def clustering(descriptor_list, descriptors_final, clusters, images):
    """
	:param descriptor_list: Descriptors of each image as a list extended
	:param descriptors_final: Descriptors of each image separate
    :param clusters: Number of visual features required
    :param images: Image names as a list
	:return: Dictionary with key as cluster number and values as images
	"""
    np.random.seed(1)
    kmeans = KMeans(n_clusters = clusters,init='k-means++')
    kmeans.fit(descriptor_list)
    
    preprocessed_image = []
    for desc in descriptors_final:
        histogram = build_histogram(desc, kmeans)
        preprocessed_image.append(histogram)
        
    kmeans1 = KMeans(n_clusters = 10,init='k-means++')
    kmeans1.fit(preprocessed_image)
    
    final = {}
    for x,y in zip(images,kmeans1.labels_):
        if y not in final.keys():
            final[y] = [x]
        else:
            final[y] += [x]
            
    return final
    
    