#!/usr/local//bin/python3

import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure
import os
from sklearn.cluster import KMeans
import random
import copy
from transformation import transform


def distance(x1, y1, x2, y2, proj_matrix):
    
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    try:
        point1 = np.transpose(np.matrix([x1, y1, 1]))
        estimated_point2 = np.dot(proj_matrix, point1)
        estimated_point2 = (1/estimated_point2.item(2))*estimated_point2
    
        point2 = np.transpose(np.matrix([x2, y2, 1]))
        error = np.linalg.norm(point2 - estimated_point2)

    except:
        return np.inf
    
    return error

# Source: https://stackoverflow.com/questions/20259025/module-object-has-no-attribute-drawmatches-opencv-python
def drawMatches_ransac(img1, img2, final_inliers):
    """
    This function takes in two images with their associated 
    keypoints, as well as a list of DMatch data structure (matches) 
    that contains which keypoints matched in which images.

    An image will be produced where a montage is shown with
    the first image followed by the second image beside it.

    Keypoints are delineated with circles, while lines are connected
    between matching keypoints.

    img1,img2 - Grayscale images
    kp1,kp2 - Detected list of keypoints through any of the OpenCV keypoint 
              detection algorithms
    matches - A list of matches of corresponding keypoints through any
              OpenCV keypoint matching algorithm
    """

    # Create a new output image that concatenates the two images together
    # (a.k.a) a montage
    rows1 = img1.shape[0]
    cols1 = img1.shape[1]
    rows2 = img2.shape[0]
    cols2 = img2.shape[1]

    out = np.zeros((max([rows1,rows2]),cols1+cols2,3), dtype='uint8')

    # Place the first image to the left
    out[:rows1,:cols1,:] = np.dstack([img1, img1, img1])

    # Place the next image to the right of it
    out[:rows2,cols1:cols1+cols2,:] = np.dstack([img2, img2, img2])

    for i in final_inliers:

        (x1,y1) = (i[0], i[1])
        (x2,y2) = (i[2], i[3])

        cv2.circle(out, (int(x1),int(y1)), 4, (255, 0, 0), 1)   
        cv2.circle(out, (int(x2)+cols1,int(y2)), 4, (255, 0, 0), 1)

        cv2.line(out, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), (255, 0, 0), 1)


    figure(figsize=(15, 6))
    plt.imshow(out, interpolation = 'bicubic')
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.show()

def ransac(n, dist, kp1, kp2, pixel_room = 2, inlier_coverage_threshold = 0.8):
    """
	:param n: Type of transformation, 1 for translation, 2 for Translation and Rotation, 3 for Affine and 4 for Projective
	:param dist: Ratio of euclidean distance, key points of first and second image, and descriptors of first and second image
        :param kp1, kp2: key points of image1 and image2
        :param pixel_room: integer value for pixel room error
        :param inlier_coverage_threshold: value between 0 to 1 giving the coverage of inliers
	:return: final_inliers and projection matrix and flag 1 if projection matrix is computed else 0
	"""
    
    
    final_inliers = []
    final_proj = []
    count = 0
    for x in range(1000):
        kp_sub_x = []
        kp_sub_y = []

        for i in range(n):
            key = list(dist.keys())[random.randrange(0,len(dist))]
            kp_sub_x.append(int(kp1[key[0]].pt[0]))
            kp_sub_y.append(int(kp1[key[0]].pt[1]))
            kp_sub_x.append(int(kp2[key[1]].pt[0]))
            kp_sub_y.append(int(kp2[key[1]].pt[1]))

        proj, flag = transform(n, kp_sub_x, kp_sub_y)
        
        if flag == 1:
            count += 1
            inliers = []
            for i in list(dist.keys()):
                error = distance(kp1[i[0]].pt[0], kp1[i[0]].pt[1], kp2[i[1]].pt[0], kp2[i[1]].pt[1], proj)
                if error < pixel_room:
                    inliers.append((int(kp1[i[0]].pt[0]), int(kp1[i[0]].pt[1]), int(kp2[i[1]].pt[0]), int(kp2[i[1]].pt[1])))

            if len(inliers) > (len(final_inliers)):
                final_inliers = copy.deepcopy(inliers)
                final_proj = copy.deepcopy(proj)
            
            if len(final_inliers) > len(dist)*inlier_coverage_threshold:
                break
            

            
    if count > 0:
        return final_inliers, final_proj, 1
    else:
        return final_inliers, final_proj, 0
