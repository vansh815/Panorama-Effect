#!/usr/local//bin/python3

# -*- coding: utf-8 -*-
"""a2.py
serves as the main function for assignment 2
arguments should be passed according to which part it wants to test on

part1 will take the list of images and output file name
ex) ./a2 part1 k img_1.png img_2.png ... img_n.png output_file.txt


part2 will take the target and source image, the output image name
and the list of coordinates
ex) ./a2 part2 4 book1.jpg book2.jpg book_output.jpg 318,256 141,131 534,372 480,159 316,670 493,630 73,473 64,601


part3 will take the target and source image, the output image name
ex) ./a2 part3 image_1.jpg image_2.jpg output.jpg


"""

import os
import sys
import cv2
from clustering import cluster
from transformation import transform
from interpolation import bilinear_interpolation, panorama
from ransac import ransac, drawMatches_ransac
from PIL import Image
from orb import orb_features, key_desc
import numpy as np


if __name__ == '__main__':
  if sys.argv[1] == 'part1':
    k = int(sys.argv[2])
    image_list = sys.argv[3:-1]
    
    print_list = cluster(image_list, n_cluster=k)
    
    output_file = open(sys.argv[-1], 'w')
    for i in range(k):
      print(' '.join(print_list[i]), file=output_file)

    output_file.close()
        
      
  elif sys.argv[1] == 'part2':
    coord_len = int(sys.argv[2])
    target_img = cv2.imread(sys.argv[3])
    source_img = cv2.imread(sys.argv[4])
    output_img = sys.argv[5]
    coord_list = sys.argv[6:]
    x_coord = []
    y_coord = []
    for coord in coord_list:
      x, y = coord.split(',')
      x_coord.append(int(x))
      y_coord.append(int(y))
    t_mat, flag = transform(coord_len, x_coord, y_coord)
    if flag != 0:
      new_img = bilinear_interpolation(source_img, np.linalg.inv(t_mat), target_img)[0]
      cv2.imwrite(output_img, new_img)
    else:
        print('error in finding the transformation matrix!')
         
    

  elif sys.argv[1] == 'part3':
    target_img = cv2.imread(sys.argv[2], cv2.IMREAD_GRAYSCALE)
    source_img = cv2.imread(sys.argv[3], cv2.IMREAD_GRAYSCALE)
    output_img = sys.argv[4]
    threshold = 0.8
    
    orb = cv2.ORB_create(nfeatures=500)
    kp1, desc1 = key_desc(orb, target_img)
    kp2, desc2 = key_desc(orb, source_img)
    dist = orb_features(kp1, kp2, desc1, desc2)

    dist1 = {key:value for key,value in dist.items() if (value <= threshold)}

    final_inliers, final_proj, flag = ransac(4, dist1, kp1, kp2)

    target_img = cv2.imread(sys.argv[2])
    source_img = cv2.imread(sys.argv[3])

    if flag != 0:
      new_img, min_x, max_x, min_y, max_y = bilinear_interpolation(source_img, np.linalg.inv(final_proj), target_img)
      panorama_img = panorama(new_img, target_img, min_x, max_x, min_y, max_y)
      cv2.imwrite(output_img, panorama_img)
    else:
      print('error in running RANSAC!')
    

  else:
    print('invalid argument!')
    
