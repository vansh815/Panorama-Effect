#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 22:06:14 2020

@author: vanshsmacpro
"""

import cv2
import numpy as np
from numpy import matmul
import sys
import matplotlib.pyplot as plt
from PIL import Image
import imageio

# nearest neighbour interpolation 

def nearest_neighbour(input_img , transformation_matrix) :
  new_img = np.empty((800, 600, 3), dtype=np.uint8) # input shape size and assigning the new matrix 
  x_max, y_max = input_img.shape[0] - 1, input_img.shape[1] - 1
  T_inv = np.linalg.inv(transformation_matrix) # taking inverse of the matrix 
  
  
  for i in range(800):
      
    
    for j in range(600):
        
            
        
      pts = np.array([i,j,1])
      x , y , _ = pts.dot(T_inv)# multiplying inverse and coordinates  
  
      ## code for finding the nearest neighbours.
      if np.floor(x) == x and np.floor(y) == y:
        x, y = int(x), int(y)
        new_img[i, j] = input_img[x,y]
      else :
        if np.abs(np.floor(x) - x) < np.abs(np.ceil(x) - x):
          x = int(np.floor(x))
        else:
          x = int(np.ceil(x))
        if np.abs(np.floor(y) - y) < np.abs(np.ceil(y) - y):
          y = int(np.floor(y))
        else:
          y = int(np.ceil(y))
        if x > x_max :
          x = x_max
        if y > y_max:
          y = y_max
        new_img[i, j] = input_img[x, y]
        
          
  plt.imshow(new_img)
  im = Image.fromarray(new_img)
  im.save('/Users/vanshsmacpro/Downloads/test.png')    
      
            
def bilinear_interpolation(input_img ,transformation_matrix ,target_img ):  
  new_img = np.zeros((input_img.shape[0]+ (2 * target_img.shape[0]) , input_img.shape[1] + (2 * target_img.shape[1]) , input_img.shape[2]), dtype=np.uint8)
  x_len , y_len , z = input_img.shape
  max_x = 0
  max_y = 0
  min_x = np.inf
  min_y = np.inf
  T_inv = np.linalg.inv(np.transpose(transformation_matrix))
  for r in range(-1 * target_img.shape[0], input_img.shape[0]+target_img.shape[0]):
    for c in range(-1 * target_img.shape[1], input_img.shape[1] + target_img.shape[1]):
      pt = np.array([c , r , 1])
      y, x, w = pt.dot(T_inv)
      x = x/w
      y = y/w

      #Check the bounds of the inverse pts we got and if they lie in the original image,
      #then copy the color from that original pt to the new matrix/image.
      if 0 <= y <= y_len-1 and 0 <= x <= x_len-1:
        x0 = int(np.floor(x))
        x1 = x0 + 1
        y0 = int(np.floor(y))
        y1 = y0 + 1

        Ia = input_img[x0, y0]
        Ib = input_img[x1, y0]
        Ic = input_img[x0, y1]
        Id = input_img[x1, y1]

        color1 = abs(x1-x) * abs(y1-y) * Ia
        color2 = abs(x1-x) * abs(y-y0) * Ib
        color3 = abs(x-x0) * abs(y1-y) * Ic
        color4 = abs(x-x0) * abs(y-y0) * Id

        weighted_avg_color = color1 + color2 + color3 + color4
        new_img[r+target_img.shape[0], c+target_img.shape[1]] = weighted_avg_color
        if max_x < r:
          max_x = r
        if max_y < c:
          max_y = c
        if min_x > r:
          min_x = r
        if min_y > c:
          min_y = c
      else : 
        new_img[r+target_img.shape[0],c+target_img.shape[1]] = [0, 0, 0]
            
  transformed_img = new_img[min_x + target_img.shape[0]:max_x+1 + target_img.shape[0], min_y +  + target_img.shape[1]:max_y+1 + target_img.shape[1]]
  return transformed_img, min_x, max_x, min_y, max_y

def panorama(transformed_img , target_img, min_x, max_x, min_y, max_y):

  target_x , target_y , _ = target_img.shape
  x_start = min([min_x, 0])
  x_fin = max([max_x, target_x-1])
  y_start = min([min_y, 0])
  y_fin = max([max_y, target_y-1])
  out_img = np.zeros(shape=(x_fin-x_start+1, y_fin-y_start+1, 3))
  for i in range(x_start, out_img.shape[0] + x_start) :
    for j in range(y_start, out_img.shape[1] + y_start):
      if 0 <= i < target_x and 0 <= j < target_y:
        out_img[i-x_start][j-y_start] = target_img[i][j]
      elif min_x <= i <= max_x and min_y <= j <= max_y:
        out_img[i-x_start][j-y_start] = transformed_img[i-min_x][j-min_y]
      else:
        out_img[i-x_start][j-y_start] = [0, 0, 0]

  return out_img
    
