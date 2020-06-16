#!/usr/local/bin/python3

import matplotlib as plt
import numpy as np
from numpy import matmul
import sys



def transform(coord_len, x_coord, y_coord):
    """
	:param coord_len: Type of transformation, 1 for translation, 2 for Translation and Rotation, 3 for Affine and 4 for Projective
	:param x_coord: stacked x coordinates - order image1, image2
    :param y_coord: stacked y coordinates - order image1, image2
	:return: transformation matrix and flag 1 if transformation matrix is computed else 0
	"""
    flag = 1
    # just setting values to make the equations visible
    if coord_len > 0:
        x_1 = x_coord[0]
        y_1 = y_coord[0]
        xp_1 = x_coord[1]
        yp_1 = y_coord[1]
    if coord_len > 1:
        x_2 = x_coord[2]
        y_2 = y_coord[2]
        xp_2 = x_coord[3]
        yp_2 = y_coord[3]
    if coord_len > 2:
        x_3 = x_coord[4]
        y_3 = y_coord[4]
        xp_3 = x_coord[5]
        yp_3 = y_coord[5]
    if coord_len > 3:
        x_4 = x_coord[6]
        y_4 = y_coord[6]
        xp_4 = x_coord[7]
        yp_4 = y_coord[7]

    # translation
    # just brute force equations
    if coord_len == 1:
        t_x = xp_1 - x_1
        t_y = yp_1 - y_1
        trans_mat = np.array([[1, 0, t_x], [0, 1, t_y], [0, 0, 1]])
    # rigid(translation + rotation)
    # just brute force equations
    if coord_len == 2:
        a = x_1 - x_2
        b = y_1 - y_2
        c = xp_1 - xp_2
        d = yp_1 - yp_2
        cand1 = (a*c - (b * np.sqrt(a**2 + b**2 - c**2)))/(a**2 + b**2)
        cand2 = (a*c + (b * np.sqrt(a**2 + b**2 - c**2)))/(a**2 + b**2)
        if cand1 or cand2 == nan:
            print("invalid!")
            return [], 0
        if abs(cand1) <= 1:
            cos_t = cand1
        elif abs(cand2) <= 1:
            cos_t = cand2
        else:
            print("invalid!")
            return [], 0
        sin_t = (a * cos_t - c) / b
        t_x = xp_1 - (x_1 * cos_t) + (y_1 * sin_t)
        t_y = yp_1 - (x_1 * sin_t) - (y_1 * cos_t)
        trans_mat = np.array([[cos_t, -1 * sin_t, t_x], [sin_t, cos_t, t_y], [0, 0, 1]])
    # affine(rigid + scale & mirror)
    # used inverse matrix
    if coord_len == 3:

        temp = np.array([[x_1, y_1, 1, 0, 0, 0],
                         [0, 0, 0, x_1, y_1, 1],
                         [x_2, y_2, 1, 0, 0, 0],
                         [0, 0, 0, x_2, y_2, 1],
                         [x_3, y_3, 1, 0, 0, 0],
                         [0, 0, 0, x_3, y_3, 1]])
        try:
            temp = np.linalg.inv(temp)
            temp_v = np.array([[xp_1], [yp_1], [xp_2], [yp_2], [xp_3], [yp_3]])
            temp = matmul(temp, temp_v)        
            trans_mat = np.array([[temp[0][0], temp[1][0], temp[2][0]], [temp[3][0], temp[4][0], temp[5][0]], [0, 0, 1]])
        except:
            return [],0
    # projection(affine + projective warps)
    # used inverse matrix
    if coord_len == 4:
        temp = np.array([[x_1, y_1, 1, 0, 0, 0, -1 * x_1 * xp_1, -1 * y_1 * xp_1],
                         [0, 0, 0, x_1, y_1, 1, -1 * x_1 * yp_1, -1 * y_1 * yp_1],
                         [x_2, y_2, 1, 0, 0, 0, -1 * x_2 * xp_2, -1 * y_2 * xp_2],
                         [0, 0, 0, x_2, y_2, 1, -1 * x_2 * yp_2, -1 * y_2 * yp_2],
                         [x_3, y_3, 1, 0, 0, 0, -1 * x_3 * xp_3, -1 * y_3 * xp_3],
                         [0, 0, 0, x_3, y_3, 1, -1 * x_3 * yp_3, -1 * y_3 * yp_3],
                         [x_4, y_4, 1, 0, 0, 0, -1 * x_4 * xp_4, -1 * y_4 * xp_4],
                         [0, 0, 0, x_4, y_4, 1, -1 * x_4 * yp_4, -1 * y_4 * yp_4]])
        try:
            temp = np.linalg.inv(temp)
            temp_v = np.array([[xp_1], [yp_1], [xp_2], [yp_2], [xp_3], [yp_3], [xp_4], [yp_4]])
            temp = matmul(temp, temp_v)
            trans_mat = np.array([[temp[0][0], temp[1][0], temp[2][0]], [temp[3][0], temp[4][0], temp[5][0]], [temp[6][0], temp[7][0], 1]])
        except:
            return [],0

    return trans_mat,flag
        
        
    
