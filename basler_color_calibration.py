#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: mchaus
"""

from charuco_calibration import CalibratedCamera
import cv2
import numpy as np
from PIL import Image
import numpy.linalg as LA
import os
import json

if __name__ == '__main__':
    # Basler from MatLab.
    basler_ir = CalibratedCamera(
            squares_x=4,
            squares_y=6,
            square_length=60,
            marker_length=30,
            camera_matrix=np.array([
            [3789.0 ,        0,         0],
            [0,    3793.8,         0],
            [669.4,    491.5,    1.0]
            ]).T,
            dist_coeff=np.array([[-0.3491,    0.5147, 0.0001728,   -0.0003025]]),
            image_size=(972, 1296)
            )

    # Color Kinect from MatLab.
    kinect_color = CalibratedCamera(
            squares_x=4,
            squares_y=6,
            square_length=60,
            marker_length=30,
            camera_matrix=np.array([
            [1050.9, 0,  956.0],
            [0,     1050.0, 534.8],
            [0,     0,      1]
            ]),
            dist_coeff=np.array([[0.038, -0.0388, 0.0010, 0.0004]]),
            image_size=(1080, 1920)
            )

    all_rotation_matrices = []
    all_tanslation_vecotrs = []
    path_to_ir_basler=r'images/basler'
    path_to_color_kinect=r'images/color_kinect'
    for filename in os.listdir(path_to_ir_basler):
        basler_ir_image_path = os.path.join(path_to_ir_basler, filename)
        kinect_color_image_path = os.path.join(path_to_color_kinect, filename)
        if (
            os.path.isfile(basler_ir_image_path) and
            os.path.isfile(kinect_color_image_path)
            ):
            print(filename)
            basler_ir_image = cv2.imread(basler_ir_image_path)
            kinect_color_image = cv2.imread(kinect_color_image_path)

            basler_ir_image = cv2.flip(basler_ir_image, 1)
            kinect_color_image = cv2.flip(kinect_color_image, 1)

            rmatr_basler_color, tvec_basler_color = basler_ir.get_disposition_charuco(basler_ir_image, kinect_color_image, kinect_color)
            all_rotation_matrices.append(rmatr_basler_color)
            all_tanslation_vecotrs.append(tvec_basler_color)
    print('Rotation vectors:\n')
    for r_matr in all_rotation_matrices:
        r_vec = cv2.Rodrigues(r_matr)[0]
        print(r_vec.T, ' norm = ', LA.norm(r_vec))
    print('Translation vectors:\n')
    for t_vec in all_tanslation_vecotrs:
        print(t_vec.T, ' norm = ', LA.norm(t_vec))


    for filename in os.listdir(path_to_ir_basler):
        basler_ir_image_path = os.path.join(path_to_ir_basler, filename)
        color_kinect_image_path = os.path.join(path_to_color_kinect, filename)
        if os.path.isfile(basler_ir_image_path) :
            basler_ir_image = cv2.imread(basler_ir_image_path)
            basler_ir_image = cv2.flip(basler_ir_image, 1)
            color_image = cv2.imread(color_kinect_image_path)
            color_image = cv2.flip(color_image, 1)
            # cv2.imshow('Axis',
            #            basler_ir.check_calibration_charuco(basler_ir_image,
            #                                                show=False,
            #                                                square_width = 60,
            #                                                column_number=4,
            #                                                row_number=6,
            #                                                line_width=2,
            #                                                scale=4)[::2,::2,:])
            # cv2.imshow('Axis2',
            #            kinect_color.check_calibration_charuco(color_image,
            #                                                show=False,
            #                                                square_width = 60,
            #                                                column_number=4,
            #                                                row_number=6,
            #                                                line_width=2,
            #                                                scale=4))
            #
            # cv2.waitKey()


    path_to_ir_image=r'images/basler/00428.png'
    path_to_color_image=r'images/color_kinect/00428.png'

    basler_ir_image = cv2.imread(path_to_ir_image)
    kinect_color_image = cv2.imread(path_to_color_image)

    basler_ir_image = cv2.flip(basler_ir_image, 1)
    kinect_color_image = cv2.flip(kinect_color_image, 1)

    for r_matr, t_vec in zip(all_rotation_matrices, all_tanslation_vecotrs):
        extrinsic_matrix = np.identity(4)
        extrinsic_matrix[0:3, 0:3] = r_matr
        extrinsic_matrix[0:3, 3] = t_vec[0:3, 0]
        extrinsic_matrix = LA.inv(extrinsic_matrix)
        kinect_color.check_transition(kinect_color_image, basler_ir_image,
                                    basler_ir, extrinsic_matrix[0:3, 0:3],
                                    extrinsic_matrix[0:3, 3].reshape(3, 1), column_number=4,
                                    row_number=6, square_width = 60,
                                    show=True, line_width=5, scale=2)
