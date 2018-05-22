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

    web_cam = CalibratedCamera(
            squares_x=4,
            squares_y=6,
            square_length=60,
            marker_length=30,
            camera_matrix=np.array([
            [2265.4,   0,   1009.4],
            [0,     2268.7, 811.5],
            [0,     0,      1]
            ]),
            dist_coeff=np.array([[-0.0276, 0.1141, 0.0000871, -0.0002941]]),
            image_size=(1536, 2048)
            )

    web_cam = CalibratedCamera(
            squares_x=4,
            squares_y=6,
            square_length=60,
            marker_length=30,
            camera_matrix=np.array([
            [2265.4,   0,   1009.4],
            [0,     2268.7, 811.5],
            [0,     0,      1]
            ]),
            dist_coeff=np.array([[-0.0276, 0.1141, 0.0000871, -0.0002941]]),
            image_size=(1536, 2048)
            )

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

    # IR Kinect from MatLab.
    kinect_ir = CalibratedCamera(
            squares_x=4,
            squares_y=6,
            square_length=60,
            marker_length=30,
            camera_matrix=np.array([
            [363.1938,         0,   258.8109],
            [       0,  363.0246,   208.8607],
            [       0,         0,          1]
            ]),
            dist_coeff=np.array([[0.0799, -0.1877, 0.0010, 0.0002]]),
            image_size=(424, 512)
            )

    rmatr_ir_color = np.array( [[ 1.0000,    0.0037,    0.0067,],
                                [-0.0037,    1.0000,    0.0043,],
                                [-0.0067,    -0.0043,    1.0000,]])

    tvec_ir_color = np.array([[-51.5740], [0.8744], [-1.3294]])

    rmatr_basler_color = cv2.Rodrigues(np.array([[0.06869832, 0.03180862, 0.0087055 ]]))[0]
    tvec_basler_color = np.array([[ 98.3966582],  [-44.34708768],  [-6.71150373]])

    # Find extrinsic matrix from basler camera to color Kinect camera
    image_charuco_color = cv2.imread('images/color_kinect/00428.png')
    image_charuco_color = cv2.flip(image_charuco_color, 1)
    image_charuco_basler = cv2.imread('images/basler/00428.png')
    image_charuco_basler = cv2.flip(image_charuco_basler, 1)
    rmatr_basler_color, tvec_basler_color = basler_ir.get_disposition_charuco(image_charuco_basler, image_charuco_color, kinect_color)
    print(rmatr_basler_color, tvec_basler_color)

    # Calirate wall using web cam
    image_wall_web_cam = cv2.imread('images/web_cam/charuco_web_cam_wall.jpg')
    rmatr_wall_web_cam, tvec_wall_web_cam = web_cam.estimate_board_pose(image_wall_web_cam)

    # Calirate screen using web cam
    image_web_cam_screen = cv2.imread('images/web_cam/charuco_web_cam_screen.jpg')
    rmatr_screen_web_cam, tvec_screen_web_cam = web_cam.estimate_board_pose(image_web_cam_screen)

    # Find extrinsic matrix from web cam to color
    image_charuco_color = cv2.imread('images/color_kinect/charuco_color_1.png')
    image_charuco_color = cv2.flip(image_charuco_color, 1)
    image_charuco_web_cam = cv2.imread('images/web_cam/charuco_web_cam_color_1.jpg')
    rmatr_web_cam_color, tvec_web_cam_color = web_cam.get_disposition_charuco(image_charuco_web_cam, image_charuco_color, kinect_color)

    extrinsic_matrix_ir_color = kinect_ir.extrinsic_matrix(rmatr_ir_color, tvec_ir_color)
    extrinsic_matrix_basler_color = basler_ir.extrinsic_matrix(rmatr_basler_color, tvec_basler_color)
    extrinsic_matrix_wall_web_cam = web_cam.extrinsic_matrix(rmatr_wall_web_cam, tvec_wall_web_cam)
    extrinsic_matrix_screen_web_cam = web_cam.extrinsic_matrix(rmatr_screen_web_cam, tvec_screen_web_cam)
    extrinsic_matrix_web_cam_clolor = web_cam.extrinsic_matrix(rmatr_web_cam_color, tvec_web_cam_color)

    extrinsic_matrix_color_ir = LA.inv(extrinsic_matrix_ir_color)
    extrinsic_matrix_web_cam_ir = extrinsic_matrix_color_ir @ extrinsic_matrix_web_cam_clolor
    extrinsic_matrix_screen_ir = extrinsic_matrix_web_cam_ir @ extrinsic_matrix_screen_web_cam
    extrinsic_matrix_wall_ir = extrinsic_matrix_web_cam_ir @ extrinsic_matrix_wall_web_cam
    extrinsic_matrix_basler_ir = extrinsic_matrix_color_ir @ extrinsic_matrix_basler_color

    extrinsic_matrices = {
        'extrinsic_matrix_color_ir' : extrinsic_matrix_color_ir.tolist(),
        'extrinsic_matrix_web_cam_ir' : extrinsic_matrix_web_cam_ir.tolist(),
        'extrinsic_matrix_screen_ir' : extrinsic_matrix_screen_ir.tolist(),
        'extrinsic_matrix_wall_ir' : extrinsic_matrix_wall_ir.tolist(),
        'extrinsic_matrix_basler_ir' :  extrinsic_matrix_basler_ir.tolist()
    }

    with open('extrinsic_matrices.json', 'w') as _file:
        json.dump(extrinsic_matrices, _file, ensure_ascii=False)
        _file.close()

    # with open('extrinsic_matrices.json', 'r') as fp:
    #     data = json.load(fp)
    # print(data)
