#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: mchaus
"""

from charuco_calibration import CalibratedCamera
import cv2
import numpy as np
from PIL import Image

if __name__ == '__main__':
    pass
    # Basler from MatLab.
    basler_ir = CalibratedCamera(
            squares_x=4,
            squares_y=6,
            square_length=0.06,
            marker_length=0.03,
            camera_matrix=np.array([
            [1896.6, 0,   654.8],
            [0,     1897.9, 461.5],
            [0,     0,      1]
            ]),
            dist_coeff=np.array([[-0.6819, 0.3729, -0.0021, -0.0005]]),
            image_size=(972, 1296)
            )

    # Color Kinect from MatLab.
    kinect_color = CalibratedCamera(
            squares_x=4,
            squares_y=6,
            square_length=0.06,
            marker_length=0.03,
            camera_matrix=np.array([
            [1050.9, 0,  959.7],
            [0,     1051.4, 539.1],
            [0,     0,      1]
            ]),
            dist_coeff=np.array([[0.0431, -0.0425, 0.0011, 0.0011]]),
            image_size=(1080, 1920)
            )

    # IR Kinect from MatLab.
    kinect_ir = CalibratedCamera(
            squares_x=4,
            squares_y=6,
            square_length=0.06,
            marker_length=0.03,
            camera_matrix=np.array([
            [363.1938,         0,   258.8109],
            [       0,  363.0246,   208.8607],
            [       0,         0,          1]
            ]),
            dist_coeff=np.array([[0.0799, -0.1877, 0.0010, 0.0002]]),
            image_size=(424, 512)
            )

    charuco_basler_1 = cv2.imread('images/charuco_basler_1.png')
    charuco_basler_1 = cv2.flip(charuco_basler_1, 1)
    basler_ir.check_calibration_charuco(charuco_basler_1, show=True, line_width=1, square_width = 0.06, row_number = 6, column_number=4,)

    charuco_color_1 = cv2.imread('images/charuco_color_1.png')
    charuco_color_1 = cv2.flip(charuco_color_1, 1)
    kinect_color.check_calibration_charuco(charuco_color_1, square_width = 0.06, row_number = 6, column_number=4, show=True, line_width=1)

    rotation_matrix_to_color, translatioin_vector_to_color = basler_ir.get_disposition_charuco(charuco_basler_1, charuco_color_1, kinect_color)

    charuco_color_2 = cv2.imread('images/charuco_color_2.png')
    charuco_color_2 = cv2.flip(charuco_color_2, 1)
    charuco_basler_2 = cv2.imread('images/charuco_basler_2.png')
    charuco_basler_2 = cv2.flip(charuco_basler_2, 1)
    basler_ir.check_transition(charuco_basler_2, charuco_color_2, kinect_color,
                               rotation_matrix_to_color, translatioin_vector_to_color,
                               column_number=4, row_number=6, square_width = 0.06,
                               show=True, line_width=5)

    rotation_matrix_to_basler, translatioin_vector_to_basler = kinect_color.get_disposition_charuco(charuco_color_1, charuco_basler_1, basler_ir)

    kinect_color.check_transition(charuco_color_2, charuco_basler_2, basler_ir,
                                  rotation_matrix_to_basler, translatioin_vector_to_basler,
                                  column_number=4, row_number=6, square_width = 0.06,
                                  show=True, line_width=5)
