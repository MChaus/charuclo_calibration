#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: mchaus
"""

from charuco_calibration import CalibratedCamera
import cv2
import numpy as np

if __name__ == '__main__':

    web_cam_wall = CalibratedCamera(
            squares_x=5,
            squares_y=8,
            square_length=100,
            marker_length=50,
            camera_matrix=np.array([
            [2265.4,   0,   1009.4],
            [0,     2268.7, 811.5],
            [0,     0,      1]
            ]),
            dist_coeff=np.array([[-0.0276, 0.1141, 0.0000871, -0.0002941]]),
            image_size=(1536, 2048)
            )

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

    kinect_ir.draw_markers(image=cv2.flip(cv2.imread(r'D:\brs\25_05_18\1527273719\DataSource\cam_5\00018.png'), 1))
    kinect_ir.estimate_board_pose(image=cv2.flip(cv2.imread(r'D:\brs\25_05_18\1527273719\DataSource\cam_5\00018.png'), 1))
    kinect_ir.draw_axis(image=cv2.flip(cv2.imread(r'D:\brs\25_05_18\1527273719\DataSource\cam_5\00018.png'), 1))
