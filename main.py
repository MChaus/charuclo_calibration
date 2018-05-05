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
    basler_image = cv2.imread('images/basler_image.png')
    ir_image = cv2.imread('images/ir_image.png')
    color_image = cv2.imread('images/color_image.png')

    basler_rotation_matrix = np.array([[ 0.0241,   -0.9058,    0.4231],
                                       [ 0.9990,    0.0380,    0.0247],
                                       [-0.0384,    0.4220,    0.9058]])
    basler_translation_vector = np.array([204.4,    116.5,    1904.7])

    ir_rotation_matrix = np.array([[    -0.0485,   -0.8855,    0.4621],
                                   [     0.9974,   -0.0181,    0.0701],
                                   [    -0.0537,    0.4643,    0.8841]])
    ir_translation_vector = np.array([-11.2,    161.9,    1902.3])

    color_rotation_matrix = np.array([[ -0.0472,   -0.8855,    0.4623],
                                      [  0.9971,   -0.0141,    0.0748],
                                      [ -0.0597,    0.4645,    0.8836]])
    color_translation_vector = np.array([34.0,    161.4,    1899.5])

    # basler_ir.check_board(basler_image, basler_rotation_matrix.T, basler_translation_vector, show=True, line_width=1)
    # kinect_ir.check_board(ir_image, ir_rotation_matrix.T, ir_translation_vector, show=True, line_width=1)
    # kinect_color.check_board(color_image, color_rotation_matrix.T, color_translation_vector, show=True, line_width=1)

    rotation_matrix_to_basler = np.array([[ 0.9972,    0.0630,   -0.0393],
                                          [-0.0610,    0.9968,    0.0509],
                                          [ 0.0424,   -0.0484,    0.9979]])
    translatioin_vector_to_basler = np.array([135.9533,   48.9501,   10.0126])

#----------FROM ONE CAMERA TO ANOTHER-----------------
    ir_extrinsic_matrix = np.identity(4)
    ir_extrinsic_matrix[0:3, 0:3] = ir_rotation_matrix.T
    ir_extrinsic_matrix[0:3, 3] = ir_translation_vector
    ir_intrinsic_matrix = np.zeros((3, 4))
    ir_intrinsic_matrix[0:3, 0:3] = kinect_ir.camera_matrix
    ir_image = cv2.undistort(ir_image, kinect_ir.camera_matrix, kinect_ir.dist_coeff)

    basler_extrinsic_matrix = np.identity(4)
    basler_extrinsic_matrix[0:3, 0:3] = basler_rotation_matrix.T
    basler_extrinsic_matrix[0:3, 3] = basler_translation_vector
    basler_intrinsic_matrix = np.zeros((3, 4))
    basler_intrinsic_matrix[0:3, 0:3] = basler_ir.camera_matrix
    basler_image = cv2.undistort(basler_image, basler_ir.camera_matrix, basler_ir.dist_coeff)

    extrinsic_matrix_to_basler = np.identity(4)
    extrinsic_matrix_to_basler[0:3, 0:3] = rotation_matrix_to_basler.T
    extrinsic_matrix_to_basler[0:3, 3] = translatioin_vector_to_basler

    kinect_ir.check_transition(ir_image, basler_image, basler_ir,
                               rotation_matrix_to_basler.T, translatioin_vector_to_basler,
                               ir_rotation_matrix.T, ir_translation_vector, color_rotation_matrix.T, color_translation_vector,
                               column_number=6, row_number=6, square_width = 100,
                               show=True, line_width=3)

#     point_on_ir_image_1 = ir_extrinsic_matrix.dot([[0], [0], [0], [1]])
#     point_on_ir_image_1 = ir_intrinsic_matrix.dot(point_on_ir_image_1)
#     point_on_ir_image_1 /= point_on_ir_image_1[2, 0]
#
#     point_on_ir_image_2 = ir_extrinsic_matrix.dot([[100], [0], [0], [1]])
#     point_on_ir_image_2 = ir_intrinsic_matrix.dot(point_on_ir_image_2)
#     point_on_ir_image_2 /= point_on_ir_image_2[2, 0]
#
#     point_on_ir_image_3 = ir_extrinsic_matrix.dot([[0], [100], [0], [1]])
#     point_on_ir_image_3 = ir_intrinsic_matrix.dot(point_on_ir_image_3)
#     point_on_ir_image_3 /= point_on_ir_image_3[2, 0]
#
#     point_on_ir_image_4 = ir_extrinsic_matrix.dot([[100], [100], [0], [1]])
#     point_on_ir_image_4 = ir_intrinsic_matrix.dot(point_on_ir_image_4)
#     point_on_ir_image_4 /= point_on_ir_image_4[2, 0]
#
#     cv2.circle(ir_image, (int(point_on_ir_image_1[0, 0]), int(point_on_ir_image_1[1, 0])), 1, (0,0,255), -1)
#     cv2.circle(ir_image, (int(point_on_ir_image_2[0, 0]), int(point_on_ir_image_2[1, 0])), 1, (0,0,255), -1)
#     cv2.circle(ir_image, (int(point_on_ir_image_3[0, 0]), int(point_on_ir_image_3[1, 0])), 1, (0,0,255), -1)
#     cv2.circle(ir_image, (int(point_on_ir_image_4[0, 0]), int(point_on_ir_image_4[1, 0])), 1, (0,0,255), -1)
#     cv2.imshow('ir_point', ir_image)
#
#     point_in_ir_coordinates_1 = ir_extrinsic_matrix.dot([[0], [0], [0], [1]])
#     point_in_basler_coordinates_1 = extrinsic_matrix_to_basler.dot(point_in_ir_coordinates_1)
#     point_on_basler_image_1 = basler_intrinsic_matrix.dot(point_in_basler_coordinates_1)
#     point_on_basler_image_1 /= point_on_basler_image_1[2, 0]
#
#     point_in_ir_coordinates_2 = ir_extrinsic_matrix.dot([[100], [0], [0], [1]])
#     point_in_basler_coordinates_2 = extrinsic_matrix_to_basler.dot(point_in_ir_coordinates_2)
#     point_on_basler_image_2 = basler_intrinsic_matrix.dot(point_in_basler_coordinates_2)
#     point_on_basler_image_2 /= point_on_basler_image_2[2, 0]
#
#     point_in_ir_coordinates_3 = ir_extrinsic_matrix.dot([[0], [100], [0], [1]])
#     point_in_basler_coordinates_3 = extrinsic_matrix_to_basler.dot(point_in_ir_coordinates_3)
#     point_on_basler_image_3 = basler_intrinsic_matrix.dot(point_in_basler_coordinates_3)
#     point_on_basler_image_3 /= point_on_basler_image_3[2, 0]
#
#     point_in_ir_coordinates_4 = ir_extrinsic_matrix.dot([[100], [100], [0], [1]])
#     point_in_basler_coordinates_4 = extrinsic_matrix_to_basler.dot(point_in_ir_coordinates_4)
#     point_on_basler_image_4 = basler_intrinsic_matrix.dot(point_in_basler_coordinates_4)
#     point_on_basler_image_4 /= point_on_basler_image_4[2, 0]
#
#
#     cv2.circle(basler_image, (int(point_on_basler_image_1[0, 0]), int(point_on_basler_image_1[1, 0])), 4, (0,0,255), -1)
#     cv2.circle(basler_image, (int(point_on_basler_image_2[0, 0]), int(point_on_basler_image_2[1, 0])), 4, (0,0,255), -1)
#     cv2.circle(basler_image, (int(point_on_basler_image_3[0, 0]), int(point_on_basler_image_3[1, 0])), 4, (0,0,255), -1)
#     cv2.circle(basler_image, (int(point_on_basler_image_4[0, 0]), int(point_on_basler_image_4[1, 0])), 4, (0,0,255), -1)
#     cv2.imshow('basler_point', basler_image)
#     cv2.waitKey()
#
#
# #----------------------------------vice versa--------------------------------------------
#
#
#     point_on_basler_image_1 = basler_extrinsic_matrix.dot([[0], [0], [0], [1]])
#     point_on_basler_image_1 = basler_intrinsic_matrix.dot(point_on_basler_image_1)
#     point_on_basler_image_1 /= point_on_basler_image_1[2, 0]
#
#     point_on_basler_image_2 = basler_extrinsic_matrix.dot([[100], [0], [0], [1]])
#     point_on_basler_image_2 = basler_intrinsic_matrix.dot(point_on_basler_image_2)
#     point_on_basler_image_2 /= point_on_basler_image_2[2, 0]
#
#     point_on_basler_image_3 = basler_extrinsic_matrix.dot([[0], [100], [0], [1]])
#     point_on_basler_image_3 = basler_intrinsic_matrix.dot(point_on_basler_image_3)
#     point_on_basler_image_3 /= point_on_basler_image_3[2, 0]
#
#     point_on_basler_image_4 = basler_extrinsic_matrix.dot([[100], [100], [0], [1]])
#     point_on_basler_image_4 = basler_intrinsic_matrix.dot(point_on_basler_image_4)
#     point_on_basler_image_4 /= point_on_basler_image_4[2, 0]
#
#     cv2.circle(basler_image, (int(point_on_basler_image_1[0, 0]), int(point_on_basler_image_1[1, 0])), 3, (0,0,255), -1)
#     cv2.circle(basler_image, (int(point_on_basler_image_2[0, 0]), int(point_on_basler_image_2[1, 0])), 3, (0,0,255), -1)
#     cv2.circle(basler_image, (int(point_on_basler_image_3[0, 0]), int(point_on_basler_image_3[1, 0])), 3, (0,0,255), -1)
#     cv2.circle(basler_image, (int(point_on_basler_image_4[0, 0]), int(point_on_basler_image_4[1, 0])), 3, (0,0,255), -1)
#     cv2.imshow('basler_point', basler_image)
#
#     point_in_basler_coordinates_1 = basler_extrinsic_matrix.dot([[0], [0], [0], [1]])
#     point_in_ir_coordinates_1 = np.linalg.inv(extrinsic_matrix_to_basler).dot(point_in_basler_coordinates_1)
#     point_on_ir_image_1 = ir_intrinsic_matrix.dot(point_in_ir_coordinates_1)
#     point_on_ir_image_1 /= point_on_ir_image_1[2, 0]
#
#     point_in_basler_coordinates_2 = basler_extrinsic_matrix.dot([[100], [0], [0], [1]])
#     point_in_ir_coordinates_2 = np.linalg.inv(extrinsic_matrix_to_basler).dot(point_in_basler_coordinates_2)
#     point_on_ir_image_2 = ir_intrinsic_matrix.dot(point_in_ir_coordinates_2)
#     point_on_ir_image_2 /= point_on_ir_image_2[2, 0]
#
#     point_in_basler_coordinates_3 = basler_extrinsic_matrix.dot([[0], [100], [0], [1]])
#     point_in_ir_coordinates_3 = np.linalg.inv(extrinsic_matrix_to_basler).dot(point_in_basler_coordinates_3)
#     point_on_ir_image_3 = ir_intrinsic_matrix.dot(point_in_ir_coordinates_3)
#     point_on_ir_image_3 /= point_on_ir_image_3[2, 0]
#
#     point_in_basler_coordinates_4 = basler_extrinsic_matrix.dot([[100], [100], [0], [1]])
#     point_in_ir_coordinates_4 = np.linalg.inv(extrinsic_matrix_to_basler).dot(point_in_basler_coordinates_4)
#     point_on_ir_image_4 = ir_intrinsic_matrix.dot(point_in_ir_coordinates_4)
#     point_on_ir_image_4 /= point_on_ir_image_4[2, 0]
#
#
#     cv2.circle(ir_image, (int(point_on_ir_image_1[0, 0]), int(point_on_ir_image_1[1, 0])), 1, (0,0,255), -1)
#     cv2.circle(ir_image, (int(point_on_ir_image_2[0, 0]), int(point_on_ir_image_2[1, 0])), 1, (0,0,255), -1)
#     cv2.circle(ir_image, (int(point_on_ir_image_3[0, 0]), int(point_on_ir_image_3[1, 0])), 1, (0,0,255), -1)
#     cv2.circle(ir_image, (int(point_on_ir_image_4[0, 0]), int(point_on_ir_image_4[1, 0])), 1, (0,0,255), -1)
#     cv2.imshow('irpoint', ir_image)
#     cv2.waitKey()

    charuco_basler = cv2.imread('images/charuco_basler.png')
    charuco_basler = cv2.flip(charuco_basler, 1)
    basler_ir.estimate_board_pose(charuco_basler)
    charuco_basler_axis = basler_ir._draw_axis(charuco_basler, show = False)
    basler_ir.check_board(charuco_basler_axis, cv2.Rodrigues(basler_ir.rvecs[-1])[0], basler_ir.tvecs[-1].T, square_width = 0.06, row_number = 6, column_number=4, show=True, line_width=2)
    cv2.waitKey()

    charuco_color = cv2.imread('images/charuco_color.png')
    charuco_color = cv2.flip(charuco_color, 1)
    kinect_color.estimate_board_pose(charuco_color)
    charuco_color_axis = kinect_color._draw_axis(charuco_color, show = False)
    kinect_color.check_board(charuco_color_axis, cv2.Rodrigues(kinect_color.rvecs[-1])[0], kinect_color.tvecs[-1].T, square_width = 0.06, row_number = 6, column_number=4, show=True, line_width=2)
    cv2.waitKey()
