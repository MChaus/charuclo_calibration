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

    all_rotation_matrices = []
    all_tanslation_vecotrs = []
    path_to_ir_basler=r'D:\project\charuco_calibration\images\basler'
    path_to_color_kinect=r'D:\project\charuco_calibration\images\color_kinect'
    i = -1
    for filename in os.listdir(path_to_ir_basler):
        basler_ir_image_path = os.path.join(path_to_ir_basler, filename)
        kinect_color_image_path = os.path.join(path_to_color_kinect, filename)
        i += 1
        if (
            os.path.isfile(basler_ir_image_path) and
            os.path.isfile(kinect_color_image_path) and
            i
            ):
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
        if os.path.isfile(basler_ir_image_path) :
            basler_ir_image = cv2.imread(basler_ir_image_path)
            basler_ir_image = cv2.flip(basler_ir_image, 1)
            cv2.imshow('Axis',
                       basler_ir.check_calibration_charuco(basler_ir_image,
                                                           show=False,
                                                           square_width = 60,
                                                           column_number=4,
                                                           row_number=6))
            cv2.waitKey()


    path_to_ir_image=r'D:\project\charuco_calibration\images\basler\00000.png'
    path_to_color_image=r'D:\project\charuco_calibration\images\color_kinect\00000.png'

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
                                    show=True, line_width=5, scale=1)

    
    # for r_matr, t_vec in zip(all_rotation_matrices, all_tanslation_vecotrs):
    #     basler_ir.check_transition(basler_ir_image, kinect_color_image,
    #                                 kinect_color, r_matr, t_vec,
    #                                 column_number=4, row_number=6, square_width = 60,
    #                                 show=True, line_width=5, scale=2)

    # # CALCULATE
    # charuco_color = cv2.imread('images/color_kinect/08_05_18_001.png')
    # charuco_color = cv2.flip(charuco_color, 1)
    # charuco_basler_1 = cv2.imread('images/basler/08_05_18_001.png')
    # charuco_basler_1 = cv2.flip(charuco_basler_1, 1)
    # rmatr_basler_color, tvec_basler_color = basler_ir.get_disposition_charuco(charuco_basler_1, charuco_color, kinect_color)
    #
    # # TEST
    # charuco_color_test = cv2.imread('images/color_kinect/08_05_18_002.png')
    # charuco_color_test = cv2.flip(charuco_color_test, 1)
    # charuco_basler_1_test = cv2.imread('images/basler/08_05_18_002.png')
    # charuco_basler_1_test = cv2.flip(charuco_basler_1_test, 1)
    # basler_ir.check_transition(charuco_basler_1_test, charuco_color_test,
    #                            kinect_color, rmatr_basler_color, tvec_basler_color,
    #                            column_number=4, row_number=6, square_width = 60,
    #                            show=True, line_width=5, scale=2)
    #
    # # CALCULATE
    # charuco_web_cam = cv2.imread('images/web_cam/web_cam_1.jpg')
    # charuco_basler_2 = cv2.imread('images/basler/web_cam_1.png')
    # charuco_basler_2 = cv2.flip(charuco_basler_2, 1)
    # rmatr_web_cam_basler, tvec_web_cam_basler = web_cam.get_disposition_charuco(charuco_web_cam, charuco_basler_2, basler_ir)
    #
    # # TEST
    # charuco_web_cam_test = cv2.imread('images/web_cam/web_cam_2.jpg')
    # charuco_basler_2_test = cv2.imread('images/basler/web_cam_2.png')
    # charuco_basler_2_test = cv2.flip(charuco_basler_2_test, 1)
    # web_cam.check_transition(charuco_web_cam_test, charuco_basler_2_test,
    #                            basler_ir, rmatr_web_cam_basler, tvec_web_cam_basler,
    #                            column_number=4, row_number=6, square_width = 60,
    #                            show=True, line_width=5, scale=2)
    #
    # # CALCULATE
    # charuco_web_cam_3 = cv2.imread('images/web_cam/web_cam_3.jpg')
    # rmatr_cam_screen_web, tvec_cam_screen_web = web_cam.estimate_board_pose(charuco_web_cam_3)
    #
    # # TEST
    # web_cam.check_calibration_charuco(charuco_web_cam_3, show=True, line_width=3, square_width = 60, row_number = 6, column_number=4,)
    #
    # # CALCULATE
    # extrinsic_ir_color = np.identity(4)
    # extrinsic_ir_color[0:3, 0:3] = rmatr_ir_color
    # extrinsic_ir_color[0:3, 3] = tvec_ir_color[0:3, 0]
    #
    # # TEST
    # rmatr_ir = np.array([   [ 0.9755,    0.0583,   -0.2119],
    #                         [-0.1574,    0.8582,   -0.4885],
    #                         [ 0.1534,    0.5099,    0.8464]]).T
    # tvec_ir =  np.array([[-446.8],  [2.3],    [1192.4]])
    #
    # rmatr_color = np.array([    [ 0.9743,    0.0538,   -0.2187],
    #                             [-0.1576,    0.8567,   -0.4911],
    #                             [ 0.1610,    0.5130,    0.8432]]).T
    #
    # tvec_color =  np.array([[-490.4],    [9.9],    [1194.0]])
    #
    # charuco_ir_test = cv2.imread('images/ir_kinect/00026.png')
    # charuco_ir_test = cv2.flip(charuco_ir_test, 1)
    # charuco_color_test = cv2.imread('images/color_kinect/00026.png')
    # charuco_color_test = cv2.flip(charuco_color_test, 1)
    # kinect_ir.check_transition_chess(charuco_ir_test, charuco_color_test,
    #                                 kinect_color, rmatr_ir, tvec_ir,
    #                                 rmatr_color, tvec_color,
    #                                 rmatr_ir_color, tvec_ir_color,
    #                                 column_number=6, row_number=5, square_width = 100,
    #                                 show=True, line_width=5, scale=1)
    #
    # # CALCULATE
    # extrinsic_basler_color = np.identity(4)
    # extrinsic_basler_color[0:3, 0:3] = rmatr_basler_color
    # extrinsic_basler_color[0:3, 3] = tvec_basler_color[0:3, 0]
    #
    # extrinsic_web_cam_basler = np.identity(4)
    # extrinsic_web_cam_basler[0:3, 0:3] = rmatr_web_cam_basler
    # extrinsic_web_cam_basler[0:3, 3] = tvec_web_cam_basler[0:3, 0]
    #
    # extrinsic_web_cam_screen = np.identity(4)
    # extrinsic_web_cam_screen[0:3, 0:3] = rmatr_cam_screen_web
    # extrinsic_web_cam_screen[0:3, 3] = tvec_cam_screen_web[0:3, 0]
    #
    # # CALCULATE
    # extrinsic_color_ir = LA.inv(extrinsic_ir_color)
    #
    # # TEST
    # kinect_color.check_transition_chess(charuco_color_test, charuco_ir_test,
    #                                 kinect_ir, rmatr_color, tvec_color,
    #                                 rmatr_ir, tvec_ir,
    #                                 extrinsic_color_ir[0:3, 0:3], extrinsic_color_ir[0:3, 3].reshape(3, 1),
    #                                 column_number=6, row_number=5, square_width = 100,
    #                                 show=True, line_width=5, scale=1)
    #
    # # CALCULATE
    # extrinsic_basler_ir = extrinsic_color_ir @ extrinsic_basler_color
    #
    # # TEST
    # charuco_ir_test = cv2.imread('images/ir_kinect/00017.png')
    # charuco_ir_test = cv2.flip(charuco_ir_test, 1)
    # charuco_basler_test = cv2.imread('images/basler/00017.png')
    # charuco_basler_test = cv2.flip(charuco_basler_test, 1)
    # basler_ir.check_transition(charuco_basler_test, charuco_ir_test,
    #                            kinect_ir, extrinsic_basler_ir[0:3, 0:3], extrinsic_basler_ir[0:3, 3].reshape(3, 1),
    #                            column_number=4, row_number=6, square_width = 60,
    #                            show=True, line_width=3, scale=1)
    #
    # # CALCULATE
    # extrinsic_web_cam_ir = extrinsic_basler_ir @ extrinsic_web_cam_basler
    #
    # # TEST
    # charuco_web_cam_test = cv2.imread('images/web_cam/web_cam_2.jpg')
    # charuco_ir_test = cv2.imread('images/ir_kinect/00037.png')
    # charuco_ir_test = cv2.flip(charuco_ir_test, 1)
    # web_cam.check_transition(charuco_web_cam_test, charuco_ir_test,
    #                            kinect_ir, extrinsic_web_cam_ir[0:3, 0:3], extrinsic_web_cam_ir[0:3, 3].reshape(3, 1),
    #                            column_number=4, row_number=6, square_width = 60,
    #                            show=True, line_width=3, scale=1)
    #
    # # CALCULATE
    # extrinsic_screen_web_cam = LA.inv(extrinsic_web_cam_screen)
    # extrinsic_screen_ir = extrinsic_web_cam_ir @ extrinsic_web_cam_screen
    #
    # # TEST
    # charuco_ir_test = cv2.imread('images/ir_kinect/00000.png')
    # charuco_ir_test = cv2.flip(charuco_ir_test, 1)
    # undistored_image = cv2.undistort(charuco_ir_test, kinect_ir.camera_matrix, kinect_ir.dist_coeff)
    # intrinsic_matrix_camera_1 = np.zeros((3, 4))
    # intrinsic_matrix_camera_1[0:3, 0:3] = kinect_ir.camera_matrix
    # for x in range(0, 4):
    #     for y in range(0, 6):
    #         # Draw point from space on image_1.
    #         point_in_ir = extrinsic_screen_ir.dot([[x * 100], [y * 100], [0], [1]])
    #         point_on_image = intrinsic_matrix_camera_1.dot(point_in_ir)
    #         point_on_image /= point_on_image[2, 0]
    #         cv2.circle(undistored_image, (int(point_on_image[0, 0]), int(point_on_image[1, 0])), 3, (0,0,255), -1)
    # cv2.imshow('ir', undistored_image)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    #
    # print('\nextrinsic_color_ir: \n', extrinsic_color_ir,
    #       '\nextrinsic_basler_ir: \n', extrinsic_basler_ir,
    #       '\nextrinsic_web_cam_ir: \n', extrinsic_web_cam_ir,
    #       '\nextrinsic_screen_ir: \n', extrinsic_screen_ir)

#SOME TRASH
    # charuco_web_cam_1 = cv2.imread('images/web_cam/web_cam_1.jpg')
    # cv2.imshow('Axis_web', web_cam._draw_axis(charuco_web_cam_1))
    # web_cam.check_calibration_charuco(charuco_web_cam_1, show=True, line_width=3, square_width = 0.06, row_number = 6, column_number=4,)

    #
    # charuco_basler_1 = cv2.imread('images/basler/web_cam_1.png')
    # charuco_basler_1 = cv2.flip(charuco_basler_1, 1)
    # basler_ir.check_calibration_charuco(charuco_basler_1, show=True, line_width=1, square_width = 0.06, row_number = 6, column_number=4,)

    # charuco_color_1 = cv2.imread('images/color_kinect/08_05_18_001.png')
    # charuco_color_1 = cv2.flip(charuco_color_1, 1)
    # kinect_color.check_calibration_charuco(charuco_color_1, square_width = 0.06, row_number = 6, column_number=4, show=True, line_width=1)
    # rotation_matrix_to_basler, translatioin_vector_to_basler = web_cam.get_disposition_charuco(charuco_web_cam_1, charuco_basler_1, basler_ir)
    #
    # charuco_web_cam_2 = cv2.imread('images/web_cam/web_cam_2.jpg')
    # web_cam.check_calibration_charuco(charuco_web_cam_2, show=True, line_width=1, square_width = 0.06, row_number = 6, column_number=4,)
    #
    # charuco_basler_2 = cv2.imread('images/basler/web_cam_2.png')
    # charuco_basler_2 = cv2.flip(charuco_basler_2, 1)
    # basler_ir.check_calibration_charuco(charuco_basler_2, show=True, line_width=1, square_width = 0.06, row_number = 6, column_number=4,)
    #
    # web_cam.check_transition(charuco_web_cam_2, charuco_basler_2, basler_ir,
    #                             rotation_matrix_to_basler, translatioin_vector_to_basler,
    #                             column_number=4, row_number=6, square_width = 0.06,
    #                             show=True, line_width=5)
    #
    # charuco_web_cam_3 = cv2.imread('images/web_cam/web_cam_3.jpg')
    # web_cam.check_calibration_charuco(charuco_web_cam_3, show=True, line_width=3, square_width = 0.06, row_number = 6, column_number=4,)

    # rotation_matrix_to_color, translatioin_vector_to_color = basler_ir.get_disposition_charuco(charuco_basler_1, charuco_color_1, kinect_color)
    #
    # charuco_color_2 = cv2.imread('images/color_kinect/08_05_18_002.png')
    # charuco_color_2 = cv2.flip(charuco_color_2, 1)
    # charuco_basler_2 = cv2.imread('images/basler/08_05_18_002.png')
    # charuco_basler_2 = cv2.flip(charuco_basler_2, 1)
    # basler_ir.check_transition(charuco_basler_2, charuco_color_2, kinect_color,
    #                            rotation_matrix_to_color, translatioin_vector_to_color,
    #                            column_number=4, row_number=6, square_width = 0.06,
    #                            show=True, line_width=5)
    #
    # rotation_matrix_to_basler, translatioin_vector_to_basler = kinect_color.get_disposition_charuco(charuco_color_1, charuco_basler_1, basler_ir)
    #
    # kinect_color.check_transition(charuco_color_2, charuco_basler_2, basler_ir,
    #                               rotation_matrix_to_basler, translatioin_vector_to_basler,
    #                               column_number=4, row_number=6, square_width = 0.06,
    #                               show=True, line_width=5)
