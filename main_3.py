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

    def get_extrinsics(camera_1, camera_2, path_1, path_2, flip_1=False, flip_2=False):
        all_rotation_vectors = []
        all_tanslation_vecotrs = []
        for filename in os.listdir(path_1):
            image_path_1 = os.path.join(path_1, filename)
            image_path_2 = os.path.join(path_2, filename)
            if (
                os.path.isfile(image_path_1) and
                os.path.isfile(image_path_2)
                ):
                image_1 = cv2.imread(image_path_1)
                image_2 = cv2.imread(image_path_2)
                if flip_1:
                    image_1 = cv2.flip(image_1, 1)
                if flip_2:
                    image_2 = cv2.flip(image_2, 1)
                print('\t', filename)
                rmatr_web_cam_1_2, tvec_web_cam_1_2 = camera_1.get_disposition_charuco(image_1, image_2, camera_2)
                all_rotation_vectors.append(cv2.Rodrigues(rmatr_web_cam_1_2)[0])
                all_tanslation_vecotrs.append(tvec_web_cam_1_2)
        rvec = np.array(all_rotation_vectors).mean(axis=0)
        tvec = np.array(all_tanslation_vecotrs).mean(axis=0)
        extrinsic_matrix_1_2 = np.identity(4)
        extrinsic_matrix_1_2[0:3, 0:3] = cv2.Rodrigues(rvec)[0]
        extrinsic_matrix_1_2[0:3, 3] = tvec[0:3, 0]
        return extrinsic_matrix_1_2

    def wall_disposotion(camera, path, flip):
        all_rotation_vectors = []
        all_tanslation_vecotrs = []
        for filename in os.listdir(path):
            image_path = os.path.join(path, filename)
            if os.path.isfile(image_path):
                image = cv2.imread(image_path)
                if flip:
                    image = cv2.flip(image, 1)
                rmatr, tvec = camera.estimate_board_pose(image)
                all_rotation_vectors.append(cv2.Rodrigues(rmatr)[0])
                all_tanslation_vecotrs.append(tvec)
        rvec = np.array(all_rotation_vectors).mean(axis=0)
        tvec = np.array(all_tanslation_vecotrs).mean(axis=0)
        extrinsic_matrix_1_2 = np.identity(4)
        extrinsic_matrix_1_2[0:3, 0:3] = cv2.Rodrigues(rvec)[0]
        extrinsic_matrix_1_2[0:3, 3] = tvec[0:3, 0]
        return extrinsic_matrix_1_2


    web_cam_wall = CalibratedCamera(
            squares_x=4,
            squares_y=6,
            square_length=220,
            marker_length=110,
            camera_matrix=np.array([
            [2265.4 * 0.625,   0,   1009.4 * 0.625],
            [0,     2268.7 * 0.625, 811.5 * 0.625],
            [0,     0,      1]
            ]) ,
            dist_coeff=np.array([[ -0.0276, 0.1141, 0.0000871, -0.0002941]]),
            image_size=(960, 1280)
            )

    web_cam_screen = CalibratedCamera(
            squares_x=4,
            squares_y=6,
            square_length=70,
            marker_length=35,
            camera_matrix=np.array([
            [2265.4 * 0.625,   0,   1009.4 * 0.625],
            [0,     2268.7 * 0.625, 811.5 * 0.625],
            [0,     0,      1]
            ]) ,
            dist_coeff=np.array([[ -0.0276, 0.1141, 0.0000871, -0.0002941]]),
            image_size=(960, 1280)
            )

    web_cam_screen_high = CalibratedCamera(
            squares_x=4,
            squares_y=6,
            square_length=70,
            marker_length=35,
            camera_matrix=np.array([
            [2265.4,   0,   1009.4],
            [0,     2268.7, 811.5],
            [0,     0,      1]
            ]) ,
            dist_coeff=np.array([[ -0.0276, 0.1141, 0.0000871, -0.0002941]]),
            image_size=(1536, 2048)
            )

    # Basler from MatLab.
    basler_ir_board = CalibratedCamera(
            squares_x=5,
            squares_y=8,
            square_length=70,
            marker_length=35,
            camera_matrix=np.array([
            [3793.8,         0,         0],
            [0,    3795.0,         0],
            [656.5,    464.3,    1.0]
            ]).T,
            dist_coeff=np.array([[ -0.3303, -0.0512, -0.0003425,  0.000983]]),
            image_size=(972, 1296)
            )

    # Color Kinect from MatLab.
    kinect_color_board = CalibratedCamera(
            squares_x=5,
            squares_y=8,
            square_length=70,
            marker_length=35,
            camera_matrix=np.array([
            [1050.9, 0,  956.0],
            [0,     1050.0, 534.8],
            [0,     0,      1]
            ]),
            dist_coeff=np.array([[0.038, -0.0388, 0.0010, 0.0004]]),
            image_size=(1080, 1920)
            )

    kinect_color_screen = CalibratedCamera(
            squares_x=4,
            squares_y=6,
            square_length=70,
            marker_length=35,
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

    extrinsic_matrix_color_ir = np.array(
        [[0.9999414245144126,-0.0037285226427599223,-0.006683574896882698,51.56535410363723],
        [0.0036709070831910983,0.9999678224799214,-0.004324456714121043,-0.690797434823698],
        [0.006715392444704288,0.004274880534957171,0.9999366248843202,1.6719174435246276],
        [0.0,0.0,0.0,1.0]])

    path_to_web_cam_color = r'D:\project-files-BRS\charuco_calibration_prev\images\web_cam\web_cam_color'
    path_to_web_cam_wall = r'D:\project-files-BRS\charuco_calibration_prev\images\web_cam\web_cam_wall'
    path_to_basler = r'D:\project-files-BRS\charuco_calibration_prev\images\basler'
    path_to_color_basler = r'D:\project-files-BRS\charuco_calibration_prev\images\color_kinect\color_basler'
    path_to_color_web_cam = r'D:\project-files-BRS\charuco_calibration_prev\images\color_kinect\color_web_cam'
    path_to_web_cam_screen = r'D:\project-files-BRS\charuco_calibration_prev\images\web_cam\web_cam_screen'

    extrinsic_matrix_web_cam_color = get_extrinsics(web_cam_screen, kinect_color_screen, path_to_web_cam_color, path_to_color_web_cam, flip_1=True, flip_2=True)
    extrinsic_matrix_basler_color = get_extrinsics(basler_ir_board, kinect_color_board, path_to_basler, path_to_color_basler, flip_1=True, flip_2=True)
    extrinsic_matrix_wall_web_cam = wall_disposotion(web_cam_wall, path_to_web_cam_wall, flip=True)
    extrinsic_matrix_screen_web_cam = wall_disposotion(web_cam_screen, path_to_web_cam_screen, flip=True)


    extrinsic_matrix_basler_ir = extrinsic_matrix_color_ir @ extrinsic_matrix_basler_color
    extrinsic_matrix_web_cam_ir = extrinsic_matrix_color_ir @ extrinsic_matrix_web_cam_color
    extrinsic_matrix_wall_ir = extrinsic_matrix_web_cam_ir @ extrinsic_matrix_wall_web_cam
    extrinsic_matrix_screen_ir = extrinsic_matrix_web_cam_ir @ extrinsic_matrix_screen_web_cam

    print(extrinsic_matrix_wall_ir)

    extrinsic_matrices = {
        'extrinsic_matrix_color_ir' : extrinsic_matrix_color_ir.tolist(),
        'extrinsic_matrix_web_cam_ir' : extrinsic_matrix_web_cam_ir.tolist(),
        'extrinsic_matrix_wall_ir' : extrinsic_matrix_wall_ir.tolist(),
        'extrinsic_matrix_screen_ir' : extrinsic_matrix_screen_ir.tolist(),
        'extrinsic_matrix_basler_ir' :  extrinsic_matrix_basler_ir.tolist()
    }

    print(extrinsic_matrices)
    with open('extrinsic_matrices.json', 'w') as _file:
        json.dump(extrinsic_matrices, _file, indent=4, ensure_ascii=False, )
        _file.close()
