#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Provides methods to calibrate camera, using charuco board.
@author: Mykhailo Chaus
'''

import cv2
import numpy as np
import yaml
import os

class CalibratedCamera:
    def __init__(self, squares_x=None, squares_y=None, square_length=None,
                 marker_length=None, camera_matrix=None, dist_coeff=None,
                 rvecs=None, tvecs=None, image_size=None, flipped=False,
                 dictionary=cv2.aruco.DICT_6X6_250):
        self.squares_x = squares_x
        self.squares_y = squares_y
        self.square_length = square_length
        self.marker_length = marker_length
        self.camera_matrix = camera_matrix
        self.dist_coeff = dist_coeff
        if rvecs is None:
            rvecs = []
        self.rvecs = rvecs
        if tvecs is None:
            tvecs = []
        self.tvecs = tvecs
        self.image_size = image_size
        self.dictionary = cv2.aruco.getPredefinedDictionary(dictionary)
        self.board = self.create_charuco_board()
        self.all_corners = []
        self.all_ids = []
        self.num_frames = 0
        self.flipped = flipped


    def __str__(self):
        message = (
                'squares_x :\n{}\n'
                'squares_x type:\n{}\n'
                'squares_y :\n{}\n'
                'squares_y type:\n{}\n'
                'square_length :\n{}\n'
                'square_length type:\n{}\n'
                'marker_length :\n{}\n'
                'marker_length type:\n{}\n'
                'camera_matrix :\n{}\n'
                'camera_matrix type:\n{}\n'
                'dist_coeff :\n{}\n'
                'dist_coeff type:\n{}\n'
                'rvecs :\n{}\n'
                'rvecs type:\n{}\n'
                'tvecs :\n{}\n'
                'tvecs type:\n{}\n'
                'image_size :\n{}\n'
                'image_size type:\n{}\n'
                ).format(
                    self.squares_x,
                    type(self.squares_x),
                    self.squares_y,
                    type(self.squares_y),
                    self.square_length,
                    type(self.square_length),
                    self.marker_length,
                    type(self.marker_length),
                    self.camera_matrix,
                    type(self.camera_matrix),
                    self.dist_coeff,
                    type(self.dist_coeff),
                    self.rvecs,
                    type(self.rvecs),
                    self.tvecs,
                    type(self.tvecs),
                    self.image_size,
                    type(self.image_size)
                        )
        return message

    def get_charuco_board(self):
        '''Creates charuco board with established parameters.

        Returns instance of chruco board with parameters that were obtained from
        constructor.
        '''
        if (
                self.squares_x is not None and
                self.squares_y is not None and
                self.square_length is not None and
                self.marker_length is not None
            ):
            return cv2.aruco.CharucoBoard_create(self.squares_x, self.squares_y,
                                                 self.square_length,
                                                 self.marker_length,
                                                 self.dictionary)
        else:
            print('Board wasn\'t created. Assign ChArUco board parameters')
            return None

    def draw_charuco_board(self, path=None, size=(1080, 1920), margin_size=100,
                           show=False):
        '''Draws, saves and shows ChArUco board image.

        If show == True this function shows ChArUco board in the window.
        If path is given, ChArUco board will be saved.
        Parameter size stands to determine image resolution.
        '''
        board_image = self.board.draw(size, marginSize = margin_size)
        if path is not None:
            cv2.imwrite(path, board_image)
        if show:
            cv2.imshow('ChArUco board', board_image)
        return board_image

    def _calibrate_charuco(self, log_out=False):
        '''Finds camera parameters.

        This method finds camera matrix, distortion coefficients, list of
        rotation vectors and translation vectors.
        '''
        retval, self.camera_matrix, self.dist_coeff, self.rvecs, self.tvecs =\
        cv2.aruco.calibrateCameraCharuco(self.all_corners, self.all_ids,
                                         self.board, self.image_size,
                                         self.camera_matrix, self.dist_coeff)
        if log_out:
            print('Coefficients were calculated')
        return self

    def draw_markers(self, image, show=True):
        '''Returns image with good markers.

        If show == True, image with detected markers will be shown.
        '''
        marked_image = np.copy(image)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        markers_coordinates, markers_id, bad_markers =\
        cv2.aruco.detectMarkers(image, self.dictionary)
        num_corners, charuco_corners, charuco_ids =\
        cv2.aruco.interpolateCornersCharuco(markers_coordinates, markers_id,
                                            gray_image, self.board)
        cv2.aruco.drawDetectedMarkers(marked_image,
                                      markers_coordinates,
                                      markers_id)
        cv2.aruco.drawDetectedCornersCharuco(marked_image,
                                             charuco_corners,
                                             cornerColor = (0, 255,255))
        if show:
            cv2.imshow('Detected markers', marked_image)
            cv2.waitKey()
        return marked_image

    def get_markers(self, image, show=False, image_name='Marked frame',
                    log_out=True):
        '''Get markers' coordinates from frame and add them to all_corners.

        If show == True, the output image with markers will be shown.
        Returns markers' coordinates and their ids.
        '''
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        markers_coordinates, markers_id, bad_markers =\
        cv2.aruco.detectMarkers(gray_image, self.dictionary)
        if len(markers_coordinates) > 0:
            num_corners, charuco_corners, charuco_ids =\
            cv2.aruco.interpolateCornersCharuco(markers_coordinates,
                                                 markers_id, gray_image,
                                                 self.board)
            if (
                    charuco_corners is not None and
                    charuco_ids is not None and
                    len(charuco_corners) > 3
                ):
                if log_out:
                    print('Corners were detected')
                self.all_corners.append(charuco_corners)
                self.all_ids.append(charuco_ids)
                self.num_frames += 1
                if show:
                    self.draw_markers(image)
        else:
            if log_out:
                print('Corners weren\'t detected')
        return charuco_corners, charuco_ids

    def charuco_calibrate_from_dir(self, path_to_data=None, show=False):
        '''Get camera parameters from images, that were stored in folder.

        Get camera parameters from already taken images, that locate in the
        folder path_to_data. If you want to see markers, pass show=True.
        Use this function for remote calibration
        '''
        for filename in os.listdir(path_to_data):
            image_path = os.path.join(path_to_data, filename)
            image = cv2.imread(image_path)
            if image:
                self.image_size = image.shape[0:2]
                self.get_markers(image, show, image_path, log_out=True)
        self._calibrate_charuco()

    
