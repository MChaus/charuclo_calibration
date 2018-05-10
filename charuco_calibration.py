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
                 rvecs=None, tvecs=None, image_size=None, flipped=False):
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
        self.dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        self.board = self._create_charuco_board()
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


    def _create_charuco_board(self):
        '''
        Create charuco board with  established parameters
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
            print('Board wasn\'t created')
            return None


    def draw_charuco_board(self, path=None, size=(1080, 1920), margin_size=100,
                           show=False):
        '''
        Save charuco board and show its instance
        '''
        image = self.board.draw(size, marginSize = margin_size)
        if path is not None:
            cv2.imwrite(path, image)
        if show:
            cv2.imshow('charuco board', image)


    def calibrate_from_images(self, path_to_data=None, show=False):
        '''
        Get camera parameters from already taken images, that locate in the
        folder path_to_data. If you want to see markers, pass show=True.
        Use this function for remote calibration
        '''
        for filename in os.listdir(path_to_data):
            image_path = os.path.join(path_to_data, filename)
            if (
                    os.path.isfile(image_path) and
                    (
                        image_path.endswith('.png') or
                        image_path.endswith('.jpg')
                        )
                ):
                image = cv2.imread(image_path)
                self.image_size = image.shape[0:2]
                self.get_markers(image, show, image_path, log_out=True)
        self._calibrate()


    def axis_on_video(self, path_to_video=cv2.CAP_ANY, write_path = None):
        '''
        Draw axis on the video. Pass write_path for saving.
        '''
        capture = cv2.VideoCapture(path_to_video)
        if capture.isOpened():
            retval, frame = capture.read()
            self.image_size = frame.shape[0:2]
        if write_path is not None:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            video_out = cv2.VideoWriter(write_path, fourcc, 30.0, (640, 480))
        while capture.isOpened():
            retval, frame = capture.read()
            self.get_markers(frame, show=False)
            if self.all_corners != [] and self.all_ids != []:
                self.estimate_board_pose(frame)
                axis_frame = self._draw_axis(frame)
            else:
                axis_frame = frame
            self.all_corners = []
            self.all_ids = []
            cv2.imshow('camera', frame)
            if write_path is not None:
                video_out.write(axis_frame)
            # if ESC was pressed
            # close all windows
            if cv2.waitKey(33) % 256 == 27:
                print('ESC pressed, closing...')
                break
        capture.release()
        if write_path is not None:
            video_out.release()
        capture.release()
        cv2.destroyAllWindows()


    def live_calibration(self, path_to_video=cv2.CAP_ANY, write_path=None,):
        '''
        If you want calibrate camera now, use this method. Take photos by
        pressing SPACE. Then press ENTER for calibration. Set write_path for
        saving images into chosen directory. To stop process, press ESC.
        '''
        capture = cv2.VideoCapture(path_to_video)
        if capture.isOpened():
            retval, frame = capture.read()
            self.image_size = frame.shape[0:2]
            frame_num = 0
        while capture.isOpened():
            retval, frame = capture.read()
            # if SPACE was pressed
            # take frame and get markers from it
            if cv2.waitKey(1) % 256 == 32:
                frame_num += 1
                self.get_markers(frame, show=True)
                self._write_image(frame, frame_num, write_path)
            # if ENTER was pressed
            # calibrate camera using Charuco_calibration
            elif cv2.waitKey(33) % 256 == 13:
                self._calibrate()
                self._draw_axis(frame)
            # if ESC was pressed
            # close all windows
            elif cv2.waitKey(33) % 256 == 27:
                print('ESC pressed, closing...')
                break
            cv2.imshow('camera', frame)
        capture.release()
        cv2.destroyAllWindows()


    def _write_image(self, frame, frame_num, write_path=None):
        '''
        Save frame to write_path with name image_{frame_num}.png
        '''
        if write_path is not None:
            cv2.imwrite(os.path.join(write_path, 'image_{}.png'.format(frame_num)), frame)


    def _draw_axis(self, frame, show=True):
        '''
        Draw axis on frame that has charuco board.
        '''
        axis_frame = np.copy(frame)
        cv2.aruco.drawAxis(
            axis_frame,
            self.camera_matrix,
            self.dist_coeff,
            self.rvecs[-1],
            self.tvecs[-1],
            length=self.square_length
            )
        if show:
            cv2.imshow('Axis', axis_frame)
        return axis_frame


    def load_data(self, path_to_data=None):
        '''
        Load main parameters to object from yaml file that has name
        path_to_data.
        '''
        if path_to_data is None:
            print('Undefind path')
        else:
            with open(path_to_data) as _file:
                metadata = yaml.load(_file)
            self.squares_x = metadata.get('squares_x')
            self.squares_y = metadata.get('squares_y')
            self.square_length = metadata.get('square_length')
            self.marker_length = metadata.get('marker_length')
            self.image_size = np.array(metadata.get('size'))
            self.camera_matrix = np.array(metadata.get('camera_matrix'))
            self.dist_coeff = np.array(metadata.get('distortion_coefficient'))
            self.rvecs = [np.array(vector)
                          for vector in metadata.get('rotation_vector')]
            self.tvecs = [np.array(vector)
                          for vector in metadata.get('translation_vector')]


    def dump_data(self, path_to_data=None):
        '''
        Dump main parameters from object to yaml file that has name
        path_to_data.
        '''
        if path_to_data is None:
            print('Undefind path')
        else:
            metadata = {
                'squares_x' : self.squares_x,
                'squares_y' : self.squares_y,
                'square_length' : self.square_length,
                'marker_length' : self.marker_length,
                'size' : np.asarray(self.image_size).tolist(),
                'camera_matrix' : np.asarray(self.camera_matrix).tolist(),
                'distortion_coefficient' : np.asarray(self.dist_coeff).tolist(),
                'rotation_vector' : np.asarray(self.rvecs).tolist(),
                'translation_vector' : np.asarray(self.tvecs).tolist(),
                }
            with open(path_to_data, 'w') as _file:
                yaml.dump(metadata, _file)


    def _calibrate(self, log_out=False):
        '''
        Finds camera parameters from corners and ids, that was already prepared.
        '''
        retval, self.camera_matrix, self.dist_coeff, self.rvecs, self.tvecs =\
        cv2.aruco.calibrateCameraCharuco(self.all_corners, self.all_ids,
                                         self.board, self.image_size,
                                         self.camera_matrix, self.dist_coeff)
        if log_out:
            print('Coefficients were calculated')
        return self


    def estimate_board_pose(self, image):
        '''
        Pose estimation for a ChArUco board.
        This function returns rotation matrix and translation vector of a
        ChArUco board.
        '''
        self.get_markers(image)
        retval, rotation_vector, translation_vector = cv2.aruco.estimatePoseCharucoBoard(
            self.all_corners[-1],
            self.all_ids[-1],
            self.board,
            self.camera_matrix,
            self.dist_coeff
        )
        if retval:
            self.rvecs.append(rotation_vector)
            self.tvecs.append(translation_vector)
            rotation_matrix = cv2.Rodrigues(rotation_vector)[0]
        return rotation_matrix, translation_vector


    def get_bad_markers(self, image):
        '''
        Returns image with bad markers.
        '''
        markers_coordinates, markers_id, bad_markers =\
        cv2.aruco.detectMarkers(image, self.dictionary)
        for markers in bad_markers:
            for i in range(0, len(markers[0])):
                point_1 = (markers[0][i][0], markers[0][i][1])
                point_2 = (markers[0][i - 1][0], markers[0][i - 1][1])
                cv2.line(image, point_1, point_2, color = (0, 0, 255))
        return image

    def get_markers(self, frame, show=False, image_name='Marked frame',
                    log_out=True):
        '''
        Get markers' coordinates from frame and add them to all_corners
        Set show=True to get output image.
        Returns markers coordinates ans ids.
        '''
        marked_frame = np.copy(frame)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        markers_coordinates, markers_id, bad_markers =\
        cv2.aruco.detectMarkers(gray_frame, self.dictionary)
        if len(markers_coordinates) > 0:
            num_corners, charuco_corners, charuco_ids =\
            cv2.aruco.interpolateCornersCharuco(markers_coordinates,
                                                 markers_id, gray_frame,
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
                    cv2.aruco.drawDetectedMarkers(marked_frame,
                                                  markers_coordinates,
                                                  markers_id)
                    cv2.aruco.drawDetectedCornersCharuco(marked_frame,
                                                         charuco_corners,
                                                         cornerColor = (0, 255,255))
                    message = 'Taken photos:' + str(self.num_frames)
                    cv2.putText(img=marked_frame, text=message, org=(20, 40),
                                fontFace=0, fontScale=0.6, color=(0, 0, 0),
                                thickness=2)
                    cv2.imshow(image_name, marked_frame)
        else:
            if log_out:
                print('Corners weren\'t detected')
        return charuco_corners, charuco_ids


    def check_calibration_charuco(self, image, column_number=7, row_number=5,
                          square_width = 100, show=True, line_width=3):
        '''
        Function for verifying the correctness of calibration, using image with
        ChArUco board. Returns image with points on the corners of the squares.
        '''
        rotation_matrix, translation_vector = self.estimate_board_pose(image)
        # Get parameters in vector form in homogeneous coordinates.
        extrinsic_matrix = np.identity(4)
        extrinsic_matrix[0:3, 0:3] = rotation_matrix
        extrinsic_matrix[0:3, 3] = translation_vector[0:3, 0]
        intrinsic_matrix = np.zeros((3, 4))
        intrinsic_matrix[0:3, 0:3] = self.camera_matrix
        undistored_image = cv2.undistort(image, self.camera_matrix, self.dist_coeff)
        # DRAW SQUARES ON BOARD.
        # Coordinates of squares on horizontal and vertical axis.
        for x in range(1, column_number + 1):
            for y in range(1, row_number + 1):
                # Get three corners of square in camera coordinate system.
                point_1 = extrinsic_matrix.dot([[square_width * x], [square_width * y], [0], [1]])
                point_2 = extrinsic_matrix.dot([[square_width * (x - 1)], [square_width * y], [0], [1]])
                point_3 = extrinsic_matrix.dot([[square_width * x], [square_width * (y - 1)], [0], [1]])
                # Get three corners of square on image in homogeneous coordinates.
                point_1 = intrinsic_matrix.dot(point_1)
                point_2 = intrinsic_matrix.dot(point_2)
                point_3 = intrinsic_matrix.dot(point_3)
                # Get normalized corners on image. Division by constant.
                point_1 /= point_1[2, 0]
                point_2 /= point_2[2, 0]
                point_3 /= point_3[2, 0]
                # Draw square on image.
                cv2.line(undistored_image, (int(point_1[0, 0]), int(point_1[1, 0])), (int(point_2[0, 0]), int(point_2[1, 0])), (255,0,0), line_width)
                cv2.line(undistored_image, (int(point_1[0, 0]), int(point_1[1, 0])), (int(point_3[0, 0]), int(point_3[1, 0])), (255,0,0), line_width)
        # DRAW AXIS ON BOARD.
        # Get three corners of board in camera coordinate system.
        point_1 = extrinsic_matrix.dot([[0], [0], [0], [1]])
        point_2 = extrinsic_matrix.dot([[square_width * column_number], [0], [0], [1]])
        point_3 = extrinsic_matrix.dot([[0], [square_width * row_number], [0], [1]])
        # Get three corners of board on image in homogeneous coordinates.
        point_1 = intrinsic_matrix.dot(point_1)
        point_2 = intrinsic_matrix.dot(point_2)
        point_3 = intrinsic_matrix.dot(point_3)
        # Get normalized corners on image. Division by constant.
        point_1 /= point_1[2, 0]
        point_2 /= point_2[2, 0]
        point_3 /= point_3[2, 0]
        # Draw axis on image.
        cv2.line(undistored_image, (int(point_1[0, 0]), int(point_1[1, 0])), (int(point_2[0, 0]), int(point_2[1, 0])), (0,0,255), line_width)
        cv2.line(undistored_image, (int(point_1[0, 0]), int(point_1[1, 0])), (int(point_3[0, 0]), int(point_3[1, 0])), (0,0,255), line_width)
        if show:
            cv2.imshow('check_board', undistored_image[::2, ::2, :])
            cv2.waitKey()
            cv2.destroyAllWindows()
        return undistored_image


    def check_calibration_charuco(self, image, column_number=7, row_number=5,
                          square_width = 100, show=True, line_width=3):
        '''
        Function for verifying the correctness of calibration, using image with
        ChArUco board. Returns image with points on the corners of the squares.
        '''
        rotation_matrix, translation_vector = self.estimate_board_pose(image)
        # Get parameters in vector form in homogeneous coordinates.
        extrinsic_matrix = np.identity(4)
        extrinsic_matrix[0:3, 0:3] = rotation_matrix
        extrinsic_matrix[0:3, 3] = translation_vector[0:3, 0]
        intrinsic_matrix = np.zeros((3, 4))
        intrinsic_matrix[0:3, 0:3] = self.camera_matrix
        undistored_image = cv2.undistort(image, self.camera_matrix, self.dist_coeff)
        # DRAW SQUARES ON BOARD.
        # Coordinates of squares on horizontal and vertical axis.
        for x in range(1, column_number + 1):
            for y in range(1, row_number + 1):
                # Get three corners of square in camera coordinate system.
                point_1 = extrinsic_matrix.dot([[square_width * x], [square_width * y], [0], [1]])
                point_2 = extrinsic_matrix.dot([[square_width * (x - 1)], [square_width * y], [0], [1]])
                point_3 = extrinsic_matrix.dot([[square_width * x], [square_width * (y - 1)], [0], [1]])
                # Get three corners of square on image in homogeneous coordinates.
                point_1 = intrinsic_matrix.dot(point_1)
                point_2 = intrinsic_matrix.dot(point_2)
                point_3 = intrinsic_matrix.dot(point_3)
                # Get normalized corners on image. Division by constant.
                point_1 /= point_1[2, 0]
                point_2 /= point_2[2, 0]
                point_3 /= point_3[2, 0]
                # Draw square on image.
                cv2.line(undistored_image, (int(point_1[0, 0]), int(point_1[1, 0])), (int(point_2[0, 0]), int(point_2[1, 0])), (255,0,0), line_width)
                cv2.line(undistored_image, (int(point_1[0, 0]), int(point_1[1, 0])), (int(point_3[0, 0]), int(point_3[1, 0])), (255,0,0), line_width)
        # DRAW AXIS ON BOARD.
        # Get three corners of board in camera coordinate system.
        point_1 = extrinsic_matrix.dot([[0], [0], [0], [1]])
        point_2 = extrinsic_matrix.dot([[square_width * column_number], [0], [0], [1]])
        point_3 = extrinsic_matrix.dot([[0], [square_width * row_number], [0], [1]])
        # Get three corners of board on image in homogeneous coordinates.
        point_1 = intrinsic_matrix.dot(point_1)
        point_2 = intrinsic_matrix.dot(point_2)
        point_3 = intrinsic_matrix.dot(point_3)
        # Get normalized corners on image. Division by constant.
        point_1 /= point_1[2, 0]
        point_2 /= point_2[2, 0]
        point_3 /= point_3[2, 0]
        # Draw axis on image.
        cv2.line(undistored_image, (int(point_1[0, 0]), int(point_1[1, 0])), (int(point_2[0, 0]), int(point_2[1, 0])), (0,0,255), line_width)
        cv2.line(undistored_image, (int(point_1[0, 0]), int(point_1[1, 0])), (int(point_3[0, 0]), int(point_3[1, 0])), (0,0,255), line_width)
        if show:
            cv2.imshow('check_board', undistored_image[::2, ::2, :])
            cv2.waitKey()
            cv2.destroyAllWindows()
        return undistored_image

    def get_disposition_charuco(self, image_1, image_2, camera_2):
        '''
        Calculates extrinsic matrix for camera_2, using images of the same
        ChArUco board. image_1 coresponds to self, image_2 coresponds to
        camera_2.
        '''
        # Calculation of extrinsic matrix for the first camera in
        # homogeneous coordinates.
        rotation_matrix_1, translation_vector_1 = self.estimate_board_pose(image_1)
        extrinsic_matrix_1 = np.identity(4)
        extrinsic_matrix_1[0:3, 0:3] = rotation_matrix_1
        extrinsic_matrix_1[0:3, 3] =  translation_vector_1[0:3, 0]
        # Calculation of extrinsic matrix for the first camera in
        # homogeneous coordinates.
        rotation_matrix_2, translation_vector_2 = camera_2.estimate_board_pose(image_2)
        extrinsic_matrix_2 = np.identity(4)
        extrinsic_matrix_2[0:3, 0:3] = rotation_matrix_2
        extrinsic_matrix_2[0:3, 3] =  translation_vector_2[0:3, 0]
        # Calculation of extrinsic matrix from first camer to second.
        extrinsic_matrix_to_camera_2 = extrinsic_matrix_2.dot(np.linalg.inv(extrinsic_matrix_1))
        rotation_matrix = extrinsic_matrix_to_camera_2[0:3, 0:3]
        translation_vector = extrinsic_matrix_to_camera_2[0:3, 3].reshape(3,-1)
        return rotation_matrix, translation_vector


    def check_transition(self, image_1, image_2, camera_2,
                         rotation_matrix_to_camera_2, translation_vector_to_camera_2,
                         column_number=6, row_number=5, square_width = 100,
                         show=True, line_width=3, scale=2):
        '''
        Function for verifying the correctness of extrinsic parameters.
        Returns images with points on the corners of the squares.
        '''
        # Estimate board pose for first camera.
        rotation_matrix_1, translation_vector_1 = self.estimate_board_pose(image_1)
        # Estimate board pose for second camera.
        undistored_image_1 = cv2.undistort(image_1, self.camera_matrix, self.dist_coeff)
        undistored_image_2 = cv2.undistort(image_2, camera_2.camera_matrix, camera_2.dist_coeff)
        # undistored_image_1 = np.copy(image_1)
        # undistored_image_2 = np.copy(image_2)
        # Build extrinsic matrix from first camera to second in vector form
        # in homogeneous coordinates.
        extrinsic_matrix_to_camera_2 = np.identity(4)
        extrinsic_matrix_to_camera_2[0:3, 0:3] = rotation_matrix_to_camera_2
        extrinsic_matrix_to_camera_2[0:3, 3] = translation_vector_to_camera_2[0:3, 0]
        # Build first camera's extrinsic and intrinsic matrix in vector
        # form in homogeneous
        # coordinates.
        extrinsic_matrix_camera_1 = np.identity(4)
        extrinsic_matrix_camera_1[0:3, 0:3] = rotation_matrix_1
        extrinsic_matrix_camera_1[0:3, 3] = translation_vector_1[0:3, 0]
        intrinsic_matrix_camera_1 = np.zeros((3, 4))
        intrinsic_matrix_camera_1[0:3, 0:3] = self.camera_matrix
        # Build second camera's intrinsic matrix in vector form in homogeneous
        # coordinates.
        intrinsic_matrix_camera_2 = np.zeros((3, 4))
        intrinsic_matrix_camera_2[0:3, 0:3] = camera_2.camera_matrix
        # DRAW CORNERS ON BOARD.
        # Coordinates of squares vertex on horizontal and vertical axis.
        for x in range(0, column_number + 1):
            for y in range(0, row_number + 1):
                # Draw point from space on image_1.
                point_in_camera_1_coordinates = extrinsic_matrix_camera_1.dot([[x * square_width], [y * square_width], [0], [1]])
                point_on_image_1 = intrinsic_matrix_camera_1.dot(point_in_camera_1_coordinates)
                point_on_image_1 /= point_on_image_1[2, 0]
                cv2.circle(undistored_image_1, (int(point_on_image_1[0, 0]), int(point_on_image_1[1, 0])), line_width, (0,0,255), -1)
                # Draw same point from space on image_2.
                point_in_camera_2_coordinates = extrinsic_matrix_to_camera_2.dot(point_in_camera_1_coordinates)
                point_on_image_2 = intrinsic_matrix_camera_2.dot(point_in_camera_2_coordinates)
                point_on_image_2 /= point_on_image_2[2, 0]
                cv2.circle(undistored_image_2, (int(point_on_image_2[0, 0]), int(point_on_image_2[1, 0])), line_width, (0,0,255), -1)
        if show:
            cv2.imshow('Camera_1', undistored_image_1[::scale, ::scale, :])
            cv2.imshow('Camera_2', undistored_image_2[::scale, ::scale, :])
            cv2.waitKey()
            cv2.destroyAllWindows()
        return undistored_image_1, undistored_image_2


    def check_transition_chess(self, image_1, image_2, camera_2,
                               rotation_matrix_1, translation_vector_1,
                               rotation_matrix_2, translation_vector_2,
                         rotation_matrix_to_camera_2, translation_vector_to_camera_2,
                         column_number=6, row_number=5, square_width = 100,
                         show=True, line_width=3, scale=2):
        '''
        Function for verifying the correctness of extrinsic parameters.
        Returns images with points on the corners of the squares.
        '''
        # Estimate board pose for second camera.
        undistored_image_1 = cv2.undistort(image_1, self.camera_matrix, self.dist_coeff)
        undistored_image_2 = cv2.undistort(image_2, camera_2.camera_matrix, camera_2.dist_coeff)
        # Build extrinsic matrix from first camera to second in vector form
        # in homogeneous coordinates.
        extrinsic_matrix_to_camera_2 = np.identity(4)
        extrinsic_matrix_to_camera_2[0:3, 0:3] = rotation_matrix_to_camera_2
        extrinsic_matrix_to_camera_2[0:3, 3] = translation_vector_to_camera_2[0:3, 0]
        # Build first camera's extrinsic matrix in vector form in homogeneous
        # coordinates.
        extrinsic_matrix_camera_1 = np.identity(4)
        extrinsic_matrix_camera_1[0:3, 0:3] = rotation_matrix_1
        extrinsic_matrix_camera_1[0:3, 3] = translation_vector_1[0:3, 0]
        intrinsic_matrix_camera_1 = np.zeros((3, 4))
        intrinsic_matrix_camera_1[0:3, 0:3] = self.camera_matrix
        # Build second camera's extrinsic matrix in vector form in homogeneous
        # coordinates.
        extrinsic_matrix_camera_2 = np.identity(4)
        extrinsic_matrix_camera_2[0:3, 0:3] = rotation_matrix_2
        extrinsic_matrix_camera_2[0:3, 3] = translation_vector_2[0:3, 0]
        intrinsic_matrix_camera_2 = np.zeros((3, 4))
        intrinsic_matrix_camera_2[0:3, 0:3] = camera_2.camera_matrix
        # DRAW CORNERS ON BOARD.
        # Coordinates of squares vertex on horizontal and vertical axis.
        for x in range(0, column_number + 1):
            for y in range(0, row_number + 1):
                # Draw point from space on image_1.
                point_in_camera_1_coordinates = extrinsic_matrix_camera_1.dot([[x * square_width], [y * square_width], [0], [1]])
                point_on_image_1 = intrinsic_matrix_camera_1.dot(point_in_camera_1_coordinates)
                point_on_image_1 /= point_on_image_1[2, 0]
                cv2.circle(undistored_image_1, (int(point_on_image_1[0, 0]), int(point_on_image_1[1, 0])), line_width, (0,0,255), -1)
                # Draw same point from space on image_2.
                point_in_camera_2_coordinates = extrinsic_matrix_to_camera_2.dot(point_in_camera_1_coordinates)
                point_on_image_2 = intrinsic_matrix_camera_2.dot(point_in_camera_2_coordinates)
                point_on_image_2 /= point_on_image_2[2, 0]
                cv2.circle(undistored_image_2, (int(point_on_image_2[0, 0]), int(point_on_image_2[1, 0])), line_width, (0,0,255), -1)
        if show:
            cv2.imshow('Camera_1', undistored_image_1[::scale, ::scale, :])
            cv2.imshow('Camera_2', undistored_image_2[::scale, ::scale, :])
            cv2.waitKey()
            cv2.destroyAllWindows()
        return undistored_image_1, undistored_image_2

if __name__ == '__main__':
    print('This module provides class for convenient charuco calibration')
