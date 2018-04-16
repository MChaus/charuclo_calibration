"""
Charuco calibration

"""
import cv2
import numpy as np
import yaml

class Charuco_calibration:
    def __init__(self, **kwargs):
        valid_keys = [
                "squaresX",
                "squaresY",
                "square_length",
                "marker_length",
                "camera_matrix",
                "dist_coeff",
                "rvecs",
                "tvecs",
                "frame_shape"
                ]
        for key in valid_keys:
            self.__dict__[key] = kwargs.get(key)
        self.dictionary = cv2.aruco.getPredefinedDictionary(
                cv2.aruco.DICT_6X6_250
                )
        if (
                self.squaresX is not None and
                self.squaresY is not None and
                self.square_length is not None and
                self.marker_length is not None
            ):
            self.board = self.create_board()
        else:
            self.board = None
        self.all_corners = []
        self.all_ids = []
        self.num_frames = 0


    def __str__(self):
        message = (
                "squaresX :\n{}\n" +
                "squaresX type:\n{}\n" +
                "squaresY :\n{}\n" +
                "squaresY type:\n{}\n" +
                "square_length :\n{}\n" +
                "square_length type:\n{}\n" +
                "marker_length :\n{}\n" +
                "marker_length type:\n{}\n" +
                "camera_matrix :\n{}\n" +
                "camera_matrix type:\n{}\n" +
                "dist_coeff :\n{}\n" +
                "dist_coeff type:\n{}\n" +
                "rvecs :\n{}\n" +
                "rvecs type:\n{}\n" +
                "tvecs :\n{}\n" +
                "tvecs type:\n{}\n" +
                "frame_shape :\n{}\n" +
                "frame_shape type:\n{}\n"
                ).format(
                    self.squaresX,
                    type(self.squaresX),
                    self.squaresY,
                    type(self.squaresY),
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
                    self.frame_shape,
                    type(self.frame_shape)
                        )
        return message


    def create_board(self):
        return cv2.aruco.CharucoBoard_create(
            self.squaresX,
            self.squaresY,
            self.square_length,
            self.marker_length,
            self.dictionary
            )


    def live_calibration(self):
        capture = cv2.VideoCapture(cv2.CAP_ANY)
        if capture.isOpened():
            retval, frame = capture.read()
            self.frame_shape = frame.shape[0:2]
        while capture.isOpened():
            retval, frame = capture.read()
            # if SPACE was pressed
            # take frame and get markers from it
            if cv2.waitKey(33) % 256 == 32:
                self.get_markers(frame, show=True)
            # if ENTER was pressed
            # calibrate camera using Charuco_calibration
            elif cv2.waitKey(33) % 256 == 13:
                self.calibrate()
                self.draw_axis(frame)
            # if ESC was pressed
            # close all windows
            elif cv2.waitKey(33) % 256 == 27:
                print('ESC pressed, closing...')
                break
            cv2.imshow("camera", frame)
        capture.release()
        cv2.destroyAllWindows()


    def draw_axis(self, frame):
        axis_frame = np.copy(frame)
        cv2.aruco.drawAxis(
            axis_frame,
            self.camera_matrix,
            self.dist_coeff,
            self.rvecs[0],
            self.tvecs[0],
            length=0.1
            )
        cv2.imshow("Axis", axis_frame)


    def load_data(self, path_to_data=None):
        if path_to_data is None:
            print('Undefind path')
        else:
            with open(path_to_data) as _file:
                metadata = yaml.load(_file)
            self.squaresX = metadata.get('squaresX')
            self.squaresY = metadata.get('squaresY')
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
        if path_to_data is None:
            print('Undefind path')
        else:
            metadata = {
                'squaresX' : self.squaresX,
                'squaresY' : self.squaresY,
                'square_length' : self.square_length,
                'marker_length' : self.marker_length,
                'size' : np.asarray(self.frame_shape).tolist(),
                'camera_matrix' : np.asarray(self.camera_matrix).tolist(),
                'distortion_coefficient' : np.asarray(self.dist_coeff).tolist(),
                'rotation_vector' : np.asarray(self.rvecs).tolist(),
                'translation_vector' : np.asarray(self.tvecs).tolist(),
                }
            with open(path_to_data, 'w') as _file:
                yaml.dump(metadata, _file)


    def calibrate(self):
        (
            retval,
            self.camera_matrix,
            self.dist_coeff,
            self.rvecs,
            self.tvecs
            ) = cv2.aruco.calibrateCameraCharuco(
                self.all_corners,
                self.all_ids,
                self.board,
                self.frame_shape,
                None,
                None
                )
        print('Coefficients were calculated')


    def get_markers(self, frame, show=False):
        marked_frame = np.copy(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        (
            markers_coordinates,
            markers_id,
            bad_markers
            ) = cv2.aruco.detectMarkers(gray, self.dictionary)
        if len(markers_coordinates):
            (
                num_corners,
                charuco_corners,
                charuco_ids
                )= cv2.aruco.interpolateCornersCharuco(
                    markers_coordinates,
                    markers_id,
                    gray,
                    self.board
                    )
            if (
                    charuco_corners is not None and
                    charuco_ids is not None and
                    len(charuco_corners) > 3
                ):
                print('Corners were detected')
                self.all_corners.append(charuco_corners)
                self.all_ids.append(charuco_ids)
                self.num_frames += 1
                if show != False:
                    cv2.aruco.drawDetectedMarkers(
                        marked_frame,
                        markers_coordinates,
                        markers_id
                        )
                    cv2.aruco.drawDetectedCornersCharuco(
                        marked_frame,
                        charuco_corners,
                        cornerColor = (0, 255,255)
                        )
                    message = 'Taken photos:' + str(self.num_frames)
                    cv2.putText(
                        img=marked_frame,
                        text=message,
                        org=(20, 40),
                        fontFace=0,
                        fontScale=0.6,
                        color=(0, 0, 0),
                        thickness=2
                        )
                    cv2.imshow('Marked frame', marked_frame)
            else:
                print('Corners weren\'t detected')


if __name__ == '__main__':
    print('This module provides class for convenient charuco calibration')
