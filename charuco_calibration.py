"""
Charuco calibration

"""
import cv2

class Charuco_calibration:
    def __init__(self, **kwargs):
        valid_keys = [
                "squaresX",
                "squaresY",
                "squareLength",
                "markerLength",
                "camera_matrix",
                "dist_coeff"
                ]
        for key in valid_keys:
            self.__dict__[key] = kwargs.get(key)
        self.dictionary = cv2.aruco.getPredefinedDictionary(
                cv2.aruco.DICT_6X6_250
                )
        self.board = cv2.aruco.CharucoBoard_create(
                self.squaresX, 
                self.squaresY, 
                self.squareLength, 
                self.markerLength, 
                self.dictionary
                )
    
    def calibrate(self):
        capture = cv2.VideoCapture(cv2.CAP_ANY)
        all_corners = []
        allIds = []
        while capture.isOpened():
            ret, frame = capture.read()
            if cv2.waitKey(33) % 256 == 32:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                res = cv2.aruco.detectMarkers(gray, self.dictionary)
                markers_coordinates, markers_id, bad_markers = res
                if len(markers_coordinates) > 0:
                    res2 = cv2.aruco.interpolateCornersCharuco(markers_coordinates,
                                                           markers_id,
                                                           gray,
                                                           self.board)
                    num_corners, charuco_corners, charuco_ids = res2
                    if (
                            charuco_corners is not None and
                            charuco_ids is not None     and
                            len(charuco_corners) > 3
                       ):
                        print('Corners were detected')
                        all_corners.append(charuco_corners)
                        allIds.append(charuco_ids)
                        cv2.aruco.drawDetectedMarkers(
                                frame,
                                markers_coordinates,
                                markers_id
                                )
                        cv2.aruco.drawDetectedCornersCharuco(
                                frame,
                                charuco_corners,
                                cornerColor = (0, 255,255)
                                )
                    else:
                        print('Corners weren\'t detected')
                    cv2.imshow('frame', frame)
            if cv2.waitKey(33) % 256 == 13:
                print('Enter was pressed')
                res3 = cv2.aruco.calibrateCameraCharuco(
                        all_corners,
                        allIds,
                        self.board,
                        gray.shape,
                        None,
                        None
                        )
                retval, camera_matrix, distortion, rvecs, tvecs = res3
                print('res was calculated')
                print(res3)
                cv2.aruco.drawAxis(
                        frame,
                        camera_matrix,
                        distortion,
                        rvecs[0],
                        tvecs[0],
                        length=0.1
                        )
                cv2.imshow('frame', frame)
            if cv2.waitKey(33) % 256 == 27:
                print('ESC pressed, closing...')
                break
            cv2.imshow('Camera', frame)
        capture.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    pass
