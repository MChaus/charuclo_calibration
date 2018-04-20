#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: mchaus
"""

from charuco_calibration import Charuco_calibration

if __name__ == '__main__':
    # cam_ch_calibr = Charuco_calibration(
    #         squaresX=5,
    #         squaresY=7,
    #         square_length=0.0725,
    #         marker_length=0.0435,
    #         )
    cam_ch_calibr = Charuco_calibration(
            squaresX=6,
            squaresY=4,
            square_length=0.1,
            marker_length=0.05,
            )
    # Examples

    # Live calibration without writing
    # cam_ch_calibr.live_calibration()

    # Get axis from camera
    cam_ch_calibr.axis_on_video()

    # Write charuco board 6x6
    # cam_ch_calibr.draw_charuco_board(path='board_6x6.png', size=(720, 720), margin_size=0)

    # Calibrate from images that were saved to path
    # path = '/home/mchaus/projects/gaze_estimation/local_charuco_calibration/dataset'
    # cam_ch_calibr.calibrate_from_images(path_to_data=path)
    # print(cam_ch_calibr)

    # Save calculated parameters
    # cam_ch_calibr.dump_data(path_to_data='dump.yaml')

    # Load calculated parameters
    # cam_ch_calibr.load_data(path_to_data='dump.yaml')
