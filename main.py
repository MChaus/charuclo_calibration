#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 18:39:47 2018

@author: mchaus
"""

from calibration import Calibration
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
    cam_ch_calibr.draw_charuco_board(path='board_4_2.png', shape=(4500, 3000), marginSize=0)
    # cam_ch_calibr.calibrate_from_image(path_to_data='/home/mchaus/projects/gaze_estimation/local_charuco_calibration/dataset')
    # cam_ch_calibr.dump_data(path_to_data='dump.yaml')
    # cam_ch_calibr.load_data(path_to_data='dump.yaml')
    # cam_ch_calibr.axis_on_video(write_path = 'output2.avi')
    # cam_ch_calibr.axis_on_video(write_path='out_3x3.avi')
