#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 18:39:47 2018

@author: mchaus
"""

from calibration import Calibration
from charuco_calibration import Charuco_calibration

if __name__ == '__main__':
    cam_calibration = Calibration(board_shape=(9, 6), path_to_dataset='./dataset')
    cam_calibration.load_metadata()
    cam_ch_calibr = Charuco_calibration(
            squaresX=5,
            squaresY=7,
            squareLength=0.0725,
            markerLength=0.0425,
            camera_matrix=cam_calibration.matrix,
            dist_coeff=cam_calibration.distortion
            )
    
    
    cam_ch_calibr.calibrate()
    