#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 18:39:47 2018

@author: mchaus
"""

from calibration import Calibration
from charuco_calibration import Charuco_calibration

if __name__ == '__main__':
    cam_ch_calibr = Charuco_calibration(
            squaresX=5,
            squaresY=7,
            square_length=0.0725,
            marker_length=0.0435,
            )
    cam_ch_calibr.live_axis()
    # cam_ch_calibr.live_calibration()
    # print(cam_ch_calibr)
    # cam_ch_calibr.dump_data('dump.yaml')
    # cam_ch_calibr_2 = Charuco_calibration()
    # cam_ch_calibr_2.load_data('dump.yaml')
    # print(cam_ch_calibr_2)
