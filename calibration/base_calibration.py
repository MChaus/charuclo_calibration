# -*- coding: utf-8 -*-
'''Base object for calibration models.

'''

# Author: Mykhailo Chaus <mchaus97@gmail.com>
# License: BSD 3 clause

import os

class CalibrationBase:
    def read_images(self, path=None):
        if path is None:
            path =
