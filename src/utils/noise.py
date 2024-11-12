# Copyright (c) 2024 Niko Natsoulas
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from typing import Tuple

class NoiseGenerator:
    def __init__(self, state_noise_std: np.ndarray, meas_noise_std: np.ndarray):
        """
        Initialize noise generator
        Args:
            state_noise_std: standard deviations for state noise [6,]
            meas_noise_std: standard deviations for measurement noise [4,]
        """
        self.state_noise_std = state_noise_std
        self.meas_noise_std = meas_noise_std
    
    def generate_state_noise(self) -> np.ndarray:
        """Generate process noise for both vehicles"""
        return np.random.normal(0, self.state_noise_std)
    
    def generate_measurement_noise(self) -> np.ndarray:
        """Generate measurement noise"""
        return np.random.normal(0, self.meas_noise_std)
