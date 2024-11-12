# Copyright (c) 2024 Niko Natsoulas
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np

def measurement_model(state: np.ndarray, noise: np.ndarray) -> np.ndarray:
    """
    Measurement model from system description
    Args:
        state: [xi_g, eta_g, theta_g, xi_a, eta_a, theta_a]
        noise: measurement noise vector [5,]
    Returns:
        measurements [azimuth_ugv, range, xi_a_gps, eta_a_gps]
    """
    xi_g, eta_g, theta_g, xi_a, eta_a, theta_a = state
    
    # Calculate relative measurements
    delta_xi = xi_a - xi_g
    delta_eta = eta_a - eta_g
    
    # Azimuth from UGV to UAV
    azimuth_g = np.arctan2(delta_eta, delta_xi) - theta_g
    
    # Range between vehicles
    range_meas = np.sqrt(delta_xi**2 + delta_eta**2)

    # Azimuth from UAV to UGV
    azimuth_a = np.arctan2(-delta_eta, -delta_xi) - theta_a
    
    # Combine measurements with noise
    measurements = np.array([
        azimuth_g,
        range_meas,
        azimuth_a,
        xi_a,
        eta_a
    ])
    
    return measurements + noise

def measurement_jacobian(state: np.ndarray) -> np.ndarray:
    """
    Compute Jacobian of measurement model
    Args:
        state: [xi_g, eta_g, theta_g, xi_a, eta_a, theta_a]
    Returns:
        H: measurement Jacobian [5, 6]
    """
    xi_g, eta_g, theta_g, xi_a, eta_a, theta_a = state
    
    # Compute relative positions
    delta_xi = xi_a - xi_g
    delta_eta = eta_a - eta_g
    r2 = delta_xi**2 + delta_eta**2
    r = np.sqrt(r2)
    
    # Initialize Jacobian matrix
    H = np.zeros((5, 6))
    
    # Derivatives for azimuth_g measurement
    H[0, 0] = -delta_eta / r2  # d(azimuth_g)/d(xi_g)
    H[0, 1] = delta_xi / r2    # d(azimuth_g)/d(eta_g)
    H[0, 2] = -1               # d(azimuth_g)/d(theta_g)
    H[0, 3] = delta_eta / r2   # d(azimuth_g)/d(xi_a)
    H[0, 4] = -delta_xi / r2   # d(azimuth_g)/d(eta_a)
    
    # Derivatives for range measurement
    H[1, 0] = -delta_xi / r    # d(range)/d(xi_g)
    H[1, 1] = -delta_eta / r   # d(range)/d(eta_g)
    H[1, 3] = delta_xi / r     # d(range)/d(xi_a)
    H[1, 4] = delta_eta / r    # d(range)/d(eta_a)
    
    # Derivatives for azimuth_a measurement
    H[2, 0] = delta_eta / r2   # d(azimuth_a)/d(xi_g)
    H[2, 1] = -delta_xi / r2   # d(azimuth_a)/d(eta_g)
    H[2, 3] = -delta_eta / r2  # d(azimuth_a)/d(xi_a)
    H[2, 4] = delta_xi / r2    # d(azimuth_a)/d(eta_a)
    H[2, 5] = -1               # d(azimuth_a)/d(theta_a)
    
    # Derivatives for GPS measurements
    H[3, 3] = 1                # d(xi_a_gps)/d(xi_a)
    H[4, 4] = 1                # d(eta_a_gps)/d(eta_a)
    
    return H
