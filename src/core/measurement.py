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
