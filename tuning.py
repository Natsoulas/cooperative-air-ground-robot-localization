import numpy as np

# Common initial state covariance parameters
def get_P0(scale: float = 1.0) -> np.ndarray:
    """Get initial state covariance matrix with increased uncertainty"""
    base_values = np.array([
        2.5,   # xi_g position - increased
        2.5,   # eta_g position - increased
        0.3,   # theta_g heading - increased
        2.5,   # xi_a position - increased
        2.5,   # eta_a position - increased
        0.3    # theta_a heading - increased
    ])
    return scale * np.diag(base_values**2)

# LKF Parameters
def get_LKF_Q() -> np.ndarray:
    """Get LKF process noise covariance matrix - increased uncertainty"""
    return np.diag([
        0.45**2,  # xi_g noise - increased
        0.45**2,  # eta_g noise - increased
        1.2**2,   # theta_g noise - significantly increased
        0.35**2,  # xi_a noise - slightly increased
        0.35**2,  # eta_a noise - slightly increased
        1.0**2    # theta_a noise - slightly increased
    ])

def get_LKF_R() -> np.ndarray:
    """Get LKF measurement noise covariance matrix - increased uncertainty"""
    return np.diag([
        0.08**2,  # azimuth_g noise - increased
        10.0**2,  # range noise - increased
        0.08**2,  # azimuth_a noise - increased
        7.0**2,   # xi_a GPS noise - increased
        7.0**2    # eta_a GPS noise - increased
    ])

# EKF Parameters
def get_EKF_Q() -> np.ndarray:
    """Get EKF process noise covariance matrix"""
    return np.diag([
        0.2,   # xi_g noise
        0.2,   # eta_g noise
        0.8,   # theta_g noise
        0.2,   # xi_a noise
        0.2,   # eta_a noise
        0.8    # theta_a noise
    ])**2

def get_EKF_R() -> np.ndarray:
    """Get EKF measurement noise covariance matrix"""
    return np.diag([
        0.05**2,  # azimuth_g noise
        8.0**2,   # range noise
        0.05**2,  # azimuth_a noise
        6.0**2,   # xi_a GPS noise
        6.0**2    # eta_a GPS noise
    ])

# Noise generator parameters
def get_state_noise_std() -> np.ndarray:
    """Get state noise standard deviations"""
    return np.array([
        0.3,   # xi_g noise
        0.3,   # eta_g noise
        0.15,  # theta_g noise (radians)
        0.3,   # xi_a noise
        0.3,   # eta_a noise
        0.15   # theta_a noise (radians)
    ])

def get_meas_noise_std() -> np.ndarray:
    """Get measurement noise standard deviations"""
    return np.array([
        0.02,    # azimuth_g noise (radians)
        8.0,     # range noise (meters)
        0.02,    # azimuth_a noise (radians)
        6.0,     # xi_a GPS noise (meters)
        6.0      # eta_a GPS noise (meters)
    ])

# UKF Parameters
def get_UKF_Q() -> np.ndarray:
    """Get UKF process noise covariance matrix"""
    return np.diag([
        0.2,   # xi_g noise
        0.2,   # eta_g noise
        0.8,   # theta_g noise
        0.2,   # xi_a noise
        0.2,   # eta_a noise
        0.8    # theta_a noise
    ])**2

def get_UKF_R() -> np.ndarray:
    """Get UKF measurement noise covariance matrix"""
    return np.diag([
        0.05**2,  # azimuth_g noise
        8.0**2,   # range noise
        0.05**2,  # azimuth_a noise
        6.0**2,   # xi_a GPS noise
        6.0**2    # eta_a GPS noise
    ])