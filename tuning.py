import numpy as np

# Common initial state covariance parameters
def get_P0(scale: float = 100.0) -> np.ndarray:
    """Get initial state covariance matrix"""
    base_values = np.array([
        0.4,   # xi_g position
        0.4,   # eta_g position
        0.06,  # theta_g heading
        0.25,  # xi_a position
        0.25,  # eta_a position
        0.06   # theta_a heading
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
        0.03**2,  # xi_g noise - matched to LKF
        0.03**2,  # eta_g noise
        0.1**2,   # theta_g noise
        0.025**2, # xi_a noise
        0.025**2, # eta_a noise
        0.08**2   # theta_a noise
    ])

def get_EKF_R() -> np.ndarray:
    """Get EKF measurement noise covariance matrix"""
    return np.diag([
        0.02**2,  # azimuth_g noise
        4.0**2,   # range noise
        0.02**2,  # azimuth_a noise
        3.0**2,   # xi_a GPS noise
        3.0**2    # eta_a GPS noise
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
        0.015,   # xi_g noise - even tighter
        0.015,   # eta_g noise
        0.05,    # theta_g noise - much tighter heading
        0.01,    # xi_a noise - very tight for UAV
        0.01,    # eta_a noise
        0.05     # theta_a noise
    ])**2

def get_UKF_R() -> np.ndarray:
    """Get UKF measurement noise covariance matrix"""
    return np.diag([
        0.01**2,   # azimuth_g noise - trust angles even more
        2.0**2,    # range noise - trust range more
        0.01**2,   # azimuth_a noise
        1.0**2,    # xi_a GPS noise - very tight
        1.0**2     # eta_a GPS noise
    ])