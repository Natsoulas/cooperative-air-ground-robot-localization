import numpy as np

# Common initial state covariance parameters
def get_P0() -> np.ndarray:
    """Initial covariance matrix - more conservative for UGV states"""
    P0 = np.diag([
        3.0**2,    # xi_g variance (m^2) - reduced
        3.0**2,    # eta_g variance (m^2) - reduced
        (np.pi/8)**2,  # theta_g variance (rad^2) - tighter
        5.0**2,    # xi_a variance (m^2) - reduced
        5.0**2,    # eta_a variance (m^2) - reduced
        (np.pi/6)**2   # theta_a variance (rad^2) - tighter
    ])
    return P0

# LKF Parameters
def get_LKF_Q() -> np.ndarray:
    """Process noise for LKF - tuned for nominal trajectory tracking"""
    Q = np.diag([
        0.5**2,    # xi_g noise - increased for nominal trajectory uncertainty
        0.5**2,    # eta_g noise - increased for nominal trajectory uncertainty
        (0.1)**2,  # theta_g noise - increased for heading uncertainty
        0.3**2,    # xi_a noise - unchanged
        0.3**2,    # eta_a noise - unchanged
        (0.1)**2   # theta_a noise - unchanged
    ])
    return Q

def get_LKF_R() -> np.ndarray:
    """Measurement noise for LKF - matched to true measurement characteristics"""
    R = np.diag([
        (0.02)**2,           # UGV azimuth - true noise
        8.0**2,              # Range - true noise
        (0.02)**2,           # UAV azimuth - true noise
        6.0**2,              # UAV GPS x - true noise
        6.0**2               # UAV GPS y - true noise
    ])
    return R

# EKF Parameters
def get_EKF_Q() -> np.ndarray:
    """Process noise for EKF - tuned for estimate-based linearization"""
    Q = np.diag([
        0.35**2,   # xi_g noise - moderate for estimate stability
        0.35**2,   # eta_g noise - moderate for estimate stability
        (0.08)**2, # theta_g noise - reduced for heading stability
        0.3**2,    # xi_a noise - unchanged
        0.3**2,    # eta_a noise - unchanged
        (0.1)**2   # theta_a noise - unchanged
    ])
    return Q

def get_EKF_R() -> np.ndarray:
    """Measurement noise for EKF - increased measurement confidence"""
    R = np.diag([
        (0.31*np.pi/180)**2,  # UGV azimuth - closer to LKF
        0.105**2,             # Range - closer to LKF
        (0.5*np.pi/180)**2,   # UAV azimuth - unchanged
        0.2**2,               # UAV GPS x - unchanged
        0.2**2                # UAV GPS y - unchanged
    ])
    return R

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
    """Process noise for UKF - similar to EKF but slightly more conservative"""
    Q = np.diag([
        0.4**2,    # xi_g noise - slightly larger than EKF
        0.4**2,    # eta_g noise - slightly larger than EKF
        (0.09)**2, # theta_g noise - slightly larger than EKF
        0.3**2,    # xi_a noise - unchanged
        0.3**2,    # eta_a noise - unchanged
        (0.1)**2   # theta_a noise - unchanged
    ])
    return Q

def get_UKF_R() -> np.ndarray:
    """Measurement noise for UKF - matched very close to LKF"""
    R = np.diag([
        (0.305*np.pi/180)**2, # UGV azimuth - nearly identical to LKF
        0.102**2,             # Range - nearly identical to LKF
        (0.5*np.pi/180)**2,   # UAV azimuth - unchanged
        0.2**2,               # UAV GPS x - unchanged
        0.2**2                # UAV GPS y - unchanged
    ])
    return R 