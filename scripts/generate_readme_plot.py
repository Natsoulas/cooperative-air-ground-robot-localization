# Copyright (c) 2024 Niko Natsoulas
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys

# Check for required packages
try:
    import numpy as np
    import matplotlib.pyplot as plt
    import scipy
    import scipy.linalg  # Explicitly check for scipy.linalg
except ImportError as e:
    print(f"Required package missing: {e}")
    print("Please install required packages using:")
    print("pip install numpy scipy matplotlib")
    sys.exit(1)

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from src.truth import TruthSimulator
from src.utils.constants import *
from src.core.filter import LinearizedKalmanFilter, ExtendedKalmanFilter, UnscentedKalmanFilter
from tuning import get_P0, get_LKF_Q, get_LKF_R, get_EKF_Q, get_EKF_R, get_UKF_Q, get_UKF_R

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

def control_input(t: float) -> np.ndarray:
    """Generate control inputs for both vehicles"""
    # UGV controls with sinusoidal steering
    v_g = V_G_0  # 2.0 m/s
    phi_g = PHI_G_0
    # UAV controls with varying turn rate
    v_a = V_A_0  # 12.0 m/s
    omega_a = OMEGA_A_0
    
    return np.array([v_g, phi_g, v_a, omega_a])

def generate_plot():
    # Load real data
    ydata = np.loadtxt('data/ydata.csv', delimiter=',')
    Rtrue = np.loadtxt('data/Rtrue.csv', delimiter=',')
    Qtrue = np.loadtxt('data/Qtrue.csv', delimiter=',')
    tvec = np.loadtxt('data/tvec.csv', delimiter=',')
    
    # Initialize truth simulator just for initial state
    truth_sim = TruthSimulator(L=L, dt=DT)
    x0 = np.array([XI_G_0, ETA_G_0, THETA_G_0, XI_A_0, ETA_A_0, THETA_A_0])
    
    # Initialize noise generator
    state_noise_std = get_state_noise_std()
    meas_noise_std = get_meas_noise_std()
    
    # Initialize Kalman filters with improved tuning
    P0 = 2*get_P0()  # Reduced initial scaling
    lkf = LinearizedKalmanFilter(x0.copy(), P0.copy(), get_LKF_Q(), get_LKF_R(), L)
    ekf = ExtendedKalmanFilter(x0.copy(), P0.copy(), get_EKF_Q(), get_EKF_R(), L)
    ukf = UnscentedKalmanFilter(x0.copy(), P0.copy(), get_UKF_Q(), get_UKF_R(), L)
    
    # Use true process and measurement noise if available, otherwise use tuned values
    Q = Qtrue if 'Qtrue' in locals() else get_EKF_Q()
    R = Rtrue if 'Rtrue' in locals() else get_EKF_R()
    
    # Prepare state storage
    num_timesteps = len(tvec)
    lkf_states = np.zeros((num_timesteps, 6))
    ekf_states = np.zeros((num_timesteps, 6))
    lkf_covs = np.zeros((num_timesteps, 6, 6))
    ekf_covs = np.zeros((num_timesteps, 6, 6))
    ukf_states = np.zeros((num_timesteps, 6))
    ukf_covs = np.zeros((num_timesteps, 6, 6))
    
    # Get control inputs (using same function as before)
    controls = np.array([control_input(t) for t in tvec])
    
    # Process measurements
    measurements = ydata.T  # Transpose to match expected format
    
    for i in range(num_timesteps):
        # Store current estimates
        lkf_states[i] = lkf.x
        ekf_states[i] = ekf.x
        lkf_covs[i] = lkf.P
        ekf_covs[i] = ekf.P
        ukf_states[i] = ukf.x
        ukf_covs[i] = ukf.P
        
        # Prediction step
        if i < num_timesteps - 1:
            dt = tvec[i+1] - tvec[i]  # Use actual time differences
            lkf.predict(controls[i], dt)
            ekf.predict(controls[i], dt)
            ukf.predict(controls[i], dt)
        
        # Update step (skip if NaN measurement)
        if not np.any(np.isnan(measurements[i])):
            lkf.update(measurements[i])
            ekf.update(measurements[i])
            ukf.update(measurements[i])
    
    # Create plot
    plt.figure(figsize=(10, 6))
    plt.plot(lkf_states[:, 0], lkf_states[:, 1], 'b-', label='LKF UGV')
    plt.plot(lkf_states[:, 3], lkf_states[:, 4], 'b--', label='LKF UAV')
    plt.plot(ekf_states[:, 0], ekf_states[:, 1], 'r-', label='EKF UGV')
    plt.plot(ekf_states[:, 3], ekf_states[:, 4], 'r--', label='EKF UAV')
    plt.plot(ukf_states[:, 0], ukf_states[:, 1], 'g-', label='UKF UGV')
    plt.plot(ukf_states[:, 3], ukf_states[:, 4], 'g--', label='UKF UAV')
    
    # Plot GPS measurements for UAV
    plt.scatter(measurements[:, 3], measurements[:, 4], 
               c='k', s=1, alpha=0.3, label='UAV GPS')
    
    plt.grid(True)
    plt.xlabel('East (m)')
    plt.ylabel('North (m)')
    plt.title('Vehicle Trajectories (Real Data)')
    plt.legend()
    plt.savefig('docs/images/trajectory_plot.png', dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    generate_plot()
