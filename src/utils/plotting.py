# Copyright (c) 2024 Niko Natsoulas
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
from scipy.stats import chi2
from src.core.measurement import measurement_model, measurement_jacobian

def setup_plots() -> Tuple[plt.Figure, np.ndarray]:
    """Create figure and subplots for visualization"""
    fig = plt.figure(figsize=(15, 10))
    axes = np.array([
        plt.subplot2grid((3, 3), (0, 0), colspan=2),  # Top-left: 2D trajectory
        plt.subplot2grid((3, 3), (0, 2)),             # Top-right: Range
        plt.subplot2grid((3, 3), (1, 0)),             # Mid-left: UGV states
        plt.subplot2grid((3, 3), (1, 1)),             # Mid-center: UAV states
        plt.subplot2grid((3, 3), (1, 2)),             # Mid-right: Relative angles
        plt.subplot2grid((3, 3), (2, 0), colspan=3)   # Bottom: Controls
    ])
    
    fig.tight_layout(pad=3.0)
    return fig, axes

def plot_simulation_results(t: np.ndarray, 
                          true_states: np.ndarray, 
                          measurements: np.ndarray,
                          controls: np.ndarray):
    """
    Plot simulation results
    Args:
        t: time vector
        true_states: array of true states [N, 6]
        measurements: array of measurements [N, 5]
        controls: array of control inputs [N, 4]
    """
    fig, axes = setup_plots()
    
    # Unpack states
    xi_g, eta_g = true_states[:, 0], true_states[:, 1]
    theta_g = true_states[:, 2]
    xi_a, eta_a = true_states[:, 3], true_states[:, 4]
    theta_a = true_states[:, 5]
    
    # Unpack controls
    v_g, phi_g = controls[:, 0], controls[:, 1]
    v_a, omega_a = controls[:, 2], controls[:, 3]
    
    # Plot 2D trajectory
    ax = axes[0]
    ax.plot(xi_g, eta_g, 'b-', label='UGV')
    ax.plot(xi_a, eta_a, 'r-', label='UAV')
    ax.plot(xi_g[0], eta_g[0], 'bo', label='UGV Start')
    ax.plot(xi_a[0], eta_a[0], 'ro', label='UAV Start')
    # Plot arrows for heading every N points
    N = len(t) // 10
    for i in range(0, len(t), N):
        # UGV heading arrow
        ax.arrow(xi_g[i], eta_g[i], 
                np.cos(theta_g[i]), np.sin(theta_g[i]), 
                head_width=0.5, head_length=1, fc='b', ec='b')
        # UAV heading arrow
        ax.arrow(xi_a[i], eta_a[i], 
                np.cos(theta_a[i]), np.sin(theta_a[i]), 
                head_width=0.5, head_length=1, fc='r', ec='r')
    ax.grid(True)
    ax.set_xlabel('East (m)')
    ax.set_ylabel('North (m)')
    ax.set_title('2D Trajectory')
    ax.legend()
    ax.axis('equal')
    
    # Plot range
    ax = axes[1]
    range_meas = measurements[:, 1]  # Range is second measurement
    true_range = np.sqrt((xi_a - xi_g)**2 + (eta_a - eta_g)**2)
    ax.plot(t, true_range, 'k-', label='True')
    ax.plot(t, range_meas, 'g.', label='Measured')
    ax.grid(True)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Range (m)')
    ax.set_title('Vehicle Range')
    ax.legend()
    
    # Plot UGV states
    ax = axes[2]
    ax.plot(t, xi_g, 'b-', label='xi_g')
    ax.plot(t, eta_g, 'r-', label='eta_g')
    ax.plot(t, theta_g, 'g-', label='theta_g')
    ax.grid(True)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('State')
    ax.set_title('UGV States')
    ax.legend()
    
    # Plot UAV states
    ax = axes[3]
    ax.plot(t, xi_a, 'b-', label='xi_a')
    ax.plot(t, eta_a, 'r-', label='eta_a')
    ax.plot(t, theta_a, 'g-', label='theta_a')
    ax.grid(True)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('State')
    ax.set_title('UAV States')
    ax.legend()
    
    # Plot relative angles
    ax = axes[4]
    true_azimuth = np.arctan2(eta_a - eta_g, xi_a - xi_g) - theta_g
    measured_azimuth = measurements[:, 0]  # Azimuth is first measurement
    ax.plot(t, true_azimuth, 'k-', label='True Azimuth')
    ax.plot(t, measured_azimuth, 'g.', label='Measured')
    ax.grid(True)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Angle (rad)')
    ax.set_title('Relative Angles')
    ax.legend()
    
    # Plot controls
    ax = axes[5]
    ax.plot(t, v_g, 'b-', label='v_g')
    ax.plot(t, phi_g, 'b--', label='phi_g')
    ax.plot(t, v_a, 'r-', label='v_a')
    ax.plot(t, omega_a, 'r--', label='omega_a')
    ax.grid(True)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Control Input')
    ax.set_title('Control Inputs')
    ax.legend()
    
    plt.show()

def plot_estimation_results(t: np.ndarray, 
                          true_states: np.ndarray,
                          lkf_states: np.ndarray,
                          ekf_states: np.ndarray,
                          lkf_covs: np.ndarray,
                          ekf_covs: np.ndarray,
                          measurements: np.ndarray):
    """Plot estimation results and comparisons"""
    fig = plt.figure(figsize=(15, 10))
    
    # Plot 2D trajectories
    ax1 = plt.subplot2grid((3, 2), (0, 0), colspan=2)
    ax1.plot(true_states[:, 0], true_states[:, 1], 'k-', label='True UGV')
    ax1.plot(true_states[:, 3], true_states[:, 4], 'k--', label='True UAV')
    ax1.plot(lkf_states[:, 0], lkf_states[:, 1], 'b-', label='LKF UGV')
    ax1.plot(lkf_states[:, 3], lkf_states[:, 4], 'b--', label='LKF UAV')
    ax1.plot(ekf_states[:, 0], ekf_states[:, 1], 'r-', label='EKF UGV')
    ax1.plot(ekf_states[:, 3], ekf_states[:, 4], 'r--', label='EKF UAV')
    ax1.grid(True)
    ax1.set_xlabel('East (m)')
    ax1.set_ylabel('North (m)')
    ax1.set_title('2D Trajectories Comparison')
    ax1.legend()
    
    # Plot UGV position errors
    ax2 = plt.subplot2grid((3, 2), (1, 0))
    lkf_ugv_err = np.sqrt((lkf_states[:, 0] - true_states[:, 0])**2 + 
                         (lkf_states[:, 1] - true_states[:, 1])**2)
    ekf_ugv_err = np.sqrt((ekf_states[:, 0] - true_states[:, 0])**2 + 
                         (ekf_states[:, 1] - true_states[:, 1])**2)
    ax2.plot(t, lkf_ugv_err, 'b-', label='LKF')
    ax2.plot(t, ekf_ugv_err, 'r-', label='EKF')
    ax2.grid(True)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Position Error (m)')
    ax2.set_title('UGV Position Error')
    ax2.legend()
    
    # Plot UAV position errors
    ax3 = plt.subplot2grid((3, 2), (1, 1))
    lkf_uav_err = np.sqrt((lkf_states[:, 3] - true_states[:, 3])**2 + 
                         (lkf_states[:, 4] - true_states[:, 4])**2)
    ekf_uav_err = np.sqrt((ekf_states[:, 3] - true_states[:, 3])**2 + 
                         (ekf_states[:, 4] - true_states[:, 4])**2)
    ax3.plot(t, lkf_uav_err, 'b-', label='LKF')
    ax3.plot(t, ekf_uav_err, 'r-', label='EKF')
    ax3.grid(True)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Position Error (m)')
    ax3.set_title('UAV Position Error')
    ax3.legend()
    
    # Plot heading errors
    ax4 = plt.subplot2grid((3, 2), (2, 0))
    lkf_heading_err = np.abs(np.mod(lkf_states[:, 2] - true_states[:, 2] + np.pi, 
                                  2*np.pi) - np.pi)
    ekf_heading_err = np.abs(np.mod(ekf_states[:, 2] - true_states[:, 2] + np.pi, 
                                  2*np.pi) - np.pi)
    ax4.plot(t, np.rad2deg(lkf_heading_err), 'b-', label='LKF UGV')
    ax4.plot(t, np.rad2deg(ekf_heading_err), 'r-', label='EKF UGV')
    ax4.grid(True)
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Heading Error (deg)')
    ax4.set_title('Heading Error')
    ax4.legend()
    
    # Plot filter uncertainties
    ax5 = plt.subplot2grid((3, 2), (2, 1))
    ax5.plot(t, np.sqrt(lkf_covs[:, 0, 0]), 'b-', label='LKF UGV East')
    ax5.plot(t, np.sqrt(ekf_covs[:, 0, 0]), 'r-', label='EKF UGV East')
    ax5.plot(t, np.sqrt(lkf_covs[:, 1, 1]), 'b--', label='LKF UGV North')
    ax5.plot(t, np.sqrt(ekf_covs[:, 1, 1]), 'r--', label='EKF UGV North')
    ax5.grid(True)
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Position Std Dev (m)')
    ax5.set_title('Filter Uncertainties')
    ax5.legend()
    
    plt.tight_layout()
    plt.show()

def plot_filter_differences(t: np.ndarray,
                          true_states: np.ndarray,
                          lkf_states: np.ndarray,
                          ekf_states: np.ndarray,
                          lkf_covs: np.ndarray,
                          ekf_covs: np.ndarray):
    """Plot the differences between LKF and EKF estimates and covariances"""
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle('Filter Differences Analysis')
    
    # Compute differences between filters
    filter_diff = lkf_states - ekf_states
    
    # Plot position differences between filters
    ax = axes[0, 0]
    ax.plot(t, filter_diff[:, 0], 'b-', label='East UGV')
    ax.plot(t, filter_diff[:, 1], 'r-', label='North UGV')
    ax.plot(t, filter_diff[:, 3], 'b--', label='East UAV')
    ax.plot(t, filter_diff[:, 4], 'r--', label='North UAV')
    ax.grid(True)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Position Difference (m)')
    ax.set_title('LKF - EKF Position Estimates')
    ax.legend()
    
    # Plot heading differences between filters
    ax = axes[0, 1]
    heading_diff_ugv = np.rad2deg(np.mod(filter_diff[:, 2] + np.pi, 2*np.pi) - np.pi)
    heading_diff_uav = np.rad2deg(np.mod(filter_diff[:, 5] + np.pi, 2*np.pi) - np.pi)
    ax.plot(t, heading_diff_ugv, 'b-', label='UGV')
    ax.plot(t, heading_diff_uav, 'r-', label='UAV')
    ax.grid(True)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Heading Difference (deg)')
    ax.set_title('LKF - EKF Heading Estimates')
    ax.legend()
    
    # Plot covariance differences
    ax = axes[1, 0]
    cov_diff = lkf_covs - ekf_covs
    ax.plot(t, cov_diff[:, 0, 0], 'b-', label='East UGV')
    ax.plot(t, cov_diff[:, 1, 1], 'r-', label='North UGV')
    ax.plot(t, cov_diff[:, 3, 3], 'b--', label='East UAV')
    ax.plot(t, cov_diff[:, 4, 4], 'r--', label='North UAV')
    ax.grid(True)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Covariance Difference')
    ax.set_title('LKF - EKF Covariance Differences')
    ax.legend()
    
    # Plot relative error to truth
    ax = axes[1, 1]
    lkf_err = np.sqrt(np.sum((lkf_states[:, [0,1,3,4]] - true_states[:, [0,1,3,4]])**2, axis=1))
    ekf_err = np.sqrt(np.sum((ekf_states[:, [0,1,3,4]] - true_states[:, [0,1,3,4]])**2, axis=1))
    ax.plot(t, lkf_err - ekf_err, 'k-')
    ax.grid(True)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Error Difference (m)')
    ax.set_title('LKF - EKF Total Position Error')
    
    # Plot normalized estimation error squared (NEES)
    ax = axes[2, 0]
    lkf_nees = compute_nees(true_states, lkf_states, lkf_covs)
    ekf_nees = compute_nees(true_states, ekf_states, ekf_covs)
    ax.plot(t, lkf_nees, 'b-', label='LKF')
    ax.plot(t, ekf_nees, 'r-', label='EKF')
    ax.grid(True)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('NEES')
    ax.set_title('Normalized Estimation Error Squared')
    ax.legend()
    
    # Plot relative computational efficiency
    ax = axes[2, 1]
    ax.plot(t, np.abs(lkf_nees - ekf_nees), 'k-')
    ax.grid(True)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('|NEES Difference|')
    ax.set_title('Absolute NEES Difference')
    
    plt.tight_layout()
    plt.show()

def compute_nees(true_states: np.ndarray, 
                filter_states: np.ndarray, 
                filter_covs: np.ndarray) -> np.ndarray:
    """
    Compute Normalized Estimation Error Squared (NEES) with proper angle wrapping
    """
    N = len(true_states)
    nees = np.zeros(N)
    
    for i in range(N):
        # Compute error with proper angle wrapping
        error = filter_states[i] - true_states[i]
        # Wrap angle differences to [-π, π]
        error[2] = np.mod(error[2] + np.pi, 2*np.pi) - np.pi  # UGV heading
        error[5] = np.mod(error[5] + np.pi, 2*np.pi) - np.pi  # UAV heading
        
        try:
            # Ensure covariance is well-conditioned
            min_eig = 1e-10
            P = filter_covs[i]
            P = (P + P.T) / 2  # Ensure symmetry
            eigvals, eigvecs = np.linalg.eigh(P)
            eigvals = np.maximum(eigvals, min_eig)
            P_inv = eigvecs @ np.diag(1/eigvals) @ eigvecs.T
            
            nees[i] = error @ P_inv @ error
        except np.linalg.LinAlgError:
            nees[i] = np.nan
            
    return nees

def compute_measurement_covs(filter_states: np.ndarray, 
                           filter_covs: np.ndarray) -> np.ndarray:
    """
    Compute measurement covariances for NIS calculation
    Args:
        filter_states: Filter state estimates [N, 6]
        filter_covs: Filter state covariances [N, 6, 6]
    Returns:
        measurement_covs: Measurement covariances [N, 5, 5]
    """
    N = len(filter_states)
    measurement_covs = np.zeros((N, 5, 5))
    
    for i in range(N):
        # Get measurement Jacobian at current state
        H = measurement_jacobian(filter_states[i])
        # Compute measurement covariance
        measurement_covs[i] = H @ filter_covs[i] @ H.T
        
    return measurement_covs

def plot_filter_performance(t: np.ndarray,
                          true_states: np.ndarray,
                          filter_states: np.ndarray,
                          filter_covs: np.ndarray,
                          measurements: np.ndarray,
                          R: np.ndarray,
                          filter_name: str):
    """Plot comprehensive performance metrics for a single filter"""
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle(f'{filter_name} Performance Analysis')
    
    # 1. Position errors with 2-sigma bounds
    ax = axes[0, 0]
    pos_err_ugv = filter_states[:, :2] - true_states[:, :2]
    pos_std_ugv = np.sqrt(np.array([filter_covs[i, :2, :2].diagonal() for i in range(len(t))]))
    
    ax.plot(t, pos_err_ugv[:, 0], 'b-', label='East Error')
    ax.plot(t, pos_err_ugv[:, 1], 'r-', label='North Error')
    ax.plot(t, 2*pos_std_ugv[:, 0], 'b--', alpha=0.5)
    ax.plot(t, -2*pos_std_ugv[:, 0], 'b--', alpha=0.5)
    ax.plot(t, 2*pos_std_ugv[:, 1], 'r--', alpha=0.5)
    ax.plot(t, -2*pos_std_ugv[:, 1], 'r--', alpha=0.5)
    ax.grid(True)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Error (m)')
    ax.set_title('UGV Position Errors with 2σ Bounds')
    ax.legend()
    
    # 2. NEES analysis
    ax = axes[0, 1]
    nees = compute_nees(true_states, filter_states, filter_covs)
    chi2_95 = chi2.ppf(0.95, df=6)

    ax.plot(t, nees, 'k-', label='NEES')
    ax.axhline(y=chi2_95, color='r', linestyle='--', label='95% Bound')
    ax.set_ylim([0, max(chi2_95 * 2, np.nanpercentile(nees, 95))])  # Ensure bound is visible
    ax.grid(True)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('NEES')
    ax.set_title('NEES')  # Changed from spelled out version
    ax.legend()
    
    # 3. NIS analysis
    ax = axes[1, 0]
    nis = compute_nis(measurements, filter_states, filter_covs, R)
    chi2_95_nis = chi2.ppf(0.95, df=5)
    
    # Plot NIS time history
    ax.plot(t, nis, 'b-', label='NIS')
    ax.axhline(y=chi2_95_nis, color='r', linestyle='--', label='95% Bound')
    ax.set_ylim([0, min(30, np.nanpercentile(nis, 99))])
    ax.grid(True)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('NIS')
    ax.set_title('Normalized Innovation Squared')
    ax.legend()
    
    # 4. NIS distribution
    ax = axes[1, 1]
    valid_nis = nis[~np.isnan(nis)]
    ax.hist(valid_nis, bins=30, density=True, alpha=0.6, color='b', label='Empirical')
    
    # Plot theoretical chi-square distribution
    x = np.linspace(0, chi2_95_nis * 1.5, 100)
    ax.plot(x, chi2.pdf(x, df=5), 'r-', label='χ²(5) PDF')
    ax.axvline(x=chi2_95_nis, color='r', linestyle='--', label='95% Bound')
    ax.grid(True)
    ax.set_xlabel('NIS')
    ax.set_ylabel('Density')
    ax.set_title('NIS Distribution')
    ax.legend()
    
    # 5. Heading error analysis
    ax = axes[2, 0]
    heading_err_ugv = np.mod(filter_states[:, 2] - true_states[:, 2] + np.pi, 2*np.pi) - np.pi
    heading_std_ugv = np.sqrt(np.array([filter_covs[i, 2, 2] for i in range(len(t))]))
    
    ax.plot(t, np.rad2deg(heading_err_ugv), 'b-', label='UGV Heading Error')
    ax.plot(t, 2*np.rad2deg(heading_std_ugv), 'r--', label='2σ Bound')
    ax.plot(t, -2*np.rad2deg(heading_std_ugv), 'r--')
    ax.grid(True)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Error (deg)')
    ax.set_title('Heading Error with 2σ Bounds')
    ax.legend()
    
    # 6. Covariance trace
    ax = axes[2, 1]
    cov_trace = np.array([np.trace(P) for P in filter_covs])
    ax.plot(t, cov_trace, 'b-')
    ax.grid(True)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Trace')
    ax.set_title('Covariance Matrix Trace')
    
    plt.tight_layout()
    plt.show()

def compute_nis(measurements: np.ndarray,
                filter_states: np.ndarray,
                filter_covs: np.ndarray,
                R: np.ndarray) -> np.ndarray:
    """
    Compute Normalized Innovation Squared (NIS)
    """
    N = len(measurements)
    nis = np.zeros(N)
    
    for i in range(N):
        # Compute predicted measurement
        pred_meas = measurement_model(filter_states[i], np.zeros(5))
        
        # Innovation with proper angle wrapping
        innovation = measurements[i] - pred_meas
        innovation[0] = np.mod(innovation[0] + np.pi, 2*np.pi) - np.pi  # azimuth_g
        innovation[2] = np.mod(innovation[2] + np.pi, 2*np.pi) - np.pi  # azimuth_a
        
        # Compute measurement Jacobian
        H = measurement_jacobian(filter_states[i])
        
        # Innovation covariance
        S = H @ filter_covs[i] @ H.T + R
        
        try:
            # Ensure S is well-conditioned
            S = (S + S.T) / 2
            nis[i] = innovation @ np.linalg.solve(S, innovation)
        except np.linalg.LinAlgError:
            nis[i] = np.nan
    
    return nis
