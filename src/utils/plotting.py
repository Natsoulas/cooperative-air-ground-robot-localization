import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

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
