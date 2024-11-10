import numpy as np
from typing import Tuple, List

def ugv_dynamics(state: np.ndarray, controls: np.ndarray, noise: np.ndarray, L: float) -> np.ndarray:
    """
    UGV dynamics equations (1-3)
    Args:
        state: [xi_g, eta_g, theta_g]
        controls: [v_g, phi_g]
        noise: [w_tilde_x_g, w_tilde_y_g, w_tilde_omega_g]
        L: wheel separation length
    Returns:
        state derivatives [xi_dot_g, eta_dot_g, theta_dot_g]
    """
    xi_g, eta_g, theta_g = state
    v_g, phi_g = controls
    w_x, w_y, w_omega = noise
    
    return np.array([
        v_g * np.cos(theta_g) + w_x,
        v_g * np.sin(theta_g) + w_y,
        (v_g / L) * np.tan(phi_g) + w_omega
    ])

def uav_dynamics(state: np.ndarray, controls: np.ndarray, noise: np.ndarray) -> np.ndarray:
    """
    UAV dynamics equations (4-6)
    Args:
        state: [xi_a, eta_a, theta_a]
        controls: [v_a, omega_a]
        noise: [w_tilde_x_a, w_tilde_y_a, w_tilde_omega_a]
    Returns:
        state derivatives [xi_dot_a, eta_dot_a, theta_dot_a]
    """
    xi_a, eta_a, theta_a = state
    v_a, omega_a = controls
    w_x, w_y, w_omega = noise
    
    return np.array([
        v_a * np.cos(theta_a) + w_x,
        v_a * np.sin(theta_a) + w_y,
        omega_a + w_omega
    ])

def combined_dynamics(state: np.ndarray, controls: np.ndarray, noise: np.ndarray, L: float) -> np.ndarray:
    """
    Combined system dynamics
    Args:
        state: [xi_g, eta_g, theta_g, xi_a, eta_a, theta_a]
        controls: [v_g, phi_g, v_a, omega_a]
        noise: [w_tilde_g (3), w_tilde_a (3)]
        L: wheel separation length
    Returns:
        combined state derivatives
    """
    # Extract states
    xi_g, eta_g, theta_g, xi_a, eta_a, theta_a = state
    
    # Extract controls
    v_g, phi_g, v_a, omega_a = controls
    
    # UGV dynamics (equations 1-3)
    xi_g_dot = v_g * np.cos(theta_g)
    eta_g_dot = v_g * np.sin(theta_g)
    theta_g_dot = (v_g / L) * np.tan(phi_g)
    
    # UAV dynamics (equations 4-6)
    xi_a_dot = v_a * np.cos(theta_a)
    eta_a_dot = v_a * np.sin(theta_a)
    theta_a_dot = omega_a
    
    return np.array([
        xi_g_dot,
        eta_g_dot,
        theta_g_dot,
        xi_a_dot,
        eta_a_dot,
        theta_a_dot
    ]) + noise
