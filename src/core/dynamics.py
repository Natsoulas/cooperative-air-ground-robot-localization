# Copyright (c) 2024 Niko Natsoulas
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from typing import Tuple, List, Callable
import src.utils.constants as constants
import scipy.linalg

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
    """Combined system dynamics"""
    ugv_state = state[:3]
    uav_state = state[3:]
    ugv_controls = controls[:2]
    uav_controls = controls[2:]
    ugv_noise = noise[:3]
    uav_noise = noise[3:]
    
    ugv_derivs = ugv_dynamics(ugv_state, ugv_controls, ugv_noise, L)
    uav_derivs = uav_dynamics(uav_state, uav_controls, uav_noise)
    
    return np.concatenate([ugv_derivs, uav_derivs])

def system_jacobian(state: np.ndarray, controls: np.ndarray, L: float) -> np.ndarray:
    """Compute Jacobian of system dynamics with respect to state"""
    xi_g, eta_g, theta_g, xi_a, eta_a, theta_a = state
    v_g, phi_g, v_a, omega_a = controls
    
    F = np.zeros((6, 6))
    
    # UGV dynamics derivatives
    F[0, 2] = -v_g * np.sin(theta_g)
    F[1, 2] = v_g * np.cos(theta_g)
    
    # UAV dynamics derivatives
    F[3, 5] = -v_a * np.sin(theta_a)
    F[4, 5] = v_a * np.cos(theta_a)
    
    return F

def input_jacobian(state: np.ndarray, controls: np.ndarray, L: float) -> np.ndarray:
    """Compute Jacobian of system dynamics with respect to inputs"""
    xi_g, eta_g, theta_g, xi_a, eta_a, theta_a = state
    v_g, phi_g, v_a, omega_a = controls
    
    B = np.zeros((6, 4))
    
    # UGV dynamics derivatives
    B[0, 0] = np.cos(theta_g)
    B[1, 0] = np.sin(theta_g)
    B[2, 0] = np.tan(phi_g) / L
    B[2, 1] = v_g / (L * np.cos(phi_g)**2)
    
    # UAV dynamics derivatives
    B[3, 2] = np.cos(theta_a)
    B[4, 2] = np.sin(theta_a)
    B[5, 3] = 1
    
    return B

def continuous_to_discrete(A: np.ndarray, B: np.ndarray, Q: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert continuous system to discrete using van Loan's method"""
    n = A.shape[0]
    
    # Form the van Loan matrix
    M = np.zeros((2*n, 2*n))
    M[:n, :n] = -A
    M[:n, n:] = Q
    M[n:, n:] = A.T
    M = dt * M
    
    # Matrix exponential
    Phi = scipy.linalg.expm(M)
    
    # Extract discrete matrices
    F = Phi[n:, n:].T
    Q_d = F @ Phi[:n, n:]
    
    # Discretize input matrix using zero-order hold
    # For singular A, use Taylor series approximation
    if dt > 0:
        try:
            G = np.linalg.inv(A) @ (F - np.eye(n)) @ B
        except np.linalg.LinAlgError:
            # Use Taylor series approximation for singular A
            G = dt * (np.eye(n) + dt * A / 2) @ B
    else:
        G = np.zeros_like(B)
    
    return F, G, Q_d

def simulate_linearized_system(x0: np.ndarray, t: np.ndarray, L: float, control_func: Callable) -> np.ndarray:
    """Simulate linearized system dynamics"""
    DT = constants.DT  # Match truth simulator timestep
    n_states = len(x0)
    n_steps = len(t)
    linear_states = np.zeros((n_steps, n_states))
    linear_states[0] = x0
    
    # Initialize nominal trajectory at initial state
    nominal_state = x0.copy()
    
    for i in range(n_steps-1):
        current_t = t[i]
        controls = control_func(current_t)
        noise = np.zeros(6)
        
        # First compute nominal trajectory point using RK4
        k1 = combined_dynamics(nominal_state, controls, noise, L)
        k2 = combined_dynamics(nominal_state + DT/2 * k1, controls, noise, L)
        k3 = combined_dynamics(nominal_state + DT/2 * k2, controls, noise, L)
        k4 = combined_dynamics(nominal_state + DT * k3, controls, noise, L)
        next_nominal = nominal_state + DT/6 * (k1 + 2*k2 + 2*k3 + k4)
        
        # Normalize angles in nominal trajectory
        next_nominal[2] = np.mod(next_nominal[2] + np.pi, 2*np.pi) - np.pi
        next_nominal[5] = np.mod(next_nominal[5] + np.pi, 2*np.pi) - np.pi
        
        # Get system matrices at current nominal point
        A = system_jacobian(nominal_state, controls, L)
        B = input_jacobian(nominal_state, controls, L)
        Q = np.zeros((6, 6))
        
        # Discretize system
        F, G, _ = continuous_to_discrete(A, B, Q, DT)
        
        # Propagate error state
        delta_x = linear_states[i] - nominal_state
        linear_states[i+1] = next_nominal + F @ delta_x
        
        # Update nominal state for next iteration
        nominal_state = next_nominal
        
        # Normalize angles
        linear_states[i+1, 2] = np.mod(linear_states[i+1, 2] + np.pi, 2*np.pi) - np.pi
        linear_states[i+1, 5] = np.mod(linear_states[i+1, 5] + np.pi, 2*np.pi) - np.pi
    
    return linear_states
