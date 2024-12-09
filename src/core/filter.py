# Copyright (c) 2024 Niko Natsoulas
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from typing import Tuple, Callable
from src.core.dynamics import combined_dynamics
from src.core.measurement import measurement_model

def system_jacobian(state: np.ndarray, controls: np.ndarray, L: float) -> np.ndarray:
    """
    Compute Jacobian of system dynamics with respect to state
    """
    xi_g, eta_g, theta_g, xi_a, eta_a, theta_a = state
    v_g, phi_g, v_a, omega_a = controls
    
    # Initialize Jacobian matrix
    F = np.zeros((6, 6))
    
    # UGV dynamics derivatives
    F[0, 2] = -v_g * np.sin(theta_g)  # d(xi_g_dot)/d(theta_g)
    F[1, 2] = v_g * np.cos(theta_g)   # d(eta_g_dot)/d(theta_g)
    
    # UAV dynamics derivatives
    F[3, 5] = -v_a * np.sin(theta_a)  # d(xi_a_dot)/d(theta_a)
    F[4, 5] = v_a * np.cos(theta_a)   # d(eta_a_dot)/d(theta_a)
    
    return F


def measurement_jacobian(state: np.ndarray) -> np.ndarray:
    """
    Compute Jacobian of measurement model with respect to state
    """
    xi_g, eta_g, theta_g, xi_a, eta_a, theta_a = state
    
    # Calculate relative positions
    delta_xi = xi_a - xi_g
    delta_eta = eta_a - eta_g
    r_squared = delta_xi**2 + delta_eta**2
    r = np.sqrt(r_squared)
    
    # Initialize Jacobian matrix (5 measurements × 6 states)
    H = np.zeros((5, 6))
    
    # Derivatives for azimuth from UGV to UAV
    H[0, 0] = delta_eta / r_squared
    H[0, 1] = -delta_xi / r_squared
    H[0, 2] = -1
    H[0, 3] = -delta_eta / r_squared
    H[0, 4] = delta_xi / r_squared
    
    # Derivatives for range measurement
    H[1, 0] = -delta_xi / r
    H[1, 1] = -delta_eta / r
    H[1, 3] = delta_xi / r
    H[1, 4] = delta_eta / r
    
    # Derivatives for azimuth from UAV to UGV
    H[2, 0] = delta_eta / r_squared
    H[2, 1] = -delta_xi / r_squared
    H[2, 3] = -delta_eta / r_squared
    H[2, 4] = delta_xi / r_squared
    H[2, 5] = -1
    
    # Derivatives for UAV GPS measurements
    H[3, 3] = 1  # d(xi_a_gps)/d(xi_a)
    H[4, 4] = 1  # d(eta_a_gps)/d(eta_a)
    
    return H

def input_jacobian(state: np.ndarray, controls: np.ndarray, L: float) -> np.ndarray:
    """
    Compute Jacobian of system dynamics with respect to inputs
    """
    xi_g, eta_g, theta_g, xi_a, eta_a, theta_a = state
    v_g, phi_g, v_a, omega_a = controls

    # Initialize Jacobian matrix
    B = np.zeros((6, 4))
    
    # UGV dynamics derivatives with respect to inputs
    B[0, 0] = np.cos(theta_g)  
    B[1, 0] = np.sin(theta_g) 
    B[2, 0] = np.tan(phi_g) / L  
    B[2, 1] = v_g * (1 / (L * np.cos(phi_g)**2)) 

    # UAV dynamics derivatives with respect to inputs
    B[3, 2] = np.cos(theta_a) 
    B[4, 2] = np.sin(theta_a)
    B[5, 3] = 1  

    return B

class LinearizedKalmanFilter:
    def __init__(self, x0: np.ndarray, P0: np.ndarray, Q: np.ndarray, R: np.ndarray, L: float):
        self.x = x0.copy()
        self.x_nominal = x0.copy()
        self.delta_x = np.zeros_like(x0)
        self.P = P0
        self.Q = Q
        self.R = R
        self.L = L
        
        # Initialize nominal trajectory storage
        self.nominal_window_size = 20
        self.nominal_points = [x0.copy()]
        self.F_nominals = []
        self.H_nominals = []
        
        # Initialize first Jacobians with zero controls
        F = system_jacobian(x0, np.zeros(4), self.L)
        H = measurement_jacobian(x0)
        self.F_nominals.append(F)
        self.H_nominals.append(H)
        
    def update_nominal_trajectory(self, controls: np.ndarray, dt: float):
        """Improved nominal trajectory propagation using actual controls"""
        current_nominal = self.nominal_points[-1]
        
        # Use multiple integration steps for better accuracy
        num_substeps = 4
        dt_sub = dt / num_substeps
        next_nominal = current_nominal.copy()
        
        for _ in range(num_substeps):
            # RK4 integration with actual controls
            noise = np.zeros(6)
            k1 = combined_dynamics(next_nominal, controls, noise, self.L)
            k2 = combined_dynamics(next_nominal + dt_sub/2 * k1, controls, noise, self.L)
            k3 = combined_dynamics(next_nominal + dt_sub/2 * k2, controls, noise, self.L)
            k4 = combined_dynamics(next_nominal + dt_sub * k3, controls, noise, self.L)
            
            next_nominal = next_nominal + dt_sub/6 * (k1 + 2*k2 + 2*k3 + k4)
            
            # Normalize angles in nominal trajectory
            next_nominal[2] = np.mod(next_nominal[2] + np.pi, 2*np.pi) - np.pi
            next_nominal[5] = np.mod(next_nominal[5] + np.pi, 2*np.pi) - np.pi
        
        # Compute Jacobians at new nominal point using actual controls
        F = system_jacobian(next_nominal, controls, self.L)
        H = measurement_jacobian(next_nominal)
        
        # Update storage with new point
        self.nominal_points.append(next_nominal)
        self.F_nominals.append(F)
        self.H_nominals.append(H)
        
        if len(self.nominal_points) > self.nominal_window_size:
            self.nominal_points.pop(0)
            self.F_nominals.pop(0)
            self.H_nominals.pop(0)
        
        self.x_nominal = next_nominal
        
    def predict(self, controls: np.ndarray, dt: float):
        # Update nominal trajectory using actual controls
        self.update_nominal_trajectory(controls, dt)
        
        # Use latest nominal trajectory Jacobian
        F_d = np.eye(6) + self.F_nominals[-1] * dt
        Q_d = self.Q * dt
        
        # Propagate error state
        self.delta_x = F_d @ self.delta_x
        
        # Update error covariance with Joseph form for better numerical stability
        self.P = F_d @ self.P @ F_d.T + Q_d
        
        # Update full state estimate
        self.x = self.x_nominal + self.delta_x
        
        # Normalize angles in state estimate
        self.x[2] = np.mod(self.x[2] + np.pi, 2*np.pi) - np.pi
        self.x[5] = np.mod(self.x[5] + np.pi, 2*np.pi) - np.pi
        
    def update(self, measurement: np.ndarray):
        # Use latest nominal measurement Jacobian
        H = self.H_nominals[-1]
        
        # Compute expected measurement using current nominal point
        expected_meas = measurement_model(self.x_nominal, np.zeros(5))
        
        # Compute innovation
        innovation = measurement - expected_meas - H @ self.delta_x
        
        # Normalize angle innovations
        innovation[0] = np.mod(innovation[0] + np.pi, 2*np.pi) - np.pi  # UGV azimuth
        innovation[2] = np.mod(innovation[2] + np.pi, 2*np.pi) - np.pi  # UAV azimuth
        
        # Compute Kalman gain with better numerical stability
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        
        # Update error state
        self.delta_x = self.delta_x + K @ innovation
        
        # Update covariance using Joseph form
        I_KH = np.eye(6) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T
        
        # Update full state estimate
        self.x = self.x_nominal + self.delta_x
        
        # Normalize angles in state estimate
        self.x[2] = np.mod(self.x[2] + np.pi, 2*np.pi) - np.pi
        self.x[5] = np.mod(self.x[5] + np.pi, 2*np.pi) - np.pi

class ExtendedKalmanFilter:
    def __init__(self, x0: np.ndarray, P0: np.ndarray, Q: np.ndarray, R: np.ndarray, L: float):
        self.x = x0  # State estimate
        self.P = P0  # State covariance
        self.Q = Q   # Process noise covariance
        self.R = R   # Measurement noise covariance
        self.L = L   # Vehicle parameter
        
        # Add numerical stability parameters
        self.min_cov_eigenval = 1e-10
        self.max_cov_eigenval = 1e6
        
    def normalize_state(self):
        """Normalize angle states to [-π, π]"""
        self.x[2] = np.mod(self.x[2] + np.pi, 2*np.pi) - np.pi  # UGV heading
        self.x[5] = np.mod(self.x[5] + np.pi, 2*np.pi) - np.pi  # UAV heading
        
    def ensure_covariance_validity(self):
        """Ensure covariance matrix stays well-conditioned"""
        # Symmetric part
        self.P = (self.P + self.P.T) / 2
        
        # Eigenvalue bounds
        eigvals, eigvecs = np.linalg.eigh(self.P)
        eigvals = np.clip(eigvals, self.min_cov_eigenval, self.max_cov_eigenval)
        self.P = eigvecs @ np.diag(eigvals) @ eigvecs.T
        
    def predict(self, controls: np.ndarray, dt: float):
        """Improved prediction step with multiple integration substeps"""
        # Use multiple integration steps for better accuracy
        num_substeps = 4
        dt_sub = dt / num_substeps
        
        for _ in range(num_substeps):
            # Compute current Jacobian
            F = system_jacobian(self.x, controls, self.L)
            
            # Discretize system for small timestep
            F_d = np.eye(6) + F * dt_sub
            Q_d = self.Q * dt_sub
            
            # Predict state using RK4 integration
            k1 = combined_dynamics(self.x, controls, np.zeros(6), self.L)
            k2 = combined_dynamics(self.x + dt_sub/2 * k1, controls, np.zeros(6), self.L)
            k3 = combined_dynamics(self.x + dt_sub/2 * k2, controls, np.zeros(6), self.L)
            k4 = combined_dynamics(self.x + dt_sub * k3, controls, np.zeros(6), self.L)
            
            self.x = self.x + dt_sub/6 * (k1 + 2*k2 + 2*k3 + k4)
            
            # Update covariance using Joseph form
            self.P = F_d @ self.P @ F_d.T + Q_d
            
            # Normalize angles and ensure covariance validity
            self.normalize_state()
            self.ensure_covariance_validity()
    
    def update(self, measurement: np.ndarray):
        """Improved update step with robust innovation handling"""
        # Get measurement Jacobian
        H = measurement_jacobian(self.x)
        
        # Compute expected measurement
        expected_meas = measurement_model(self.x, np.zeros(5))
        
        # Compute innovation with angle wrapping
        innovation = measurement - expected_meas
        
        # Normalize angle innovations
        innovation[0] = np.mod(innovation[0] + np.pi, 2*np.pi) - np.pi  # UGV azimuth
        innovation[2] = np.mod(innovation[2] + np.pi, 2*np.pi) - np.pi  # UAV azimuth
        
        # Robust innovation covariance
        S = H @ self.P @ H.T + self.R
        
        # Ensure S is well-conditioned
        S = (S + S.T) / 2  # Ensure symmetry
        
        try:
            # Use SVD for more stable inverse
            U, s, Vh = np.linalg.svd(S)
            s_inv = np.where(s > 1e-10, 1/s, 0)
            S_inv = (Vh.T * s_inv) @ U.T
            
            # Compute Kalman gain
            K = self.P @ H.T @ S_inv
            
            # Update state
            self.x = self.x + K @ innovation
            
            # Update covariance using Joseph form
            I_KH = np.eye(6) - K @ H
            self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T
            
        except np.linalg.LinAlgError:
            # If SVD fails, skip update
            print("Warning: Update step failed due to numerical issues")
            return
        
        # Normalize angles and ensure covariance validity
        self.normalize_state()
        self.ensure_covariance_validity()
