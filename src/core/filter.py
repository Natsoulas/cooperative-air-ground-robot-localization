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
    H[2, 0] = -delta_eta / r_squared
    H[2, 1] = delta_xi / r_squared
    H[2, 3] = delta_eta / r_squared
    H[2, 4] = -delta_xi / r_squared
    H[2, 5] = -1
    
    # Derivatives for UAV GPS measurements
    H[3, 3] = 1  # d(xi_a_gps)/d(xi_a)
    H[4, 4] = 1  # d(eta_a_gps)/d(eta_a)
    
    return H

class LinearizedKalmanFilter:
    def __init__(self, x0: np.ndarray, P0: np.ndarray, Q: np.ndarray, R: np.ndarray, L: float):
        self.x = x0.copy()  # State estimate
        self.x_nominal = x0.copy()  # Nominal trajectory
        self.delta_x = np.zeros_like(x0)  # Error state
        self.P = P0  # Error state covariance
        self.Q = Q   # Process noise covariance
        self.R = R   # Measurement noise covariance
        self.L = L   # Vehicle parameter
        
        # Get initial controls
        from src.utils.constants import V_G_0, PHI_G_0, V_A_0, OMEGA_A_0
        self.nominal_controls = np.array([V_G_0, PHI_G_0, V_A_0, OMEGA_A_0])
        
        # Pre-compute nominal trajectory Jacobian (should remain fixed)
        self.F_nominal = system_jacobian(self.x_nominal, self.nominal_controls, self.L)
        self.H_nominal = measurement_jacobian(self.x_nominal)
        
    def predict(self, controls: np.ndarray, dt: float):
        # Use fixed nominal trajectory Jacobian
        F_d = np.eye(6) + self.F_nominal * dt
        Q_d = self.Q * dt
        
        # Propagate nominal trajectory using nominal controls
        noise = np.zeros(6)
        dx_nominal = combined_dynamics(self.x_nominal, self.nominal_controls, noise, self.L)
        self.x_nominal = self.x_nominal + dx_nominal * dt
        
        # Propagate error state
        self.delta_x = F_d @ self.delta_x
        
        # Update full state estimate
        self.x = self.x_nominal + self.delta_x
        
        # Update error covariance
        self.P = F_d @ self.P @ F_d.T + Q_d
        
    def update(self, measurement: np.ndarray):
        # Use fixed nominal measurement Jacobian
        H = self.H_nominal
        
        # Compute expected measurement using nominal trajectory
        expected_meas = measurement_model(self.x_nominal, np.zeros(5))
        
        # Compute innovation
        innovation = measurement - expected_meas - H @ self.delta_x
        
        # Compute Kalman gain
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        
        # Update error state and covariance
        self.delta_x = self.delta_x + K @ innovation
        self.P = (np.eye(6) - K @ H) @ self.P
        
        # Update full state estimate
        self.x = self.x_nominal + self.delta_x

class ExtendedKalmanFilter:
    def __init__(self, x0: np.ndarray, P0: np.ndarray, Q: np.ndarray, R: np.ndarray, L: float):
        self.x = x0  # State estimate
        self.P = P0  # State covariance
        self.Q = Q   # Process noise covariance
        self.R = R   # Measurement noise covariance
        self.L = L   # Vehicle parameter
        
    def predict(self, controls: np.ndarray, dt: float):
        # Compute Jacobian at current state
        F = system_jacobian(self.x, controls, self.L)
        
        # Discretize system
        F_d = np.eye(6) + F * dt
        Q_d = self.Q * dt
        
        # Predict state using full nonlinear model
        noise = np.zeros(6)
        dx = combined_dynamics(self.x, controls, noise, self.L)
        self.x = self.x + dx * dt
        
        # Update covariance
        self.P = F_d @ self.P @ F_d.T + Q_d
        
    def update(self, measurement: np.ndarray):
        # Get measurement Jacobian
        H = measurement_jacobian(self.x)
        
        # Compute Kalman gain
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        
        # Compute expected measurement using full nonlinear model
        expected_meas = measurement_model(self.x, np.zeros(5))
        
        # Update state and covariance
        innovation = measurement - expected_meas
        self.x = self.x + K @ innovation
        self.P = (np.eye(6) - K @ H) @ self.P
