# Copyright (c) 2024 Niko Natsoulas
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from typing import Tuple, Callable
from src.core.dynamics import (
    combined_dynamics, system_jacobian, 
    input_jacobian, continuous_to_discrete
)
from src.core.measurement import measurement_model, measurement_jacobian
import scipy.linalg

class KalmanFilterBase:
    """Base class for Kalman filters with common utilities"""
    def __init__(self, x0: np.ndarray, P0: np.ndarray, Q: np.ndarray, R: np.ndarray, L: float):
        self.x = x0.copy()
        self.P = P0.copy()
        self.Q = Q.copy()
        self.R = R.copy()
        self.L = L
        
        # Numerical conditioning parameters
        self.min_cov_eigenval = 1e-8
        self.max_cov_eigenval = 1e4
        
        # Increased validation gate threshold (chi-square with 5 DOF, 99.9%)
        self.gate_threshold = 20.5  # Much more permissive
        
    @staticmethod
    def normalize_angle(angle: float) -> float:
        """Normalize angle to [-π, π]"""
        return np.arctan2(np.sin(angle), np.cos(angle))
    
    def normalize_state(self) -> None:
        """Normalize all angle states"""
        self.x[2] = self.normalize_angle(self.x[2])
        self.x[5] = self.normalize_angle(self.x[5])
    
    def ensure_covariance_validity(self) -> None:
        """Ensure covariance stays well-conditioned with safety margin"""
        self.P = (self.P + self.P.T) / 2
        eigvals, eigvecs = np.linalg.eigh(self.P)
        # Add small inflation factor for stability
        eigvals = np.clip(eigvals * 1.01, self.min_cov_eigenval, self.max_cov_eigenval)
        self.P = eigvecs @ np.diag(eigvals) @ eigvecs.T
    
    def validate_measurement(self, innovation: np.ndarray, S: np.ndarray) -> bool:
        """Validate measurement with more permissive threshold"""
        try:
            gamma = innovation.T @ np.linalg.solve(S, innovation)
            return gamma < self.gate_threshold
        except np.linalg.LinAlgError:
            return True  # Accept measurement if numerical issues
    
    def joseph_form_update(self, K: np.ndarray, H: np.ndarray, R: np.ndarray) -> None:
        """Joseph form covariance update with better numerical stability"""
        I = np.eye(len(self.x))
        PHt = self.P @ H.T
        self.P = (I - K @ H) @ self.P @ (I - K @ H).T + K @ R @ K.T
        self.ensure_covariance_validity()

class LinearizedKalmanFilter(KalmanFilterBase):
    def __init__(self, x0: np.ndarray, P0: np.ndarray, Q: np.ndarray, R: np.ndarray, L: float):
        super().__init__(x0, P0, Q, R, L)
        self.x_nominal = x0.copy()
        self.delta_x = np.zeros_like(x0)
    
    def predict(self, controls: np.ndarray, dt: float) -> None:
        """Prediction step with improved heading handling"""
        # RK4 integration
        k1 = combined_dynamics(self.x_nominal, controls, np.zeros(6), self.L)
        k2 = combined_dynamics(self.x_nominal + dt/2 * k1, controls, np.zeros(6), self.L)
        k3 = combined_dynamics(self.x_nominal + dt/2 * k2, controls, np.zeros(6), self.L)
        k4 = combined_dynamics(self.x_nominal + dt * k3, controls, np.zeros(6), self.L)
        
        # Update nominal state
        next_nominal = self.x_nominal + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
        
        # Normalize angles in nominal trajectory
        next_nominal[2] = self.normalize_angle(next_nominal[2])
        next_nominal[5] = self.normalize_angle(next_nominal[5])
        
        # Get system matrices
        A = system_jacobian(self.x_nominal, controls, self.L)
        B = input_jacobian(self.x_nominal, controls, self.L)
        
        # Discretize system with increased process noise for heading
        Q_c = self.Q.copy()
        Q_c[2,2] *= 2.0  # Increase heading process noise
        Q_c[5,5] *= 2.0  # Increase heading process noise
        F, _, Q_d = continuous_to_discrete(A, B, Q_c, dt)
        
        # Propagate error state
        self.delta_x = F @ self.delta_x
        
        # Normalize heading error states
        self.delta_x[2] = self.normalize_angle(self.delta_x[2])
        self.delta_x[5] = self.normalize_angle(self.delta_x[5])
        
        # Propagate covariance with Joseph form
        self.P = F @ self.P @ F.T + Q_d
        self.ensure_covariance_validity()
        
        # Update nominal trajectory
        self.x_nominal = next_nominal
        
        # Update total state with proper angle handling
        self.x = self.x_nominal.copy()
        self.x[0:2] += self.delta_x[0:2]  # UGV position
        self.x[3:5] += self.delta_x[3:5]  # UAV position
        self.x[2] = self.normalize_angle(self.x_nominal[2] + self.delta_x[2])  # UGV heading
        self.x[5] = self.normalize_angle(self.x_nominal[5] + self.delta_x[5])  # UAV heading
    
    def update(self, measurement: np.ndarray) -> None:
        """Simplified update step"""
        # Compute predicted measurement and Jacobian
        pred_meas = measurement_model(self.x_nominal, np.zeros(5))
        H = measurement_jacobian(self.x_nominal)
        
        # Compute innovation (with angle wrapping for angular measurements)
        innovation = measurement - pred_meas
        innovation[0] = self.normalize_angle(innovation[0])  # UGV azimuth
        innovation[2] = self.normalize_angle(innovation[2])  # UAV azimuth
        
        # Innovation covariance
        S = H @ self.P @ H.T + self.R
        S = (S + S.T) / 2
        
        # Skip validation temporarily to debug
        # if not self.validate_measurement(innovation, S):
        #     return
        
        # Kalman gain
        try:
            K = self.P @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            print("Warning: Innovation covariance inversion failed")
            return
            
        # Update error state
        self.delta_x = self.delta_x + K @ (innovation - H @ self.delta_x)
        
        # Use Joseph form for covariance update
        self.joseph_form_update(K, H, self.R)
        
        # Update total state with proper angle handling
        self.x = self.x_nominal.copy()
        self.x[0:2] += self.delta_x[0:2]  # UGV position
        self.x[3:5] += self.delta_x[3:5]  # UAV position
        self.x[2] = self.normalize_angle(self.x_nominal[2] + self.delta_x[2])
        self.x[5] = self.normalize_angle(self.x_nominal[5] + self.delta_x[5])

class ExtendedKalmanFilter(KalmanFilterBase):
    def predict(self, controls: np.ndarray, dt: float) -> None:
        """Simplified EKF prediction"""
        # RK4 integration
        k1 = combined_dynamics(self.x, controls, np.zeros(6), self.L)
        k2 = combined_dynamics(self.x + dt/2 * k1, controls, np.zeros(6), self.L)
        k3 = combined_dynamics(self.x + dt/2 * k2, controls, np.zeros(6), self.L)
        k4 = combined_dynamics(self.x + dt * k3, controls, np.zeros(6), self.L)
        
        self.x = self.x + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
        self.normalize_state()
        
        # Linearize and discretize
        A = system_jacobian(self.x, controls, self.L)
        B = input_jacobian(self.x, controls, self.L)
        F, _, Q_d = continuous_to_discrete(A, B, self.Q, dt)
        
        # Covariance propagation
        self.P = F @ self.P @ F.T + Q_d
        self.ensure_covariance_validity()
    
    def update(self, measurement: np.ndarray) -> None:
        """Simplified EKF update"""
        # Compute predicted measurement and Jacobian
        pred_meas = measurement_model(self.x, np.zeros(5))
        H = measurement_jacobian(self.x)
        
        # Innovation with angle wrapping
        innovation = measurement - pred_meas
        innovation[0] = self.normalize_angle(innovation[0])
        innovation[2] = self.normalize_angle(innovation[2])
        
        # Innovation covariance
        S = H @ self.P @ H.T + self.R
        S = (S + S.T) / 2
        
        # Skip validation temporarily
        # if not self.validate_measurement(innovation, S):
        #     return
        
        # Kalman gain
        try:
            K = self.P @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            print("Warning: Innovation covariance inversion failed")
            return
            
        # State update
        self.x = self.x + K @ innovation
        self.normalize_state()
        
        # Use Joseph form for covariance update
        self.joseph_form_update(K, H, self.R)

class UnscentedKalmanFilter(KalmanFilterBase):
    def __init__(self, x0: np.ndarray, P0: np.ndarray, Q: np.ndarray, R: np.ndarray, L: float):
        super().__init__(x0, P0, Q, R, L)
        
        # UKF parameters - matched to system dynamics
        self.n = len(x0)
        self.alpha = 0.1     # Much smaller alpha for more conservative estimates
        self.beta = 2.0      # Optimal for Gaussian
        self.kappa = 0.0     # Simplified tuning
        
        # Derived parameters
        self.lambda_ = self.alpha**2 * (self.n + self.kappa) - self.n
        self.gamma = np.sqrt(self.n + self.lambda_)
        
        # Calculate weights
        self.weights_m = np.full(2 * self.n + 1, 1.0 / (2 * (self.n + self.lambda_)))
        self.weights_c = self.weights_m.copy()
        self.weights_m[0] = self.lambda_ / (self.n + self.lambda_)
        self.weights_c[0] = self.weights_m[0] + (1 - self.alpha**2 + self.beta)

    def generate_sigma_points(self) -> np.ndarray:
        """Generate sigma points using current state and covariance"""
        # Ensure covariance is valid before computing sigma points
        self.ensure_covariance_validity()
        
        # Compute square root of P using Cholesky decomposition
        try:
            L = np.linalg.cholesky(self.P)
        except np.linalg.LinAlgError:
            # If Cholesky fails, use eigendecomposition as fallback
            eigvals, eigvecs = np.linalg.eigh(self.P)
            L = eigvecs @ np.diag(np.sqrt(np.maximum(eigvals, 1e-8))) @ eigvecs.T
        
        # Initialize sigma points matrix
        sigma_points = np.zeros((2 * self.n + 1, self.n))
        
        # Set mean as first sigma point
        sigma_points[0] = self.x
        
        # Generate remaining sigma points
        for i in range(self.n):
            # Positive direction
            sigma_points[i + 1] = self.x + self.gamma * L[:, i]
            # Negative direction
            sigma_points[i + 1 + self.n] = self.x - self.gamma * L[:, i]
            
            # Normalize angle states
            sigma_points[i + 1, 2] = self.normalize_angle(sigma_points[i + 1, 2])
            sigma_points[i + 1, 5] = self.normalize_angle(sigma_points[i + 1, 5])
            sigma_points[i + 1 + self.n, 2] = self.normalize_angle(sigma_points[i + 1 + self.n, 2])
            sigma_points[i + 1 + self.n, 5] = self.normalize_angle(sigma_points[i + 1 + self.n, 5])
        
        return sigma_points

    def predict(self, controls: np.ndarray, dt: float) -> None:
        """UKF prediction step with improved uncertainty handling"""
        # Generate sigma points with scaled parameters for better spread
        sigma_points = self.generate_sigma_points()
        propagated_points = np.zeros_like(sigma_points)
        
        # Pre-allocate arrays for efficiency
        sin_theta_g = np.zeros(len(sigma_points))
        cos_theta_g = np.zeros(len(sigma_points))
        sin_theta_a = np.zeros(len(sigma_points))
        cos_theta_a = np.zeros(len(sigma_points))
        
        # Propagate each sigma point with adaptive step size
        for i in range(len(sigma_points)):
            # Adaptive RK4 integration with error estimation
            state = sigma_points[i]
            h = dt  # Initial step size
            
            while h > 1e-6:  # Minimum step size threshold
                # RK4 with current step
                k1 = combined_dynamics(state, controls, np.zeros(6), self.L)
                k2 = combined_dynamics(state + h/2 * k1, controls, np.zeros(6), self.L)
                k3 = combined_dynamics(state + h/2 * k2, controls, np.zeros(6), self.L)
                k4 = combined_dynamics(state + h * k3, controls, np.zeros(6), self.L)
                
                # Compute two solutions for error estimation
                sol1 = state + h/6 * (k1 + 2*k2 + 2*k3 + k4)
                sol2 = state + h/6 * (k1 + 4*k2 + k4)  # Alternative formula
                
                # Estimate error
                error = np.max(np.abs(sol1 - sol2))
                
                if error < 1e-6:  # Error tolerance
                    propagated_points[i] = sol1
                    break
                
                h *= 0.5  # Reduce step size if error too large
            
            # Store trig values for efficient angle averaging
            propagated_points[i, 2] = self.normalize_angle(propagated_points[i, 2])
            propagated_points[i, 5] = self.normalize_angle(propagated_points[i, 5])
            sin_theta_g[i] = np.sin(propagated_points[i, 2])
            cos_theta_g[i] = np.cos(propagated_points[i, 2])
            sin_theta_a[i] = np.sin(propagated_points[i, 5])
            cos_theta_a[i] = np.cos(propagated_points[i, 5])
        
        # Update state estimate with robust weighted mean
        self.x = np.zeros(self.n)
        
        # Position states with Huber-like robust weighting
        for i in range(len(propagated_points)):
            # Compute robust weights
            pos_diff = np.linalg.norm(propagated_points[i, [0,1,3,4]] - 
                                    np.mean(propagated_points[:, [0,1,3,4]], axis=0))
            weight = min(1.0, 2.0/max(pos_diff, 1e-6))
            w = self.weights_m[i] * weight
            
            # Update positions
            self.x[0:2] += w * propagated_points[i, 0:2]
            self.x[3:5] += w * propagated_points[i, 3:5]
        
        # Angle states using vectorized circular mean
        self.x[2] = np.arctan2(
            np.sum(self.weights_m * sin_theta_g),
            np.sum(self.weights_m * cos_theta_g)
        )
        self.x[5] = np.arctan2(
            np.sum(self.weights_m * sin_theta_a),
            np.sum(self.weights_m * cos_theta_a)
        )
        
        # Modified covariance update with controlled scaling
        self.P = np.zeros((self.n, self.n))
        max_diff = 0.0
        
        for i in range(len(propagated_points)):
            diff = np.zeros(self.n)
            diff[0:2] = propagated_points[i, 0:2] - self.x[0:2]
            diff[3:5] = propagated_points[i, 3:5] - self.x[3:5]
            diff[2] = self.normalize_angle(propagated_points[i, 2] - self.x[2])
            diff[5] = self.normalize_angle(propagated_points[i, 5] - self.x[5])
            
            # Use Huber-like robust weighting
            diff_norm = np.linalg.norm(diff)
            weight = min(1.0, 2.0/max(diff_norm, 1e-6))
            self.P += weight * self.weights_c[i] * np.outer(diff, diff)
        
        # More conservative process noise scaling
        noise_scale = min(1.5, 1.0 + 0.1 * np.log1p(max_diff))
        self.P += noise_scale * dt * self.Q
        
        # Additional numerical stability checks
        eigvals = np.linalg.eigvals(self.P)
        if np.any(eigvals < self.min_cov_eigenval):
            self.P += self.min_cov_eigenval * np.eye(self.n)
        
        self.ensure_covariance_validity()
    
    def update(self, measurement: np.ndarray) -> None:
        """UKF update step"""
        # Generate sigma points
        sigma_points = self.generate_sigma_points()
        
        # Propagate sigma points through measurement function
        predicted_measurements = np.zeros((len(sigma_points), len(measurement)))
        for i in range(len(sigma_points)):
            predicted_measurements[i] = measurement_model(sigma_points[i], np.zeros(5))
            # Normalize predicted azimuth measurements
            predicted_measurements[i, 0] = self.normalize_angle(predicted_measurements[i, 0])
            predicted_measurements[i, 2] = self.normalize_angle(predicted_measurements[i, 2])
        
        # Compute predicted measurement mean with angle averaging
        y_pred = np.zeros(len(measurement))
        # Special handling for angular measurements
        y_pred[0] = np.arctan2(
            np.sum(self.weights_m * np.sin(predicted_measurements[:, 0])),
            np.sum(self.weights_m * np.cos(predicted_measurements[:, 0]))
        )
        y_pred[2] = np.arctan2(
            np.sum(self.weights_m * np.sin(predicted_measurements[:, 2])),
            np.sum(self.weights_m * np.cos(predicted_measurements[:, 2]))
        )
        # Regular averaging for non-angular measurements
        y_pred[1] = np.sum(self.weights_m * predicted_measurements[:, 1])
        y_pred[3:] = np.sum(self.weights_m.reshape(-1, 1) * predicted_measurements[:, 3:], axis=0)
        
        # Compute innovation covariance
        S = np.zeros((len(measurement), len(measurement)))
        Pxy = np.zeros((self.n, len(measurement)))
        
        for i in range(len(sigma_points)):
            diff_x = sigma_points[i] - self.x
            diff_y = predicted_measurements[i] - y_pred
            
            # Normalize angle differences
            diff_x[2] = self.normalize_angle(diff_x[2])
            diff_x[5] = self.normalize_angle(diff_x[5])
            diff_y[0] = self.normalize_angle(diff_y[0])  # UGV azimuth
            diff_y[2] = self.normalize_angle(diff_y[2])  # UAV azimuth
            
            S += self.weights_c[i] * np.outer(diff_y, diff_y)
            Pxy += self.weights_c[i] * np.outer(diff_x, diff_y)
        
        # Add measurement noise
        S += self.R
        
        # Compute Kalman gain
        try:
            K = Pxy @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            print("Warning: Innovation covariance inversion failed")
            return
        
        # Compute innovation with angle wrapping
        innovation = measurement - y_pred
        innovation[0] = self.normalize_angle(innovation[0])
        innovation[2] = self.normalize_angle(innovation[2])
        
        # Update state and covariance
        self.x = self.x + K @ innovation
        self.P = self.P - K @ S @ K.T
        
        # Normalize angles and ensure covariance validity
        self.normalize_state()
        self.ensure_covariance_validity()
