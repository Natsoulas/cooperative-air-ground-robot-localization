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
