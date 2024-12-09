# Copyright (c) 2024 Niko Natsoulas
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from src.utils.constants import *
from src.truth import TruthSimulator
from src.utils.noise import NoiseGenerator
from src.core.measurement import measurement_model
from src.core.dynamics import combined_dynamics, ugv_dynamics, uav_dynamics
from src.utils.plotting import plot_simulation_results, plot_estimation_results, plot_filter_differences, plot_filter_performance, plot_linearization_comparison
from src.core.filter import LinearizedKalmanFilter, ExtendedKalmanFilter, continuous_to_discrete, system_jacobian, input_jacobian
from src.utils.analysis import perform_nees_hypothesis_test, perform_nis_hypothesis_test
from src.utils.plotting import compute_nees, compute_nis
from typing import Callable
import src.utils.constants as constants

def control_input(t: float) -> np.ndarray:
    """Generate control inputs for both vehicles"""
    # UGV controls with sinusoidal steering
    v_g = V_G_0  # 2.0 m/s
    phi_g = PHI_G_0
    # UAV controls with varying turn rate
    v_a = V_A_0  # 12.0 m/s
    omega_a = OMEGA_A_0
    
    return np.array([v_g, phi_g, v_a, omega_a])

def simulate_linearized_system(x0: np.ndarray, t: np.ndarray, L: float, control_func: Callable) -> np.ndarray:
    """Simulate linearized system dynamics"""
    DT = constants.DT # Match truth simulator timestep
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

def main():
    # Load real data
    ydata = np.loadtxt('data/ydata.csv', delimiter=',')
    Rtrue = np.loadtxt('data/Rtrue.csv', delimiter=',')
    Qtrue = np.loadtxt('data/Qtrue.csv', delimiter=',')
    tvec = np.loadtxt('data/tvec.csv', delimiter=',')
    
    # Initialize truth simulator just for initial state
    truth_sim = TruthSimulator(L=L, dt=DT)
    x0 = np.array([XI_G_0, ETA_G_0, THETA_G_0, XI_A_0, ETA_A_0, THETA_A_0])
    
    # Initialize Kalman filters with true noise parameters
    P0 = 10*np.diag([3.0, 3.0, 0.5, 3.0, 3.0, 0.5])**2
    
    # Use true process and measurement noise
    Q = Qtrue
    R = Rtrue
    
    # Initialize filters
    lkf = LinearizedKalmanFilter(x0.copy(), P0.copy(), Q.copy(), R.copy(), L)
    ekf = ExtendedKalmanFilter(x0.copy(), P0.copy(), Q.copy(), R.copy(), L)
    
    # Prepare state storage
    num_timesteps = len(tvec)
    lkf_states = np.zeros((num_timesteps, 6))
    ekf_states = np.zeros((num_timesteps, 6))
    lkf_covs = np.zeros((num_timesteps, 6, 6))
    ekf_covs = np.zeros((num_timesteps, 6, 6))
    
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
        
        # Prediction step
        if i < num_timesteps - 1:
            dt = tvec[i+1] - tvec[i]  # Use actual time differences
            lkf.predict(controls[i], dt)
            ekf.predict(controls[i], dt)
        
        # Update step (skip if NaN measurement)
        if not np.any(np.isnan(measurements[i])):
            lkf.update(measurements[i])
            ekf.update(measurements[i])
    
    # Plot results (modified to use real measurements)
    plot_estimation_results(tvec, None, lkf_states, ekf_states, 
                          lkf_covs, ekf_covs, measurements)
    
    # Plot individual filter performance
    plot_filter_performance(tvec, None, lkf_states, lkf_covs, 
                          measurements, R, "LKF")
    plot_filter_performance(tvec, None, ekf_states, ekf_covs, 
                          measurements, R, "EKF")
    
    true_states = None
    
    if true_states is not None:
        # Compute NEES values
        lkf_nees = compute_nees(true_states, lkf_states, lkf_covs)
        ekf_nees = compute_nees(true_states, ekf_states, ekf_covs)
        
        # Perform hypothesis tests
        lkf_results = perform_nees_hypothesis_test(lkf_nees)
        ekf_results = perform_nees_hypothesis_test(ekf_nees)
        
        # Print results
        print("\nLinearized Kalman Filter NEES Test Results:")
        print(f"Average NEES: {lkf_results['average_nees']:.2f}")
        print(f"Expected bounds: [{lkf_results['lower_bound']:.2f}, {lkf_results['upper_bound']:.2f}]")
        print(f"Percent in bounds: {lkf_results['percent_in_bounds']:.1f}%")
        print(f"Filter is {'consistent' if lkf_results['filter_consistent'] else 'inconsistent'}")
        
        print("\nExtended Kalman Filter NEES Test Results:")
        print(f"Average NEES: {ekf_results['average_nees']:.2f}")
        print(f"Expected bounds: [{ekf_results['lower_bound']:.2f}, {ekf_results['upper_bound']:.2f}]")
        print(f"Percent in bounds: {ekf_results['percent_in_bounds']:.1f}%")
        print(f"Filter is {'consistent' if ekf_results['filter_consistent'] else 'inconsistent'}")
    
    # Compute NIS values
    lkf_nis = compute_nis(measurements, lkf_states, lkf_covs, R)
    ekf_nis = compute_nis(measurements, ekf_states, ekf_covs, R)
    
    # Perform NIS hypothesis tests
    print("\nFilter Consistency Analysis:")
    print("\nLKF Results:")
    lkf_results = perform_nis_hypothesis_test(lkf_nis)
    print(f"Average NIS: {lkf_results['average_nis']:.2f}")
    print(f"Expected bounds: [{lkf_results['lower_bound']:.2f}, {lkf_results['upper_bound']:.2f}]")
    print(f"Percent in bounds: {lkf_results['percent_in_bounds']:.1f}%")
    print(f"Filter consistent: {lkf_results['filter_consistent']}")
    
    print("\nEKF Results:")
    ekf_results = perform_nis_hypothesis_test(ekf_nis)
    print(f"Average NIS: {ekf_results['average_nis']:.2f}")
    print(f"Expected bounds: [{ekf_results['lower_bound']:.2f}, {ekf_results['upper_bound']:.2f}]")
    print(f"Percent in bounds: {ekf_results['percent_in_bounds']:.1f}%")
    print(f"Filter consistent: {ekf_results['filter_consistent']}")

if __name__ == "__main__":
    main()
