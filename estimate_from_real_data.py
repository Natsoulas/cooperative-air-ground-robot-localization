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
from src.utils.plotting import plot_simulation_results, plot_estimation_results, plot_filter_differences, plot_filter_performance, plot_linearization_comparison, plot_uncertainty_bounds
from src.core.filter import LinearizedKalmanFilter, ExtendedKalmanFilter, UnscentedKalmanFilter, continuous_to_discrete, system_jacobian, input_jacobian
from src.utils.analysis import perform_nees_hypothesis_test, perform_nis_hypothesis_test
from src.utils.plotting import compute_nees, compute_nis
from typing import Callable
import src.utils.constants as constants
from tuning import (
    get_P0, get_LKF_Q, get_LKF_R, 
    get_EKF_Q, get_EKF_R,
    get_UKF_Q, get_UKF_R,
    get_state_noise_std, get_meas_noise_std
)

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
    
    # Initialize noise generator
    state_noise_std = get_state_noise_std()
    meas_noise_std = get_meas_noise_std()
    
    # Initialize Kalman filters with true noise parameters
    P0 = get_P0()
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
    
    # Plot results (modified to use real measurements)
    plot_estimation_results(tvec, None, lkf_states, ekf_states, ukf_states,
                          lkf_covs, ekf_covs, ukf_covs, measurements)
    
    # Plot uncertainty bounds
    plot_uncertainty_bounds(tvec, lkf_covs, ekf_covs, ukf_covs)
    
    # Plot individual filter performance
    plot_filter_performance(tvec, None, lkf_states, lkf_covs, 
                          measurements, R, "LKF")
    plot_filter_performance(tvec, None, ekf_states, ekf_covs, 
                          measurements, R, "EKF")
    plot_filter_performance(tvec, None, ukf_states, ukf_covs, 
                          measurements, R, "UKF")
    
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
    
    # Compute NIS values for UKF
    ukf_nis = compute_nis(measurements, ukf_states, ukf_covs, R)
    
    # Perform NIS hypothesis test for UKF
    print("\nUKF Results:")
    ukf_results = perform_nis_hypothesis_test(ukf_nis)
    print(f"Average NIS: {ukf_results['average_nis']:.2f}")
    print(f"Expected bounds: [{ukf_results['lower_bound']:.2f}, {ukf_results['upper_bound']:.2f}]")
    print(f"Percent in bounds: {ukf_results['percent_in_bounds']:.1f}%")
    print(f"Filter consistent: {ukf_results['filter_consistent']}")

if __name__ == "__main__":
    main()
