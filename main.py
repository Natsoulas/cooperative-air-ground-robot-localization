# Copyright (c) 2024 Niko Natsoulas
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from src.utils.constants import *
from src.truth import TruthSimulator
from src.utils.noise import NoiseGenerator
from src.core.measurement import measurement_model, measurement_jacobian
from src.core.dynamics import combined_dynamics, ugv_dynamics, uav_dynamics
from src.utils.plotting import plot_simulation_results, plot_estimation_results, plot_filter_differences, plot_filter_performance, plot_linearization_comparison
from src.core.filter import LinearizedKalmanFilter, ExtendedKalmanFilter, continuous_to_discrete, system_jacobian, input_jacobian, UnscentedKalmanFilter
from src.utils.analysis import perform_nees_hypothesis_test
from src.utils.plotting import compute_nees
from typing import Callable
import src.utils.constants as constants
from tuning import get_UKF_Q, get_UKF_R

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
    
    # Add linearized measurement computation
    linear_measurements = np.zeros((n_steps, 5))
    for i in range(n_steps):
        # Get measurement Jacobian at current state instead of nominal
        H = measurement_jacobian(linear_states[i])
        # Compute linearized measurement around current state
        delta_y = H @ (linear_states[i] - linear_states[i])  # Will be zero
        y_nominal = measurement_model(linear_states[i], np.zeros(5))
        linear_measurements[i] = y_nominal + delta_y
    
    return linear_states, linear_measurements

def main():
    # Get T_FINAL from constants and ensure minimum 400 timesteps
    T_FINAL = max(constants.T_FINAL, 400 * constants.DT)
    
    # Initialize truth simulator
    truth_sim = TruthSimulator(L=L, dt=DT)
    
    # Set initial state
    x0 = np.array([XI_G_0, ETA_G_0, THETA_G_0, XI_A_0, ETA_A_0, THETA_A_0])
    
    # Add perturbation to initial state for comparison
    perturbation = np.array([1.0, 1.0, 0.1, 1.0, 1.0, 0.1])
    x0_perturbed = x0 + perturbation
    
    # Run nonlinear simulation
    t, true_states = truth_sim.simulate(
        initial_state=x0_perturbed,
        t_span=(0, T_FINAL),
        control_func=control_input
    )
    
    # Run linearized simulation
    linear_states, linear_measurements = simulate_linearized_system(x0_perturbed, t, L, control_input)
    
    # Generate nonlinear measurements without noise
    nonlinear_measurements = np.array([
        measurement_model(state, np.zeros(5)) 
        for state in true_states
    ])
    
    # Plot linearization comparison
    plot_linearization_comparison(
        t, true_states, linear_states,
        nonlinear_measurements, linear_measurements
    )
    
    # Store control inputs
    controls = np.array([control_input(t_i) for t_i in t])
    
    # Initialize noise generator with more accurate values
    state_noise_std = np.array([
        0.3,   # xi_g noise
        0.3,   # eta_g noise
        0.15,  # theta_g noise (radians)
        0.3,   # xi_a noise
        0.3,   # eta_a noise
        0.15   # theta_a noise (radians)
    ])
    
    meas_noise_std = np.array([
        0.02,    # azimuth_g noise (radians)
        8.0,     # range noise (meters)
        0.02,    # azimuth_a noise (radians)
        6.0,     # xi_a GPS noise (meters)
        6.0      # eta_a GPS noise (meters)
    ])
    
    noise_gen = NoiseGenerator(state_noise_std, meas_noise_std)
    
    # Generate noisy measurements
    measurements = []
    for state in true_states:
        meas_noise = noise_gen.generate_measurement_noise()
        meas = measurement_model(state, meas_noise)
        measurements.append(meas)
    measurements = np.array(measurements)
    
    # Initialize Kalman filters with better angle uncertainties
    P0 = np.diag([
        3.0,   # xi_g position
        3.0,   # eta_g position
        0.5,   # theta_g heading - increased initial uncertainty
        3.0,   # xi_a position
        3.0,   # eta_a position
        0.5    # theta_a heading - increased initial uncertainty
    ])**2
    
    # Process noise with better heading tuning
    Q = np.diag([
        0.2,   # xi_g noise
        0.2,   # eta_g noise
        0.8,   # theta_g noise - increased for better consistency
        0.2,   # xi_a noise
        0.2,   # eta_a noise
        0.8    # theta_a noise - increased for better consistency
    ])**2
    
    # Measurement noise with more realistic angular uncertainty
    R = np.diag([
        0.05**2,  # azimuth_g noise (radians) - more realistic
        8.0**2,   # range noise (meters)
        0.05**2,  # azimuth_a noise (radians) - more realistic
        6.0**2,   # xi_a GPS noise (meters)
        6.0**2    # eta_a GPS noise (meters)
    ])
    
    lkf = LinearizedKalmanFilter(x0.copy(), P0.copy(), Q.copy(), R.copy(), L)
    ekf = ExtendedKalmanFilter(x0.copy(), P0.copy(), Q.copy(), R.copy(), L)
    
    # Run filters
    lkf_states = np.zeros_like(true_states)
    ekf_states = np.zeros_like(true_states)
    lkf_covs = np.zeros((len(t), 6, 6))
    ekf_covs = np.zeros((len(t), 6, 6))
    
    for i in range(len(t)):
        # Store current estimates
        lkf_states[i] = lkf.x
        ekf_states[i] = ekf.x
        lkf_covs[i] = lkf.P
        ekf_covs[i] = ekf.P
        
        # Prediction step
        if i < len(t) - 1:
            lkf.predict(controls[i], DT)
            ekf.predict(controls[i], DT)
        
        # Update step
        lkf.update(measurements[i])
        ekf.update(measurements[i])
    
    # Initialize UKF
    ukf = UnscentedKalmanFilter(x0.copy(), P0.copy(), get_UKF_Q(), get_UKF_R(), L)

    # Add UKF state storage
    ukf_states = np.zeros_like(true_states)
    ukf_covs = np.zeros((len(t), 6, 6))

    # In the main loop, add UKF prediction and update steps
    for i in range(len(t)):
        # Store current estimates
        ukf_states[i] = ukf.x
        ukf_covs[i] = ukf.P
        
        # Prediction step
        if i < len(t) - 1:
            ukf.predict(controls[i], DT)
        
        # Update step
        ukf.update(measurements[i])

    # Plot UKF results
    plot_filter_performance(t, true_states, ukf_states, ukf_covs, 
                           measurements, R, "UKF")
    
    # Plot results
    # Original truth and measurement plots
    plot_simulation_results(t, true_states, measurements, controls)
    
    # State estimation results plots
    plot_estimation_results(t, true_states, lkf_states, ekf_states, ukf_states,
                          lkf_covs, ekf_covs, ukf_covs, measurements)
    
    # Plot individual filter performance analysis:

    # LKF Performance
    plot_filter_performance(t, true_states, lkf_states, lkf_covs, 
                          measurements, R, "LKF")
    
    # EKF Performance
    plot_filter_performance(t, true_states, ekf_states, ekf_covs, 
                          measurements, R, "EKF")
    
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

if __name__ == "__main__":
    main()
