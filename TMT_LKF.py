import numpy as np
from scipy.stats import chi2
from src.utils.constants import *
from src.truth import TruthSimulator
from src.utils.noise import NoiseGenerator
from src.core.measurement import measurement_model
from src.core.filter import LinearizedKalmanFilter
from src.utils.analysis import perform_nees_hypothesis_test, perform_nis_hypothesis_test
from src.utils.plotting import compute_nees, compute_nis, plot_filter_performance, plot_monte_carlo_results
import matplotlib.pyplot as plt
from typing import List, Dict

def control_input(t: float) -> np.ndarray:
    """Generate control inputs for both vehicles"""
    # UGV controls with sinusoidal steering
    v_g = V_G_0  # 2.0 m/s
    phi_g = PHI_G_0
    # UAV controls with varying turn rate
    v_a = V_A_0  # 12.0 m/s
    omega_a = OMEGA_A_0
    
    return np.array([v_g, phi_g, v_a, omega_a])

def run_monte_carlo_simulation(n_runs: int, t_span: tuple) -> Dict:
    """Run Monte Carlo simulations for filter testing"""
    # Time vector
    t = np.arange(t_span[0], t_span[1], DT)
    n_steps = len(t)
    
    # Initialize storage for LKF NEES and NIS
    lkf_nees_values = np.zeros((n_runs, n_steps))
    lkf_nis_values = np.zeros((n_runs, n_steps))
    
    # Initial state with perturbation
    x0 = np.array([XI_G_0, ETA_G_0, THETA_G_0, XI_A_0, ETA_A_0, THETA_A_0])
    
    # Initialize noise generator with more accurate values (from main.py)
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
    
    # Filter parameters (matched to estimate_from_real_data.py)
    P0 = np.diag([
        3.0,   # xi_g position
        3.0,   # eta_g position
        0.5,   # theta_g heading
        3.0,   # xi_a position
        3.0,   # eta_a position
        0.5    # theta_a heading
    ])**2
    
    Q = np.diag([
        0.25,  # xi_g noise
        0.25,  # eta_g noise
        0.9,   # theta_g noise
        0.25,  # xi_a noise
        0.25,  # eta_a noise
        0.9    # theta_a noise
    ])**2
    
    R = np.diag([
        0.05**2,  # azimuth_g noise
        8.0**2,   # range noise
        0.05**2,  # azimuth_a noise
        6.0**2,   # xi_a GPS noise
        6.0**2    # eta_a GPS noise
    ])
    
    # Initialize truth simulator
    truth_sim = TruthSimulator(L=L, dt=DT)
    
    # Adjust noise generator to match filter parameters
    noise_gen = NoiseGenerator(
        state_noise_std=np.sqrt(np.diag(Q)),  # Match filter's Q exactly
        meas_noise_std=np.sqrt(np.diag(R))    # Match filter's R exactly
    )
    
    # Store first run for plotting
    first_run_data = None
    
    # Monte Carlo simulation loop
    for i in range(n_runs):
        # Add small perturbation to initial state
        perturbation = np.random.multivariate_normal(
            np.zeros(6), 
            0.01 * P0  # Small perturbation
        )
        x0_perturbed = x0 + perturbation
        
        # Simulate true trajectory
        t, true_states = truth_sim.simulate(
            initial_state=x0_perturbed,
            t_span=t_span,
            control_func=control_input
        )
        
        # Generate measurements
        measurements = []
        for state in true_states:
            meas_noise = noise_gen.generate_measurement_noise()
            meas = measurement_model(state, meas_noise)
            measurements.append(meas)
        measurements = np.array(measurements)
        
        # Initialize LKF
        lkf = LinearizedKalmanFilter(x0.copy(), P0.copy(), Q.copy(), R.copy(), L)
        
        # Storage for filter states and covariances
        lkf_states = np.zeros_like(true_states)
        lkf_covs = np.zeros((len(t), 6, 6))
        
        # Run LKF
        for j in range(len(t)):
            # Store current estimates
            lkf_states[j] = lkf.x
            lkf_covs[j] = lkf.P
            
            # Prediction step
            if j < len(t) - 1:
                controls = control_input(t[j])
                lkf.predict(controls, DT)
            
            # Update step
            lkf.update(measurements[j])
        
        # Compute NEES and NIS values for LKF
        lkf_nees_values[i] = compute_nees(true_states, lkf_states, lkf_covs)
        lkf_nis_values[i] = compute_nis(measurements, lkf_states, lkf_covs, R)
        
        # Store first run data for plotting
        if i == 0:
            first_run_data = {
                't': t,
                'true_states': true_states,
                'measurements': measurements,
                'lkf_states': lkf_states,
                'lkf_covs': lkf_covs
            }
    
    # Compute average NEES and NIS across all runs
    avg_nees = np.nanmean(lkf_nees_values, axis=0)
    avg_nis = np.nanmean(lkf_nis_values, axis=0)
    
    return {
        'first_run_data': first_run_data,
        'avg_nees': avg_nees,
        'avg_nis': avg_nis,
        't': t,
        'lkf_nees_values': lkf_nees_values,
        'lkf_nis_values': lkf_nis_values,
    }

if __name__ == "__main__":
    # Run Monte Carlo simulation
    n_runs = 100
    t_span = (0, T_FINAL)
    
    print("Running Monte Carlo simulations...")
    results = run_monte_carlo_simulation(n_runs, t_span)
    
    # Plot results for LKF
    print("\nPlotting LKF results...")
    plot_monte_carlo_results(results=results, filter_type="LKF", alpha=0.05)
    
    # Perform hypothesis tests on average values for LKF
    print("\nPerforming hypothesis tests for LKF...")
    
    # NEES test
    nees_results = perform_nees_hypothesis_test(np.nanmean(results['lkf_nees_values'], axis=0))
    print("\nLKF NEES Test Results:")
    print(f"Average NEES: {nees_results['average_nees']:.2f}")
    print(f"Expected bounds: [{nees_results['lower_bound']:.2f}, {nees_results['upper_bound']:.2f}]")
    print(f"Percent in bounds: {nees_results['percent_in_bounds']:.1f}%")
    print(f"Filter is {'consistent' if nees_results['filter_consistent'] else 'inconsistent'}")
    
    # NIS test
    nis_results = perform_nis_hypothesis_test(np.nanmean(results['lkf_nis_values'], axis=0))
    print("\nLKF NIS Test Results:")
    print(f"Average NIS: {nis_results['average_nis']:.2f}")
    print(f"Expected bounds: [{nis_results['lower_bound']:.2f}, {nis_results['upper_bound']:.2f}]")
    print(f"Percent in bounds: {nis_results['percent_in_bounds']:.1f}%")
    print(f"Filter is {'consistent' if nis_results['filter_consistent'] else 'inconsistent'}")