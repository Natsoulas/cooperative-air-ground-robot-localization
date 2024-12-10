import numpy as np
from scipy.stats import chi2
from src.utils.constants import *
from src.truth import TruthSimulator
from src.utils.noise import NoiseGenerator
from src.core.measurement import measurement_model
from src.core.filter import UnscentedKalmanFilter
from src.utils.analysis import perform_nees_hypothesis_test, perform_nis_hypothesis_test
from src.utils.plotting import compute_nees, compute_nis, plot_filter_performance, plot_monte_carlo_results, plot_single_simulation_results
import matplotlib.pyplot as plt
from typing import List, Dict
from tuning import get_P0, get_UKF_Q, get_UKF_R, get_state_noise_std, get_meas_noise_std

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
    
    # Initialize storage for UKF NEES and NIS
    ukf_nees_values = np.zeros((n_runs, n_steps))
    ukf_nis_values = np.zeros((n_runs, n_steps))
    
    # Initial state
    x0 = np.array([XI_G_0, ETA_G_0, THETA_G_0, XI_A_0, ETA_A_0, THETA_A_0])
    
    # Initialize noise generator with more accurate values (matching EKF)
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
    
    # Filter parameters (matched to EKF)
    P0 = get_P0(scale=10.0)
    
    # Use true process and measurement noise if available
    try:
        Q = np.loadtxt('data/Qtrue.csv', delimiter=',')
        R = np.loadtxt('data/Rtrue.csv', delimiter=',')
    except FileNotFoundError:
        Q = np.diag(state_noise_std**2)
        R = np.diag(meas_noise_std**2)
    
    # Initialize truth simulator
    truth_sim = TruthSimulator(L=L, dt=DT)
    
    # Adjust noise generator to match filter parameters exactly
    noise_gen = NoiseGenerator(
        state_noise_std=np.sqrt(np.diag(Q)),
        meas_noise_std=np.sqrt(np.diag(R))
    )
    
    # Store first run for plotting
    first_run_data = None
    
    # Monte Carlo simulation loop
    for i in range(n_runs):
        # Add small perturbation to initial state (matching LKF/EKF)
        perturbation = np.random.multivariate_normal(
            np.zeros(6), 
            0.01 * P0
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
        
        # Initialize UKF
        ukf = UnscentedKalmanFilter(x0.copy(), P0.copy(), Q.copy(), R.copy(), L)
        
        # Storage for filter states and covariances
        ukf_states = np.zeros_like(true_states)
        ukf_covs = np.zeros((len(t), 6, 6))
        
        # Run UKF
        for j in range(len(t)):
            # Store current estimates before prediction/update
            ukf_states[j] = ukf.x.copy()
            ukf_covs[j] = ukf.P.copy()
            
            # Prediction step
            if j < len(t) - 1:
                controls = control_input(t[j])
                ukf.predict(controls, DT)
            
            # Update step
            ukf.update(measurements[j])
            
            # Normalize angles after update
            ukf.x[2] = np.mod(ukf.x[2] + np.pi, 2 * np.pi) - np.pi
            ukf.x[5] = np.mod(ukf.x[5] + np.pi, 2 * np.pi) - np.pi
        
        # Compute NEES and NIS values
        ukf_nees_values[i] = compute_nees(true_states, ukf_states, ukf_covs)
        ukf_nis_values[i] = compute_nis(measurements, ukf_states, ukf_covs, R)
        
        # Store first run data
        if i == 0:
            first_run_data = {
                't': t,
                'true_states': true_states.copy(),
                'measurements': measurements.copy(),
                'ukf_states': ukf_states.copy(),
                'ukf_covs': ukf_covs.copy()
            }
    
    return {
        'first_run_data': first_run_data,
        'avg_nees': np.nanmean(ukf_nees_values, axis=0),
        'avg_nis': np.nanmean(ukf_nis_values, axis=0),
        't': t,
        'ukf_nees_values': ukf_nees_values,
        'ukf_nis_values': ukf_nis_values
    }

if __name__ == "__main__":
    # Run Monte Carlo simulation
    n_runs = 10
    t_span = (0, T_FINAL)
    
    print("Running Monte Carlo simulations...")
    results = run_monte_carlo_simulation(n_runs, t_span)
    
    # Plot single simulation results first
    first_run = results['first_run_data']
    plot_single_simulation_results(
        first_run['t'],
        first_run['true_states'],
        first_run['ukf_states'],
        first_run['measurements'],
        first_run['ukf_covs']
    )
    
    # Plot results for UKF
    print("\nPlotting UKF results...")
    plot_monte_carlo_results(results=results, filter_type="UKF", alpha=0.05)
    
    # Perform hypothesis tests on average values for UKF
    print("\nPerforming hypothesis tests for UKF...")
    
    # NEES test
    nees_results = perform_nees_hypothesis_test(np.nanmean(results['ukf_nees_values'], axis=0))
    print("\nUKF NEES Test Results:")
    print(f"Average NEES: {nees_results['average_nees']:.2f}")
    print(f"Expected bounds: [{nees_results['lower_bound']:.2f}, {nees_results['upper_bound']:.2f}]")
    print(f"Percent in bounds: {nees_results['percent_in_bounds']:.1f}%")
    print(f"Filter is {'consistent' if nees_results['filter_consistent'] else 'inconsistent'}")
    
    # NIS test
    nis_results = perform_nis_hypothesis_test(np.nanmean(results['ukf_nis_values'], axis=0))
    print("\nUKF NIS Test Results:")
    print(f"Average NIS: {nis_results['average_nis']:.2f}")
    print(f"Expected bounds: [{nis_results['lower_bound']:.2f}, {nis_results['upper_bound']:.2f}]")
    print(f"Percent in bounds: {nis_results['percent_in_bounds']:.1f}%")
    print(f"Filter is {'consistent' if nis_results['filter_consistent'] else 'inconsistent'}")