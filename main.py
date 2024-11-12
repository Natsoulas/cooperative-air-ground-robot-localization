import numpy as np
from src.utils.constants import *
from src.truth import TruthSimulator
from src.utils.noise import NoiseGenerator
from src.core.measurement import measurement_model
from src.utils.plotting import plot_simulation_results, plot_estimation_results, plot_filter_differences
from src.core.filter import LinearizedKalmanFilter, ExtendedKalmanFilter

def control_input(t: float) -> np.ndarray:
    """Generate control inputs for both vehicles"""
    # UGV controls with sinusoidal steering
    v_g = V_G_0  # 2.0 m/s
    phi_g = PHI_G_0 * np.cos(0.1 * t)  # Time-varying steering
    
    # UAV controls with varying turn rate
    v_a = V_A_0  # 12.0 m/s
    omega_a = OMEGA_A_0 * np.cos(0.05 * t)  # Time-varying turn rate
    
    return np.array([v_g, phi_g, v_a, omega_a])

def main():
    # Initialize truth simulator
    truth_sim = TruthSimulator(L=L, dt=DT)
    
    # Set initial state
    x0 = np.array([XI_G_0, ETA_G_0, THETA_G_0, XI_A_0, ETA_A_0, THETA_A_0])
    
    # Run truth simulation
    t, true_states = truth_sim.simulate(
        initial_state=x0,
        t_span=(0, T_FINAL),
        control_func=control_input
    )
    
    # Store control inputs
    controls = np.array([control_input(t_i) for t_i in t])
    
    # Initialize noise generator with higher noise values
    state_noise_std = np.array([
        0.5,  # xi_g noise
        0.5,  # eta_g noise
        0.2,  # theta_g noise (radians)
        0.5,  # xi_a noise
        0.5,  # eta_a noise
        0.2   # theta_a noise (radians)
    ])
    
    meas_noise_std = np.array([
        0.3,    # azimuth_g noise (radians)
        2.0,    # range noise (meters)
        0.3,    # azimuth_a noise (radians)
        1.0,    # xi_a GPS noise (meters)
        1.0     # eta_a GPS noise (meters)
    ])
    
    noise_gen = NoiseGenerator(state_noise_std, meas_noise_std)
    
    # Generate noisy measurements
    measurements = []
    for state in true_states:
        meas_noise = noise_gen.generate_measurement_noise()
        meas = measurement_model(state, meas_noise)
        measurements.append(meas)
    measurements = np.array(measurements)
    
    # Initialize Kalman filters with tuned parameters
    P0 = np.diag([
        0.5,  # xi_g position
        0.5,  # eta_g position
        0.1,  # theta_g heading (reduced uncertainty in angles)
        0.5,  # xi_a position
        0.5,  # eta_a position
        0.1   # theta_a heading
    ]) * COVARIANCE_INIT_SCALE
    
    # Adjust process noise covariance
    Q = np.diag([
        0.5,  # Increased position noise
        0.5,
        0.2,  # Increased angle noise
        0.5,
        0.5,
        0.2
    ])**2
    
    # Keep measurement noise as is
    R = np.diag(meas_noise_std**2)
    
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
    
    # Plot results
    # Original truth and measurement plots
    plot_simulation_results(t, true_states, measurements, controls)
    
    # Additional estimation results plots
    plot_estimation_results(t, true_states, lkf_states, ekf_states, 
                          lkf_covs, ekf_covs, measurements)
    
    # Plot filter differences analysis
    plot_filter_differences(t, true_states, lkf_states, ekf_states,
                          lkf_covs, ekf_covs)

if __name__ == "__main__":
    main()
