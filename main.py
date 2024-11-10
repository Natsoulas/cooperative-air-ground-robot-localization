import numpy as np
from src.utils.constants import *
from src.truth import TruthSimulator
from src.utils.noise import NoiseGenerator
from src.core.measurement import measurement_model
from src.utils.plotting import plot_simulation_results

def control_input(t: float) -> np.ndarray:
    """Generate control inputs for both vehicles"""
    # UGV controls
    v_g = V_G_0  # 2.0 m/s
    phi_g = PHI_G_0  # -π/18 rad
    
    # UAV controls
    v_a = V_A_0  # 12.0 m/s
    omega_a = OMEGA_A_0  # π/25 rad/s
    
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
    
    # Initialize noise generator (example standard deviations)
    state_noise_std = 0.1 * np.ones(6)
    meas_noise_std = 0.1 * np.ones(5)
    noise_gen = NoiseGenerator(state_noise_std, meas_noise_std)
    
    # Generate noisy measurements
    measurements = []
    for state in true_states:
        meas_noise = noise_gen.generate_measurement_noise()
        meas = measurement_model(state, meas_noise)
        measurements.append(meas)
    
    measurements = np.array(measurements)
    
    # Plot results
    plot_simulation_results(t, true_states, measurements, controls)
    
if __name__ == "__main__":
    main()
