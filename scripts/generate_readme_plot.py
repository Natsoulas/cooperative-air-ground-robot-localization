# Copyright (c) 2024 Niko Natsoulas
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

import numpy as np
from src.truth import TruthSimulator
from src.utils.constants import *
from src.core.filter import LinearizedKalmanFilter, ExtendedKalmanFilter
from src.utils.noise import NoiseGenerator
from src.core.measurement import measurement_model

def control_input(t: float) -> np.ndarray:
    v_g = V_G_0
    phi_g = PHI_G_0 * np.cos(0.1 * t)
    v_a = V_A_0
    omega_a = OMEGA_A_0 * np.cos(0.05 * t)
    return np.array([v_g, phi_g, v_a, omega_a])

def generate_plot():
    import matplotlib.pyplot as plt
    
    # Run simulation
    truth_sim = TruthSimulator(L=L, dt=DT)
    x0 = np.array([XI_G_0, ETA_G_0, THETA_G_0, XI_A_0, ETA_A_0, THETA_A_0])
    t, true_states = truth_sim.simulate(x0, (0, T_FINAL), control_input)
    
    # Generate measurements
    noise_gen = NoiseGenerator(
        state_noise_std=np.array([0.5, 0.5, 0.2, 0.5, 0.5, 0.2]),
        meas_noise_std=np.array([0.3, 2.0, 0.3, 1.0, 1.0])
    )
    
    measurements = []
    for state in true_states:
        meas_noise = noise_gen.generate_measurement_noise()
        meas = measurement_model(state, meas_noise)
        measurements.append(meas)
    measurements = np.array(measurements)
    
    # Run filters
    P0 = np.diag([1.0] * 6)
    Q = np.diag([0.5, 0.5, 0.2, 0.5, 0.5, 0.2])**2
    R = np.diag([0.3, 2.0, 0.3, 1.0, 1.0])**2
    
    lkf = LinearizedKalmanFilter(x0.copy(), P0.copy(), Q.copy(), R.copy(), L)
    ekf = ExtendedKalmanFilter(x0.copy(), P0.copy(), Q.copy(), R.copy(), L)
    
    lkf_states = np.zeros_like(true_states)
    ekf_states = np.zeros_like(true_states)
    
    for i in range(len(t)):
        lkf_states[i] = lkf.x
        ekf_states[i] = ekf.x
        
        if i < len(t) - 1:
            controls = control_input(t[i])
            lkf.predict(controls, DT)
            ekf.predict(controls, DT)
        
        lkf.update(measurements[i])
        ekf.update(measurements[i])
    
    # Create plot
    plt.figure(figsize=(10, 6))
    plt.plot(true_states[:, 0], true_states[:, 1], 'k-', label='True UGV')
    plt.plot(true_states[:, 3], true_states[:, 4], 'k--', label='True UAV')
    plt.plot(lkf_states[:, 0], lkf_states[:, 1], 'b-', label='LKF UGV')
    plt.plot(lkf_states[:, 3], lkf_states[:, 4], 'b--', label='LKF UAV')
    plt.plot(ekf_states[:, 0], ekf_states[:, 1], 'r-', label='EKF UGV')
    plt.plot(ekf_states[:, 3], ekf_states[:, 4], 'r--', label='EKF UAV')
    plt.grid(True)
    plt.xlabel('East (m)')
    plt.ylabel('North (m)')
    plt.title('Vehicle Trajectories')
    plt.legend()
    plt.savefig('docs/images/trajectory_plot.png', dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    generate_plot()
