# Copyright (c) 2024 Niko Natsoulas
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from typing import Tuple, Callable
from src.core.dynamics import combined_dynamics

class TruthSimulator:
    def __init__(self, L: float, dt: float):
        self.L = L
        self.dt = dt
        
    def simulate(self, 
                initial_state: np.ndarray,
                t_span: Tuple[float, float],
                control_func: Callable[[float], np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate the true system using RK4 integration
        """
        # Setup time vector
        t = np.arange(t_span[0], t_span[1], self.dt)
        n_steps = len(t)
        
        # Initialize state history
        states = np.zeros((n_steps, len(initial_state)))
        states[0] = initial_state
        
        # RK4 integration
        for i in range(n_steps-1):
            current_t = t[i]
            current_state = states[i]
            controls = control_func(current_t)
            noise = np.zeros(6)
            
            # RK4 steps
            k1 = combined_dynamics(current_state, controls, noise, self.L)
            k2 = combined_dynamics(current_state + self.dt/2 * k1, controls, noise, self.L)
            k3 = combined_dynamics(current_state + self.dt/2 * k2, controls, noise, self.L)
            k4 = combined_dynamics(current_state + self.dt * k3, controls, noise, self.L)
            
            # Update state
            states[i+1] = current_state + self.dt/6 * (k1 + 2*k2 + 2*k3 + k4)
            
            # Normalize angles to [-π, π]
            states[i+1, 2] = np.mod(states[i+1, 2] + np.pi, 2*np.pi) - np.pi  # theta_g
            states[i+1, 5] = np.mod(states[i+1, 5] + np.pi, 2*np.pi) - np.pi  # theta_a
        
        return t, states
