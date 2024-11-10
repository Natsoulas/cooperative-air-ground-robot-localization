# Simulation Architecture

## Overview
The simulation implements a cooperative localization system between a UGV and UAV, as described in the system description document.

## Simulation Flow
1. **Initialization**
   - Set initial states for both vehicles
   - Configure simulation parameters
   - Initialize truth simulator

2. **Truth Simulation**
   - Propagate true states using RK4 integration
   - Apply control inputs
   - Handle process noise

3. **Measurement Generation**
   - Compute relative measurements between vehicles
   - Add measurement noise
   - Generate GPS measurements for UAV

4. **Visualization**
   - Plot vehicle trajectories
   - Display state histories
   - Show measurement data

## Key Components

### State Propagation
- Uses 4th order Runge-Kutta integration
- Handles non-linear vehicle dynamics
- Maintains angle normalization

### Control Inputs
- Constant control inputs for demonstration
- UGV: constant velocity and steering angle
- UAV: constant velocity and turn rate

### Measurement System
- Range measurements between vehicles
- Relative azimuth angles
- UAV GPS measurements

### Noise Models
- Additive white Gaussian noise (AWGN)
- Separate noise for process and measurements
- Configurable noise standard deviations

## Simulation Parameters
All simulation parameters are defined in `constants.py`, including:
- Vehicle physical parameters
- Initial conditions
- Control inputs
- Simulation time step and duration
