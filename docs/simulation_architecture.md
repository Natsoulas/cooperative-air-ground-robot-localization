# Simulation Architecture

## Overview
The simulation implements a cooperative localization system between a UGV and UAV, as described in the system description document.

## Simulation Modes

### 1. Main Simulation (main.py)
- Initializes truth simulator with vehicle parameters
- Runs nonlinear and linearized simulations for comparison
- Generates measurements with realistic noise parameters:
  * Position noise: 0.3m std
  * Heading noise: 0.15 rad std
  * Range noise: 8.0m std
  * Azimuth noise: 0.05 rad std
  * GPS noise: 6.0m std
- Runs all three filters (LKF, EKF, UKF) simultaneously
- Performs consistency analysis (NEES/NIS tests)
- Generates comprehensive performance plots

### 2. Monte Carlo Testing (TMT_*.py)
- Runs multiple simulation trials (default: 10 runs)
- Uses consistent noise parameters across all trials
- Adds small random perturbations to initial states
- Generates statistical performance metrics
- Performs hypothesis testing for filter consistency
- Creates individual and aggregate performance plots
- Separate scripts for each filter type:
  * TMT_LKF.py: Linearized Kalman Filter
  * TMT_EKF.py: Extended Kalman Filter
  * TMT_UKF.py: Unscented Kalman Filter

### 3. Real Data Testing (estimate_from_real_data.py)
- Loads measurement data from data/ directory
- Uses true noise parameters when available
- Runs all three filters on the same dataset
- Compares estimation performance
- Generates analysis plots

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
