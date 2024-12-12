# Running the Code

## Prerequisites
- Python 3.8+
- NumPy
- SciPy
- Matplotlib

## Installation
1. Clone the repository:
```bash
git clone https://github.com/your-username/cooperative-air-ground-robot-localization.git
cd cooperative-air-ground-robot-localization
```

2. Install dependencies:
```bash
pip install numpy scipy matplotlib
```

## Running the Simulation

### Main Simulation
Run the main simulation:
```bash
python main.py
```

This will:
1. Initialize the truth simulator with the UGV and UAV parameters
2. Run the simulation with:
   - UGV moving at 2 m/s with -π/18 rad steering angle
   - UAV moving at 12 m/s with π/25 rad/s turn rate
3. Generate noisy measurements with:
   - Position noise: 0.3m standard deviation
   - Heading noise: 0.15 rad standard deviation
   - Range noise: 8.0m standard deviation
   - Azimuth noise: 0.05 rad standard deviation
   - GPS noise: 6.0m standard deviation
4. Run three Kalman filters (LKF, EKF, UKF)
5. Display plots showing:
   - Vehicle trajectories
   - State estimates
   - Filter performance metrics
   - NEES/NIS consistency analysis

### Monte Carlo Testing
Run individual filter Monte Carlo tests:
```bash
python TMT_LKF.py  # Linearized Kalman Filter
python TMT_EKF.py  # Extended Kalman Filter
python TMT_UKF.py  # Unscented Kalman Filter
```

Each Monte Carlo test will:
1. Run 10 simulation trials
2. Generate statistical performance metrics
3. Plot filter consistency results
4. Display NEES/NIS test results

### Real Data Testing
Test the filters on real data:
```bash
python estimate_from_real_data.py
```

This will:
1. Load measurement data from `data/` directory
2. Run all three filters on the dataset
3. Compare estimation performance
4. Generate analysis plots
