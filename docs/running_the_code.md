# Running the Code

## Prerequisites
- Python 3.8+
- NumPy
- Matplotlib

## Installation
1. Clone the repository:
```bash
git clone https://github.com/your-username/cooperative-air-ground-robot-localization.git
cd cooperative-air-ground-robot-localization
```

2. Install dependencies:
```bash
pip install numpy matplotlib
```

## Running the Simulation
The main simulation can be run using:
```bash
python main.py
```

This will:
1. Initialize the truth simulator with the UGV and UAV parameters
2. Run the simulation for 30 seconds with:
   - UGV moving at 2 m/s with -π/18 rad steering angle
   - UAV moving at 12 m/s with π/25 rad/s turn rate
3. Generate noisy measurements
4. Display plots showing:
   - 2D trajectory of both vehicles
   - Range between vehicles
   - UGV states
   - UAV states
   - Relative angles
   - Control inputs
