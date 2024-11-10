# Software Architecture

## Project Structuresrc/
```
src/
├── core/
│   ├── dynamics.py        # Vehicle dynamics models
│   └── measurement.py     # Measurement models
├── utils/
│   ├── constants.py       # System parameters and constants
│   ├── noise.py          # Noise generation utilities
│   └── plotting.py       # Visualization functions
└── truth.py              # Truth state simulator
```


## Core Components

### Dynamics Module
- Implements UGV and UAV dynamics equations
- Provides combined system dynamics
- Handles process noise integration

### Measurement Module
- Implements sensor measurement models
- Handles measurement noise integration
- Computes relative measurements between vehicles

### Truth Simulator
- Simulates true vehicle states using RK4 integration
- Manages state propagation
- Handles angle normalization

## Utility Components

### Constants
- Defines system parameters
- Stores initial conditions
- Sets simulation parameters

### Noise Generator
- Generates process and measurement noise
- Manages noise standard deviations

### Plotting
- Creates visualization plots
- Displays simulation results
- Manages figure layout
