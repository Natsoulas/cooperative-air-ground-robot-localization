import numpy as np

# UGV Parameters
L = 0.5  # wheel separation length (m)
PHI_G_MAX = 5 * np.pi / 12  # max steering angle (rad)
V_G_MAX = 3.0  # max ground speed (m/s)

# UAV Parameters
OMEGA_A_MAX = np.pi / 6  # max turn rate (rad/s)
V_A_MIN = 10.0  # min airspeed (m/s)
V_A_MAX = 20.0  # max airspeed (m/s)

# Initial Conditions
# UGV
XI_G_0 = 10.0  # initial east position (m)
ETA_G_0 = 0.0  # initial north position (m)
THETA_G_0 = np.pi/2  # initial heading (rad) - pointing north
V_G_0 = 2.0  # initial velocity (m/s)
PHI_G_0 = -np.pi/18  # initial steering angle (rad)

# UAV
XI_A_0 = -60.0  # initial east position (m)
ETA_A_0 = 0.0  # initial north position (m)
THETA_A_0 = -np.pi/2  # initial heading (rad) - pointing south
V_A_0 = 12.0  # initial velocity (m/s)
OMEGA_A_0 = np.pi/25  # initial turn rate (rad/s)

# Simulation Parameters
DT = 0.01  # simulation time step (s)
T_FINAL = 30.0  # simulation duration (s)
