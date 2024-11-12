# Kalman Filter Designs

This document describes two Kalman filter implementations for UGV-UAV cooperative localization:
1. Linearized Kalman Filter (LKF)
2. Extended Kalman Filter (EKF)

## Common Elements
- State vector: x = [ξ_g, η_g, θ_g, ξ_a, η_a, θ_a]ᵀ
- Process noise covariance: Q
- Measurement noise covariance: R
- Vehicle parameter: L (wheel separation)

## Key Implementation Differences

### Linearization Strategy
- **LKF (Open-Loop)**: Linearizes around a pre-defined nominal trajectory
  * Nominal trajectory satisfies nonlinear system ODEs
  * Linearization points independent of state estimates
  * Can be computed offline or online without feedback from estimator
  * Example from implementation:
    ```python
    # Pre-compute nominal trajectory Jacobian (should remain fixed)
    self.F_nominal = system_jacobian(self.x_nominal, self.nominal_controls, self.L)
    self.H_nominal = measurement_jacobian(self.x_nominal)
    ```

- **EKF (Closed-Loop)**: Linearizes around current state estimate
  * Creates feedback loop where estimates affect future linearizations
  * Must recompute Jacobians at each step
  * Example from implementation:
    ```python
    # Compute Jacobian at current state
    F = system_jacobian(self.x, controls, self.L)
    ```

### State Representation
- **LKF**: 
  * x_nominal: Nominal trajectory
  * δx: Error state
  * x = x_nominal + δx
- **EKF**: Single state vector x

### Computational Efficiency
- **LKF**: More efficient (fixed Jacobians)
- **EKF**: More computationally intensive (continuous Jacobian updates)

### Performance Characteristics
- **LKF**:
  * More stable with good nominal trajectory
  * Better for small deviations
  * Linearization errors independent of estimates
  * Can adjust nominal trajectory without affecting stability
- **EKF**:
  * More adaptive to large deviations
  * Estimation errors affect future linearizations
  * Potential for error coupling and divergence
  * Automatically adapts but may suffer from estimate-dependent linearization errors

### Error Propagation
- **LKF**: Linearization errors are independent of estimation errors
- **EKF**: Linearization and estimation errors are coupled, potentially leading to divergence

## Implementation Details
See the source code for detailed implementation:
- LKF Implementation: src/core/filter.py (LinearizedKalmanFilter class)
- EKF Implementation: src/core/filter.py (ExtendedKalmanFilter class)

## Filter Steps

### Linearized Kalman Filter
1. **Initialization**
   - Set initial nominal trajectory
   - Initialize error state to zero
   - Pre-compute system and measurement Jacobians

2. **Prediction**
   - Use fixed nominal Jacobian for state transition
   - Propagate nominal trajectory with nominal controls
   - Propagate error state and covariance

3. **Update**
   - Use fixed nominal measurement Jacobian
   - Compute innovation using nominal state
   - Update error state and covariance

### Extended Kalman Filter
1. **Initialization**
   - Set initial state estimate
   - Initialize state covariance

2. **Prediction**
   - Compute current Jacobian
   - Predict state using nonlinear model
   - Propagate covariance

3. **Update**
   - Compute current measurement Jacobian
   - Update state and covariance using nonlinear measurement model