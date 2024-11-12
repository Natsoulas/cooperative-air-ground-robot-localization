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
- **LKF (Online)**: Linearizes around online-computed nominal trajectory
  * Nominal trajectory computed using actual controls
  * Linearization points independent of state estimates
  * Uses RK4 integration for accurate nominal propagation
  * Example from implementation:
    ```python
    # Compute Jacobians at new nominal point using actual controls
    F = system_jacobian(next_nominal, controls, self.L)
    H = measurement_jacobian(next_nominal)
    ```

- **EKF (Closed-Loop)**: Linearizes around current state estimate
  * Creates feedback loop where estimates affect future linearizations
  * Must recompute Jacobians at each step
  * Example from implementation:
    ```python
    # Compute current Jacobian
    F = system_jacobian(self.x, controls, self.L)
    ```

### State Representation
- **LKF**: 
  * x_nominal: Online nominal trajectory with sliding window
  * δx: Error state
  * x = x_nominal + δx
- **EKF**: Single state vector x

### Computational Efficiency
- **LKF**: Moderate efficiency (online nominal computation, sliding window of Jacobians)
- **EKF**: Similar computational cost (continuous Jacobian updates)

### Performance Characteristics
- **LKF**:
  * Maintains separation between nominal and error dynamics
  * Better numerical stability through error-state formulation
  * Linearization errors independent of estimates
  * Uses sliding window for memory efficiency
- **EKF**:
  * More adaptive to large deviations
  * Estimation errors affect future linearizations
  * Uses eigenvalue bounds for numerical stability
  * Direct state estimation without error separation

### Numerical Stability Features
- **LKF**:
  * Sliding window of nominal points and Jacobians
  * Joseph form for covariance updates
  * Angle normalization for both nominal and error states
- **EKF**:
  * Explicit covariance conditioning
  * SVD-based innovation computation
  * Eigenvalue bounds for numerical stability
  * Robust angle normalization

## Implementation Details
See the source code for detailed implementation:
- LKF Implementation: src/core/filter.py (LinearizedKalmanFilter class)
- EKF Implementation: src/core/filter.py (ExtendedKalmanFilter class)

## Filter Steps

### Linearized Kalman Filter
1. **Initialization**
   - Initialize nominal trajectory storage
   - Initialize error state to zero
   - Set up sliding window parameters

2. **Prediction**
   - Update nominal trajectory using RK4 integration
   - Compute new Jacobians at nominal point
   - Propagate error state and covariance
   - Maintain sliding window of nominal points

3. **Update**
   - Use latest nominal measurement Jacobian
   - Compute innovation using nominal state
   - Update error state and covariance
   - Maintain numerical stability through Joseph form

### Extended Kalman Filter
1. **Initialization**
   - Set initial state estimate
   - Initialize state covariance
   - Set numerical stability parameters

2. **Prediction**
   - Use multiple integration substeps
   - Compute Jacobians at current estimate
   - Predict state using RK4 integration
   - Ensure covariance validity

3. **Update**
   - Robust innovation computation using SVD
   - Update state and covariance with numerical safeguards
   - Handle potential numerical failures gracefully
   - Maintain covariance conditioning