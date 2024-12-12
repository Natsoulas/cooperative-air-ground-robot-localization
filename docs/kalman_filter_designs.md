# Kalman Filter Designs

This document describes three Kalman filter implementations for UGV-UAV cooperative localization:
1. Linearized Kalman Filter (LKF)
2. Extended Kalman Filter (EKF) 
3. Unscented Kalman Filter (UKF)

## Common Elements
- State vector: x = [ξ_g, η_g, θ_g, ξ_a, η_a, θ_a]ᵀ
- Process noise covariance: Q
- Measurement noise covariance: R 
- Vehicle parameter: L (wheel separation)

## Key Implementation Differences

### Linearization/Approximation Strategy
- **LKF**: Linearizes around online-computed nominal trajectory
  * Nominal trajectory computed using actual controls
  * Linearization points independent of state estimates
  * Uses RK4 integration for accurate nominal propagation
  * Example from implementation:
    ```python
    # Compute Jacobians at new nominal point using actual controls
    F = system_jacobian(self.x_nominal, controls, self.L)
    H = measurement_jacobian(self.x_nominal)
    ```

- **EKF**: Linearizes around current state estimate
  * Creates feedback loop where estimates affect future linearizations
  * Must recompute Jacobians at each step
  * Example from implementation:
    ```python
    # Compute current Jacobian
    F = system_jacobian(self.x, controls, self.L)
    ```

- **UKF**: Uses sigma points for nonlinear transformation
  * No explicit linearization required
  * Captures higher-order statistical moments
  * Uses carefully selected sigma points for state propagation
  * Example from implementation:
    ```python
    # Generate and propagate sigma points
    sigma_points = self.generate_sigma_points()
    propagated_points = np.zeros_like(sigma_points)
    for i in range(len(sigma_points)):
        state = sigma_points[i]
        # RK4 integration of each sigma point
    ```

### State Representation
- **LKF**: 
  * x_nominal: Online nominal trajectory
  * δx: Error state
  * x = x_nominal + δx
- **EKF**: Single state vector x
- **UKF**: Set of sigma points representing state distribution

### Computational Efficiency
- **LKF**: Moderate efficiency (online nominal computation)
- **EKF**: Similar computational cost (continuous Jacobian updates)
- **UKF**: Higher computational cost (multiple sigma point propagations)

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
- **UKF**:
  * Better handles strong nonlinearities
  * Captures higher-order statistical moments
  * No explicit Jacobian computation needed
  * More robust to initialization errors

### Numerical Stability Features
- **LKF**:
  * Joseph form for covariance updates
  * Angle normalization for both nominal and error states
  * Explicit covariance validity checks
- **EKF**:
  * Explicit covariance conditioning
  * SVD-based innovation computation
  * Eigenvalue bounds for numerical stability
  * Robust angle normalization
- **UKF**:
  * Careful sigma point generation and weighting
  * Angle-aware mean computation
  * Robust covariance reconstruction
  * Special handling for angular measurements

## Implementation Details
See the source code for detailed implementation:
- LKF Implementation: src/core/filter.py (LinearizedKalmanFilter class)
- EKF Implementation: src/core/filter.py (ExtendedKalmanFilter class)
- UKF Implementation: src/core/filter.py (UnscentedKalmanFilter class)

## Filter Steps

### Linearized Kalman Filter
1. **Initialization**
   - Initialize nominal trajectory storage
   - Initialize error state to zero
   - Set up initial covariance

2. **Prediction**
   - Update nominal trajectory using RK4 integration
   - Compute new Jacobians at nominal point
   - Propagate error state and covariance
   - Handle angle normalization

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
   - Robust innovation computation
   - Update state and covariance with numerical safeguards
   - Handle potential numerical failures gracefully
   - Maintain covariance conditioning

### Unscented Kalman Filter
1. **Initialization**
   - Set initial state and covariance
   - Configure sigma point parameters:
     * α = 0.15 (spread of sigma points)
     * β = 2.0 (optimal for Gaussian)
     * κ = -1.0 (secondary spread parameter)
   - Calculate weights for mean and covariance reconstruction

2. **Prediction**
   - Generate sigma points using current state and covariance
   - Propagate each sigma point through RK4 integration:
     * Use combined vehicle dynamics
     * Apply angle normalization per point
     * Handle process noise integration
   - Reconstruct predicted mean and covariance:
     * Special circular mean for angular states
     * Angle-aware covariance computation
     * Add process noise scaled by timestep

3. **Update**
   - Generate new sigma points from predicted state
   - Transform points through measurement model
   - Compute predicted measurement statistics:
     * Circular mean for azimuth measurements
     * Regular mean for range and GPS
   - Calculate cross-correlation and innovation
   - Update state and covariance with angle awareness
   - Maintain numerical stability through conditioning