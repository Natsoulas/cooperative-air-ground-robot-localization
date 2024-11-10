# Cooperative Localization System Description

## 1. Introduction

Robust and reliable outdoor localization and inertial navigation remains challenging for autonomous robotic vehicles. Although Global Positioning System (GPS) devices are now ubiquitous, GPS may not operate reliably in certain locations (e.g. urban environments, forest canopies, mountain valleys, etc.) and can be subject to spoofing, jamming, or cyberattacks. 

With the expected deployment of multi-vehicle robot teams for many applications, one way to address this challenge is for robots to augment their existing localization solutions with peer-to-peer tracking information derived from other nearby robots. This approach is called cooperative localization.

## 2. Physical System

The system depicted in Figure 1 is a simplified 2D variation of a uncrewed air-ground robotic team performing cooperative navigation. The system consists of:

- A 4-wheeled unmanned ground vehicle (UGV) that moves along the ground
- An unmanned aerial vehicle (UAV) that flies a 2D path at a constant altitude above the ground vehicle

The UAV and UGV are controlled by independent inputs, but obtain relative measurements to one another through their encounter. The UGV cannot obtain reliable GPS measurements and effectively switches its receiver off during its encounter with the UAV, while the UAV maintains reliable GPS throughout the encounter.

Although the system is designed for decentralized cooperative localization (i.e. where the UAV and UGV can independently estimate their own and each other's states), a centralized base station can also be used to collect measurements from both vehicles to estimate their combined states and thus provide a 'gold standard' for assessing decentralized algorithms. This assignment will focus on the development of the centralized estimator only, using simple non-linear vehicle motion models.

## 3. Dynamical System

### UGV Motion Model
The UGV's motion is modeled kinematically here as a simple 4-wheeled steerable Dubin's car with:
- Front-rear wheel separation length L
- East position ξ_g and North position η_g in an inertial frame
- Heading angle θ_g
- Control inputs u_g = [v_g, φ_g]ᵀ (linear velocity [m/s] and steering angle [rad])

This leads to the non-linear equations of motion:

1. ξ̇_g = v_g cos(θ_g) + w̃_x_g
2. η̇_g = v_g sin(θ_g) + w̃_y_g
3. θ̇_g = (v_g/L) tan(φ_g) + w̃_ω_g

where w̃_g = [w̃_x_g, w̃_y_g, w̃_ω_g]ᵀ is the process noise on the UGV states.

### UAV Motion Model
The fixed-wing UAV motion is modeled kinematically as a simple Dubin's unicycle with:
- East position ξ_a and North position η_a in an inertial frame
- Heading angle θ_a
- Control inputs u_a = [v_a, ω_a]ᵀ (linear velocity [m/s] and angular rate [rad/s])

This leads to the non-linear equations of motion:

4. ξ̇_a = v_a cos(θ_a) + w̃_x_a
5. η̇_a = v_a sin(θ_a) + w̃_y_a
6. θ̇_a = ω_a + w̃_ω_a

where w̃_a = [w̃_x_a, w̃_y_a, w̃_ω_a]ᵀ is the process noise on the UAV states.

### Combined System
The combined system state, control inputs, and disturbance inputs are:
- x(t) = [ξ_g, η_g, θ_g, ξ_a, η_a, θ_a]ᵀ
- u(t) = [u_g, u_a]ᵀ
- w(t) = [w̃_g, w̃_a]ᵀ

### Sensing Model
The sensing model for this system is given by a combination of:
- Noisy ranges and azimuth angles of the UGV relative to the UAV
- Noisy azimuth angles of the UAV relative to the UGV
- Noisy UAV GPS measurements

y(t) = [arctan((η_a - η_g)/(ξ_a - ξ_g)) - θ_g, √((ξ_g - ξ_a)² + (η_g - η_a)²) - θ_a, arctan((η_g - η_a)/(ξ_g - ξ_a)) - θ_a, ξ_a, η_a]ᵀ + ṽ(t)

where ṽ(t) ∈ ℝ⁵ is the sensor error vector (which can be modeled by AWGN).

## 4. Nominal System Parameters

### Vehicle Parameters
- UGV wheel separation: L = 0.5 m
- UGV steering angle range: φ_g ∈ [-5π/12, 5π/12] rad
- UGV maximum speed: v_g,max = 3 m/s
- UAV turn rate limits: ω_g ∈ [-π/6, π/6] rad/s
- UAV velocity range: v_a ∈ [10, 20] m/s

### Initial Conditions
UGV:
- Starting position: (ξ_g, η_g) = (10, 0)
- Initial heading: θ_g = π/2 rad
- Steady maneuver: v_g = 2 m/s, φ = -π/18 rad

UAV:
- Starting position: (ξ_a, η_a) = (-60, 0)
- Initial heading: θ_a = -π/2 rad
- Steady maneuver: v_a = 12 m/s, ω_a = π/25 rad/s


