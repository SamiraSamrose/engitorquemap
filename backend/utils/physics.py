#backend/utils/physics.py (Physics Calculations)
"""
Physics utility functions for energy calculations
"""
import numpy as np
from config import settings


def calculate_braking_power(speed: float, brake_pressure_f: float, brake_pressure_r: float, deceleration: float) -> float:
    """
    Calculate braking power (energy dissipated)
    
    Args:
        speed: vehicle speed (m/s)
        brake_pressure_f: front brake pressure (bar)
        brake_pressure_r: rear brake pressure (bar)
        deceleration: deceleration rate (m/s^2)
    
    Returns:
        Braking power in Watts (negative value)
    """
    # Force = mass * deceleration
    braking_force = settings.VEHICLE_MASS * deceleration * 9.81  # Convert g to m/s^2
    
    # Power = Force * velocity
    braking_power = braking_force * speed
    
    # Scale by brake pressure (normalized)
    total_brake_pressure = brake_pressure_f + brake_pressure_r
    if total_brake_pressure > 0:
        pressure_factor = total_brake_pressure / 100.0  # Normalize to 0-1
        braking_power *= pressure_factor
    
    return braking_power


def calculate_acceleration_power(speed: float, throttle: float, rpm: float, gear: int, acceleration: float) -> float:
    """
    Calculate acceleration power (energy input)
    
    Args:
        speed: vehicle speed (m/s)
        throttle: throttle position (0-100%)
        rpm: engine RPM
        gear: current gear
        acceleration: acceleration rate (m/s^2)
    
    Returns:
        Acceleration power in Watts
    """
    # Simplified engine power model
    # Assume peak power around 6000 RPM
    rpm_factor = min(1.0, rpm / 6000.0) if rpm > 0 else 0.0
    throttle_factor = throttle / 100.0
    
    # Estimate engine power output
    max_power = 300000  # 300 kW peak power (example)
    engine_power = max_power * rpm_factor * throttle_factor
    
    # Gear efficiency
    gear_efficiency = 0.85 + (gear * 0.02)  # Higher gears slightly more efficient
    gear_efficiency = min(0.95, gear_efficiency)
    
    # Wheel power
    wheel_power = engine_power * gear_efficiency
    
    # Alternatively, calculate from acceleration
    accel_power = settings.VEHICLE_MASS * acceleration * 9.81 * speed
    
    # Use the more realistic value
    acceleration_power = max(wheel_power, accel_power)
    
    return acceleration_power


def calculate_yaw_energy(speed: float, lateral_acc: float, steering_angle: float) -> float:
    """
    Calculate rotational/yaw energy in corners
    
    Args:
        speed: vehicle speed (m/s)
        lateral_acc: lateral acceleration (g)
        steering_angle: steering angle (degrees)
    
    Returns:
        Yaw energy in Joules
    """
    # Moment of inertia (rough estimate for race car)
    moment_of_inertia = 1500.0  # kg*m^2
    
    # Calculate yaw rate from lateral acceleration and speed
    if speed > 0.1:
        # v^2/r = lateral_acc * g
        # yaw_rate = v/r
        radius = (speed ** 2) / (abs(lateral_acc) * 9.81 + 0.001)
        yaw_rate = speed / radius
    else:
        yaw_rate = 0.0
    
    # Rotational kinetic energy: 0.5 * I * omega^2
    yaw_energy = 0.5 * moment_of_inertia * (yaw_rate ** 2)
    
    # Scale by steering angle
    steering_factor = abs(steering_angle) / 360.0
    yaw_energy *= (1.0 + steering_factor)
    
    return yaw_energy


def calculate_tire_load(speed: float, lateral_acc: float, longitudinal_acc: float) -> float:
    """
    Calculate tire load energy (from deformation and slip)
    
    Args:
        speed: vehicle speed (m/s)
        lateral_acc: lateral acceleration (g)
        longitudinal_acc: longitudinal acceleration (g)
    
    Returns:
        Tire load energy in Watts
        """
    # Combined acceleration (g-force magnitude)
    total_acc = np.sqrt(lateral_acc**2 + longitudinal_acc**2)
    
    # Tire load increases with speed and acceleration
    # Energy dissipated through tire deformation and slip
    
    # Base tire load (rolling at constant speed)
    base_load = settings.VEHICLE_MASS * 9.81 * 0.01  # 1% energy loss
    
    # Additional load from acceleration
    dynamic_load = settings.VEHICLE_MASS * total_acc * 9.81 * 0.05  # 5% energy loss under load
    
    # Power = Force * velocity
    tire_load_power = (base_load + dynamic_load) * speed
    
    return tire_load_power


def calculate_aerodynamic_drag(speed: float) -> float:
    """
    Calculate aerodynamic drag power loss
    
    Args:
        speed: vehicle speed (m/s)
    
    Returns:
        Drag power in Watts
    """
    # Drag force: F = 0.5 * rho * Cd * A * v^2
    drag_force = 0.5 * settings.AIR_DENSITY * settings.DRAG_COEFFICIENT * settings.FRONTAL_AREA * (speed ** 2)
    
    # Power = Force * velocity
    drag_power = drag_force * speed
    
    return drag_power


def calculate_rolling_resistance(speed: float) -> float:
    """
    Calculate rolling resistance power loss
    
    Args:
        speed: vehicle speed (m/s)
    
    Returns:
        Rolling resistance power in Watts
    """
    # Rolling resistance force: F = Crr * m * g
    rolling_force = settings.ROLLING_RESISTANCE * settings.VEHICLE_MASS * 9.81
    
    # Power = Force * velocity
    rolling_power = rolling_force * speed
    
    return rolling_power


def compute_segment_energy_potential(elevation: float, curvature: float, position: int, total_length: int) -> Dict:
    """
    Compute energy potential for a track segment
    Used in STEP 1: Energy Grid Generation
    
    Args:
        elevation: segment elevation (meters)
        curvature: segment curvature (1/meters)
        position: segment position index
        total_length: total track length in grid points
    
    Returns:
        Dictionary with energy components
    """
    # Potential energy from elevation
    # PE = m * g * h
    potential_energy = settings.VEHICLE_MASS * 9.81 * elevation
    
    # Estimated kinetic energy (based on typical cornering speed)
    # Higher curvature = lower speed = lower kinetic energy
    if abs(curvature) > 0.01:  # Tight corner
        estimated_speed = 20.0  # m/s
    elif abs(curvature) > 0.005:  # Medium corner
        estimated_speed = 35.0
    else:  # Straight or fast corner
        estimated_speed = 50.0
    
    kinetic_energy = 0.5 * settings.VEHICLE_MASS * (estimated_speed ** 2)
    
    # Total mechanical energy
    total_energy = potential_energy + kinetic_energy
    
    return {
        "potential": float(potential_energy),
        "kinetic": float(kinetic_energy),
        "total": float(total_energy)
    }


def calculate_brake_balance(brake_f: float, brake_r: float) -> float:
    """
    Calculate brake balance (front/rear distribution)
    
    Args:
        brake_f: front brake pressure (bar)
        brake_r: rear brake pressure (bar)
    
    Returns:
        Brake balance (0.0 = all rear, 1.0 = all front)
    """
    total = brake_f + brake_r
    if total > 0:
        return brake_f / total
    return 0.5  # Default to 50/50


def estimate_corner_exit_speed(entry_speed: float, curvature: float, throttle_point: float) -> float:
    """
    Estimate corner exit speed based on entry conditions
    
    Args:
        entry_speed: speed at corner entry (m/s)
        curvature: corner curvature (1/m)
        throttle_point: distance from apex to throttle application (m)
    
    Returns:
        Estimated exit speed (m/s)
    """
    # Minimum corner speed based on curvature
    # v_max = sqrt(g * mu / curvature)
    # Assume grip coefficient mu = 1.5
    if curvature > 0:
        max_corner_speed = np.sqrt(9.81 * 1.5 / curvature)
    else:
        max_corner_speed = entry_speed
    
    # Speed loss through corner
    min_speed = min(entry_speed, max_corner_speed)
    
    # Acceleration from apex to exit
    # Simplified: assume constant acceleration
    accel_rate = 3.0  # m/s^2 (conservative)
    time_to_exit = throttle_point / min_speed if min_speed > 0 else 0
    
    exit_speed = min_speed + accel_rate * time_to_exit
    
    return exit_speed


def calculate_tire_temperature_estimate(tire_load_history: List[float], ambient_temp: float = 25.0) -> float:
    """
    Estimate tire temperature based on load history
    
    Args:
        tire_load_history: recent tire load values (Watts)
        ambient_temp: ambient temperature (Celsius)
    
    Returns:
        Estimated tire temperature (Celsius)
    """
    # Average recent load
    avg_load = np.mean(tire_load_history[-50:]) if len(tire_load_history) > 0 else 0
    
    # Temperature rise proportional to load
    # Typical race tire: 80-110°C
    temp_rise = (avg_load / 10000.0) * 60.0  # Scale factor
    
    tire_temp = ambient_temp + temp_rise
    
    # Clamp to realistic range
    tire_temp = max(30.0, min(120.0, tire_temp))
    
    return tire_temp


def calculate_fuel_consumption_estimate(power_timeline: List[float], duration: float) -> float:
    """
    Estimate fuel consumption from power usage
    
    Args:
        power_timeline: timeline of power values (Watts)
        duration: duration of stint (seconds)
    
    Returns:
        Estimated fuel consumed (liters)
    """
    # Average power
    avg_power = np.mean(power_timeline) if len(power_timeline) > 0 else 0
    
    # Energy consumed (Joules)
    energy_consumed = avg_power * duration
    
    # Fuel energy content: ~32 MJ/liter for gasoline
    fuel_energy_content = 32e6  # J/liter
    
    # Engine efficiency: ~30%
    engine_efficiency = 0.30
    
    # Fuel consumed
    fuel_consumed = energy_consumed / (fuel_energy_content * engine_efficiency)
    
    return max(0.0, fuel_consumed)