#backend/tests/test_predictor.py
"""
Unit tests for time-shift predictor
"""
import pytest
import numpy as np
from models.timeshift_predictor import TimeShiftPredictor


@pytest.fixture
def predictor():
    return TimeShiftPredictor()


def test_build_feature_vector(predictor):
    """Test feature vector construction"""
    telemetry = {
        'speed': 50.0,
        'brake_pressure': 60.0,
        'throttle': 80.0,
        'lateral_g': 1.2,
        'longitudinal_g': 0.8,
        'steering_angle': 20.0,
        'corner_curvature': 0.02,
        'elevation_change': 5.0
    }
    
    features = predictor._build_feature_vector(
        telemetry, 
        brake_shift=12.0, 
        throttle_shift=-8.0,
        apex_speed_change=2.0,
        rotation_change=5.0
    )
    
    assert len(features) == 12
    assert features[6] == 12.0  # brake_shift


def test_estimate_time_delta_physics(predictor):
    """Test physics-based time delta estimation"""
    modifications = {
        'brake_point_shift': 10.0,
        'throttle_point_shift': -5.0,
        'apex_speed_change': 1.0
    }
    
    delta = predictor._estimate_time_delta_physics(modifications)
    
    assert isinstance(delta, float)


def test_calculate_energy_impact(predictor):
    """Test energy impact calculation"""
    modifications = {
        'brake_point_shift': 12.0,
        'throttle_point_shift': -8.0
    }
    
    impact = predictor._calculate_energy_impact(modifications)
    
    assert 'brake_energy_saved_kj' in impact
    assert 'throttle_energy_added_kj' in impact
    assert 'net_energy_change_kj' in impact