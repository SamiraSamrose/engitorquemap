#backend/tests/test_energy.py
"""
Unit tests for energy vector calculations
"""
import pytest
import numpy as np
from services.energy_vectors import EnergyVectorService
from config import settings


@pytest.fixture
def energy_service():
    return EnergyVectorService()


@pytest.mark.asyncio
async def test_compute_vectors(energy_service):
    """Test energy vector computation"""
    telemetry = {
        'speed': 45.0,
        'gear': 4,
        'nmot': 5000,
        'aps': 80,
        'ath': 75,
        'pbrake_f': 0,
        'pbrake_r': 0,
        'accx_can': 0.5,
        'accy_can': 0.8,
        'Steering_Angle': 15
    }
    
    result = await energy_service.compute_vectors(telemetry)
    
    assert 'braking_power' in result
    assert 'acceleration_power' in result
    assert 'energy_efficiency' in result
    assert result['speed'] == 45.0


def test_calculate_efficiency(energy_service):
    """Test efficiency calculation"""
    efficiency = energy_service._calculate_efficiency(100000, 30000)
    
    assert 0.0 <= efficiency <= 1.0


@pytest.mark.asyncio
async def test_analyze_session(energy_service):
    """Test session analysis"""
    # This would require mock data
    # For now, just test the method exists
    assert hasattr(energy_service, 'analyze_session')