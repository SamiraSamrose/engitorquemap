#backend/utils/geometry.py (Geometric Utilities)
"""
Geometric utility functions for track analysis
"""
import numpy as np
from typing import Tuple


def calculate_curvature(p_prev: np.ndarray, p_curr: np.ndarray, p_next: np.ndarray) -> float:
    """
    Calculate curvature at a point using three consecutive points
    
    Args:
        p_prev: previous point [x, y]
        p_curr: current point [x, y]
        p_next: next point [x, y]
    
    Returns:
        Curvature value (1/radius)
    """
    # Vectors
    v1 = p_curr - p_prev
    v2 = p_next - p_curr
    
    # Lengths
    len1 = np.linalg.norm(v1)
    len2 = np.linalg.norm(v2)
    
    if len1 < 1e-6 or len2 < 1e-6:
        return 0.0
    
    # Normalize
    v1 = v1 / len1
    v2 = v2 / len2
    
    # Angle change
    dot_product = np.clip(np.dot(v1, v2), -1.0, 1.0)
    angle = np.arccos(dot_product)
    
    # Curvature = angle / arc_length
    arc_length = (len1 + len2) / 2.0
    curvature = angle / arc_length if arc_length > 0 else 0.0
    
    # Sign of curvature (left/right turn)
    cross = v1[0] * v2[1] - v1[1] * v2[0]
    curvature *= np.sign(cross)
    
    return curvature


def classify_corner(avg_curvature: float, max_curvature: float) -> str:
    """
    Classify corner type based on curvature
    
    Args:
        avg_curvature: average curvature in corner
        max_curvature: maximum curvature in corner
    
    Returns:
        Corner classification string
    """
    abs_avg = abs(avg_curvature)
    abs_max = abs(max_curvature)
    
    if abs_max > 0.05:
        return "hairpin"
    elif abs_max > 0.02:
        return "slow"
    elif abs_max > 0.01:
        return "medium"
    elif abs_max > 0.005:
        return "fast"
    else:
        return "straight"


def compute_track_width(boundaries: Tuple[np.ndarray, np.ndarray], position: int) -> float:
    """
    Compute track width at a specific position
    
    Args:
        boundaries: tuple of (inner_boundary, outer_boundary) arrays
        position: index position along track
    
    Returns:
        Track width in meters (approximate)
    """
    inner, outer = boundaries
    
    if position >= len(inner) or position >= len(outer):
        return 12.0  # Default
    
    # Distance between inner and outer boundary
    diff = outer[position] - inner[position]
    
    # Convert degrees to meters (rough approximation)
    width_degrees = np.linalg.norm(diff)
    width_meters = width_degrees * 111000.0  # 1 degree ≈ 111 km
    
    return width_meters


def find_racing_line(center_line: np.ndarray, curvature: np.ndarray, 
                     track_width: float = 12.0) -> np.ndarray:
    """
    Calculate optimal racing line (simplified)
    
    Args:
        center_line: track center line coordinates
        curvature: curvature at each point
        track_width: available track width (meters)
    
    Returns:
        Racing line coordinates
    """
    racing_line = np.copy(center_line)
    
    for i in range(len(center_line)):
        # In corners, move to outside before apex, inside at apex, outside after
        curv = curvature[i]
        
        if abs(curv) > 0.005:  # In a corner
            # Calculate perpendicular offset
            i_next = (i + 1) % len(center_line)
            tangent = center_line[i_next] - center_line[i]
            tangent_norm = np.linalg.norm(tangent)
            
            if tangent_norm > 0:
                tangent = tangent / tangent_norm
                perpendicular = np.array([-tangent[1], tangent[0]])
                
                # Offset toward inside of corner (opposite to curvature sign)
                offset_direction = -np.sign(curv)
                offset_magnitude = track_width * 0.3  # Use 30% of track width
                
                # Convert to degrees
                offset_degrees = (offset_magnitude / 111000.0)
                
                racing_line[i] = center_line[i] + perpendicular * offset_direction * offset_degrees
    
    return racing_line


def interpolate_track_position(lap_distance: float, track_length: float, 
                               total_points: int) -> int:
    """
    Convert lap distance to track position index
    
    Args:
        lap_distance: distance along lap (meters)
        track_length: total track length (meters)
        total_points: number of points in track representation
    
    Returns:
        Position index
    """
    # Normalize distance
    normalized = (lap_distance % track_length) / track_length
    
    # Convert to index
    index = int(normalized * total_points)
    
    return min(max(0, index), total_points - 1)


def calculate_segment_gradient(elevation_profile: np.ndarray, segment_length: float = 1.0) -> np.ndarray:
    """
    Calculate gradient (slope) for each segment
    
    Args:
        elevation_profile: elevation at each point (meters)
        segment_length: length of each segment (meters)
    
    Returns:
        Gradient array (rise/run ratio)
    """
    gradients = np.gradient(elevation_profile) / segment_length
    return gradients


def smooth_trajectory(trajectory: np.ndarray, window_size: int = 5) -> np.ndarray:
    """
    Smooth a trajectory using moving average
    
    Args:
        trajectory: array of coordinates
        window_size: smoothing window size
    
    Returns:
        Smoothed trajectory
    """
    from scipy.ndimage import uniform_filter1d
    
    smoothed = np.copy(trajectory)
    
    for dim in range(trajectory.shape[1]):
        smoothed[:, dim] = uniform_filter1d(trajectory[:, dim], size=window_size, mode='wrap')
    
    return smoothed


def calculate_apex_position(corner_start: int, corner_end: int, curvature: np.ndarray) -> int:
    """
    Find apex position within a corner
    
    Args:
        corner_start: corner start index
        corner_end: corner end index
        curvature: curvature array
    
    Returns:
        Apex index
    """
    corner_curvature = curvature[corner_start:corner_end]
    apex_offset = np.argmax(np.abs(corner_curvature))
    apex_index = corner_start + apex_offset
    
    return apex_index


def calculate_corner_radius(curvature: float) -> float:
    """
    Calculate corner radius from curvature
    
    Args:
        curvature: curvature value (1/radius)
    
    Returns:
        Radius in meters
    """
    if abs(curvature) < 1e-6:
        return float('inf')
    
    radius = 1.0 / abs(curvature)
    
    # Convert from coordinate space to meters (rough approximation)
    radius_meters = radius * 111000.0
    
    return radius_meters