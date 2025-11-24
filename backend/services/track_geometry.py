#backend/services/track_geometry.py (Track-Energy Geometry)
"""
Track Geometry Service - STEP 1
Reconstructs track geometry from ZIP datasets with spline interpolation,
curvature extraction, corner classification, and 1-meter energy grid creation
"""
import numpy as np
import pandas as pd
from scipy.interpolate import splprep, splev, UnivariateSpline
from scipy.spatial.distance import cdist
from shapely.geometry import Point, LineString, Polygon, MultiPolygon
from shapely.ops import unary_union
import geopandas as gpd
from typing import Dict, List, Tuple, Optional
import zipfile
import os
import json
import logging

from config import settings
from utils.geometry import calculate_curvature, classify_corner, compute_track_width
from utils.physics import compute_segment_energy_potential

logger = logging.getLogger(__name__)


class TrackGeometryService:
    """
    Handles track geometry reconstruction and energy grid generation
    """
    
    def __init__(self):
        self.track_cache = {}
        self.energy_grids = {}
        
    async def get_track_geometry(self, track_name: str) -> Dict:
        """
        Main method to get complete track geometry
        Returns: polygons, elevation curves, center line, boundaries
        """
        if track_name in self.track_cache:
            logger.info(f"Returning cached geometry for {track_name}")
            return self.track_cache[track_name]
        
        logger.info(f"Processing track geometry for {track_name}")
        
        # Load track data from ZIP
        track_data = await self._load_track_data(track_name)
        
        # Extract GPS coordinates from telemetry
        gps_coords = self._extract_gps_coordinates(track_data)
        
        # Reconstruct center line with spline interpolation
        center_line = self._reconstruct_center_line(gps_coords)
        
        # Extract elevation profile
        elevation_profile = self._extract_elevation(track_data, center_line)
        
        # Calculate curvature along track
        curvature = self._calculate_track_curvature(center_line)
        
        # Classify corners
        corners = self._classify_corners(center_line, curvature)
        
        # Reconstruct track boundaries (red/white lines)
        boundaries = self._reconstruct_boundaries(track_data, center_line)
        
        # Create track polygon
        track_polygon = self._create_track_polygon(boundaries)
        
        # Generate sector definitions from "Analysis with Sections"
        sectors = await self._generate_sectors(track_data)
        
        geometry = {
            "track_name": track_name,
            "center_line": center_line.tolist(),
            "elevation_profile": elevation_profile.tolist(),
            "curvature": curvature.tolist(),
            "corners": corners,
            "boundaries": {
                "inner": boundaries["inner"].tolist(),
                "outer": boundaries["outer"].tolist()
            },
            "polygon": self._polygon_to_geojson(track_polygon),
            "sectors": sectors,
            "length_meters": self._calculate_track_length(center_line)
        }
        
        self.track_cache[track_name] = geometry
        return geometry
    
    def _load_track_data(self, track_name: str) -> Dict:
        """Load all CSV files from track ZIP"""
        zip_path = os.path.join(settings.RAW_DATA_DIR, f"{track_name}.zip")
        
        if not os.path.exists(zip_path):
            raise FileNotFoundError(f"Track data not found: {zip_path}")
        
        data = {}
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            for file_info in zip_ref.filelist:
                if file_info.filename.endswith('.csv') or file_info.filename.endswith('.CSV'):
                    file_key = os.path.basename(file_info.filename).replace('.csv', '').replace('.CSV', '')
                    try:
                        with zip_ref.open(file_info.filename) as f:
                            df = pd.read_csv(f)
                            data[file_key] = df
                            logger.info(f"Loaded {file_key}: {len(df)} rows")
                    except Exception as e:
                        logger.warning(f"Could not load {file_info.filename}: {e}")
        
        return data
    
    def _extract_gps_coordinates(self, track_data: Dict) -> np.ndarray:
        """Extract GPS coordinates from telemetry data"""
        # Try to find telemetry data
        telemetry_keys = [k for k in track_data.keys() if 'telemetry' in k.lower()]
        
        if not telemetry_keys:
            raise ValueError("No telemetry data found")
        
        telemetry = track_data[telemetry_keys[0]]
        
        # Extract longitude and latitude
        lon_col = [c for c in telemetry.columns if 'long' in c.lower() and 'vbox' in c.lower()]
        lat_col = [c for c in telemetry.columns if 'lat' in c.lower() and 'vbox' in c.lower()]
        
        if not lon_col or not lat_col:
            raise ValueError("GPS columns not found in telemetry")
        
        # Convert from minutes format to decimal degrees
        lon = telemetry[lon_col[0]].values / 60.0
        lat = telemetry[lat_col[0]].values / 60.0
        
        # Remove invalid coordinates
        valid_mask = ~(np.isnan(lon) | np.isnan(lat))
        gps_coords = np.column_stack([lon[valid_mask], lat[valid_mask]])
        
        # Sample evenly to reduce data size
        if len(gps_coords) > 10000:
            indices = np.linspace(0, len(gps_coords)-1, 10000, dtype=int)
            gps_coords = gps_coords[indices]
        
        logger.info(f"Extracted {len(gps_coords)} GPS coordinates")
        return gps_coords
    
    def _reconstruct_center_line(self, gps_coords: np.ndarray) -> np.ndarray:
        """
        Reconstruct smooth center line using B-spline interpolation
        This creates a high-resolution representation of the track
        """
        # Fit parametric spline through GPS points
        # s parameter controls smoothness (higher = smoother)
        tck, u = splprep([gps_coords[:, 0], gps_coords[:, 1]], s=0.0001, k=3, per=True)
        
        # Evaluate spline at high resolution (1-meter intervals)
        track_length_estimate = self._estimate_track_length(gps_coords)
        num_points = int(track_length_estimate / settings.ENERGY_GRID_RESOLUTION)
        
        u_fine = np.linspace(0, 1, num_points)
        lon_fine, lat_fine = splev(u_fine, tck)
        
        center_line = np.column_stack([lon_fine, lat_fine])
        
        logger.info(f"Reconstructed center line with {len(center_line)} points")
        return center_line
    
    def _estimate_track_length(self, coords: np.ndarray) -> float:
        """Estimate track length in meters from GPS coordinates"""
        # Rough approximation: 1 degree lat/lon ≈ 111 km at mid-latitudes
        diffs = np.diff(coords, axis=0)
        distances = np.sqrt((diffs[:, 0] * 111000 * np.cos(np.radians(coords[:-1, 1])))**2 +
                           (diffs[:, 1] * 111000)**2)
        return np.sum(distances)
    
    def _calculate_track_length(self, center_line: np.ndarray) -> float:
        """Calculate accurate track length"""
        return self._estimate_track_length(center_line)
    
    def _extract_elevation(self, track_data: Dict, center_line: np.ndarray) -> np.ndarray:
        """Extract elevation profile along track"""
        # Try to find elevation data in telemetry
        telemetry_keys = [k for k in track_data.keys() if 'telemetry' in k.lower()]
        
        if not telemetry_keys:
            # No elevation data, return flat track
            return np.zeros(len(center_line))
        
        telemetry = track_data[telemetry_keys[0]]
        
        # Look for altitude/elevation columns
        elev_cols = [c for c in telemetry.columns if any(x in c.lower() for x in ['alt', 'elev', 'height'])]
        
        if not elev_cols:
            return np.zeros(len(center_line))
        
        elevation_raw = telemetry[elev_cols[0]].values
        
        # Interpolate to match center line resolution
        if len(elevation_raw) != len(center_line):
            x_old = np.linspace(0, 1, len(elevation_raw))
            x_new = np.linspace(0, 1, len(center_line))
            elevation = np.interp(x_new, x_old, elevation_raw)
        else:
            elevation = elevation_raw
        
        # Smooth elevation profile
        from scipy.signal import savgol_filter
        elevation = savgol_filter(elevation, window_length=51, polyorder=3)
        
        return elevation
    
    def _calculate_track_curvature(self, center_line: np.ndarray) -> np.ndarray:
        """Calculate curvature at each point along track"""
        curvature = np.zeros(len(center_line))
        
        for i in range(len(center_line)):
            i_prev = (i - 1) % len(center_line)
            i_next = (i + 1) % len(center_line)
            
            p_prev = center_line[i_prev]
            p_curr = center_line[i]
            p_next = center_line[i_next]
            
            curvature[i] = calculate_curvature(p_prev, p_curr, p_next)
        
        # Smooth curvature
        from scipy.signal import savgol_filter
        curvature = savgol_filter(curvature, window_length=51, polyorder=3)
        
        return curvature
    
    def _classify_corners(self, center_line: np.ndarray, curvature: np.ndarray) -> List[Dict]:
        """
        Classify corners as hairpin, slow, medium, fast, straight
        Based on curvature analysis
        """
        # Threshold for corner detection
        curvature_threshold = 0.001
        
        # Find regions of high curvature
        corner_mask = np.abs(curvature) > curvature_threshold
        
        # Find contiguous corner regions
        corners = []
        in_corner = False
        corner_start = 0
        
        for i in range(len(corner_mask)):
            if corner_mask[i] and not in_corner:
                corner_start = i
                in_corner = True
            elif not corner_mask[i] and in_corner:
                corner_end = i
                in_corner = False
                # Analyze corner
                corner_curvature = curvature[corner_start:corner_end]
                avg_curvature = np.mean(np.abs(corner_curvature))
                max_curvature = np.max(np.abs(corner_curvature))
                
                corner_type = classify_corner(avg_curvature, max_curvature)
                
                corners.append({
                    "start_index": int(corner_start),
                    "end_index": int(corner_end),
                    "type": corner_type,
                    "average_curvature": float(avg_curvature),
                    "max_curvature": float(max_curvature),
                    "apex_index": int(corner_start + np.argmax(np.abs(corner_curvature)))
                })
        
        logger.info(f"Classified {len(corners)} corners")
        return corners
    
    def _reconstruct_boundaries(self, track_data: Dict, center_line: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Reconstruct inner and outer track boundaries
        Using track width from "Analysis with Sections" data
        """
        # Try to find section analysis data
        section_keys = [k for k in track_data.keys() if 'section' in k.lower() or 'analysis' in k.lower()]
        
        # Default track width if not found
        track_width = 12.0  # meters
        
        if section_keys:
            sections_df = track_data[section_keys[0]]
            # Try to extract track width information
            # This varies by dataset format
            pass
        
        # Calculate perpendicular vectors at each point
        inner_boundary = np.zeros_like(center_line)
        outer_boundary = np.zeros_like(center_line)
        
        for i in range(len(center_line)):
            i_next = (i + 1) % len(center_line)
            
            # Tangent vector
            tangent = center_line[i_next] - center_line[i]
            tangent_norm = np.linalg.norm(tangent)
            
            if tangent_norm > 0:
                tangent = tangent / tangent_norm
                
                # Perpendicular vector (rotate 90 degrees)
                perpendicular = np.array([-tangent[1], tangent[0]])
                
                # Scale by half track width
                # Convert to approximate meters (rough conversion)
                width_degrees = (track_width / 2) / 111000.0
                
                inner_boundary[i] = center_line[i] - perpendicular * width_degrees
                outer_boundary[i] = center_line[i] + perpendicular * width_degrees
        
        return {
            "inner": inner_boundary,
            "outer": outer_boundary
        }
    
    def _create_track_polygon(self, boundaries: Dict[str, np.ndarray]) -> Polygon:
        """Create track polygon from boundaries"""
        inner = boundaries["inner"]
        outer = boundaries["outer"]
        
        # Create polygon by combining outer and reversed inner boundaries
        coords = np.vstack([outer, inner[::-1]])
        
        return Polygon(coords)
    
    def _polygon_to_geojson(self, polygon: Polygon) -> Dict:
        """Convert Shapely polygon to GeoJSON"""
        return {
            "type": "Polygon",
            "coordinates": [list(polygon.exterior.coords)]
        }
    
    async def _generate_sectors(self, track_data: Dict) -> List[Dict]:
        """
        Generate sector definitions from "Analysis with Sections" dataset
        Aligns with GPS coordinates and creates sector boundaries
        """
        section_keys = [k for k in track_data.keys() if 'section' in k.lower()]
        
        if not section_keys:
            logger.warning("No section data found, creating default sectors")
            return self._create_default_sectors()
        
        sections_df = track_data[section_keys[0]]
        
        # Extract unique sectors (e.g., IM1a, IM1b, FL)
        sector_col = [c for c in sections_df.columns if 'sector' in c.lower() or 'section' in c.lower()]
        
        if not sector_col:
            return self._create_default_sectors()
        
        unique_sectors = sections_df[sector_col[0]].unique()
        
        sectors = []
        for i, sector_name in enumerate(unique_sectors):
            sector_data = sections_df[sections_df[sector_col[0]] == sector_name]
            
            sectors.append({
                "id": i,
                "name": str(sector_name),
                "start_index": 0,  # Would be calculated from GPS alignment
                "end_index": 0,
                "length": 0.0,
                "type": self._classify_sector_type(sector_name)
            })
        
        return sectors
    
    def _create_default_sectors(self) -> List[Dict]:
        """Create default 3-sector split if no data available"""
        return [
            {"id": 0, "name": "Sector 1", "type": "sector"},
            {"id": 1, "name": "Sector 2", "type": "sector"},
            {"id": 2, "name": "Sector 3", "type": "sector"}
        ]
    
    def _classify_sector_type(self, sector_name: str) -> str:
        """Classify sector type based on naming convention"""
        name_lower = str(sector_name).lower()
        
        if 'im' in name_lower or 'intermediate' in name_lower:
            return "intermediate"
        elif 'fl' in name_lower or 'finish' in name_lower:
            return "finish"
        elif 'sf' in name_lower or 'start' in name_lower:
            return "start_finish"
        else:
            return "sector"
    
    async def get_sectors(self, track_name: str) -> List[Dict]:
        """Get sector definitions for a track"""
        geometry = await self.get_track_geometry(track_name)
        return geometry["sectors"]
    
    async def get_energy_grid(self, track_name: str) -> Dict:
        """
        Generate 1-meter resolution energy grid for track
        STEP 1: Create energy grid per track segment
        """
        if track_name in self.energy_grids:
            return self.energy_grids[track_name]
        
        logger.info(f"Generating energy grid for {track_name}")
        
        geometry = await self.get_track_geometry(track_name)
        center_line = np.array(geometry["center_line"])
        elevation = np.array(geometry["elevation_profile"])
        curvature = np.array(geometry["curvature"])
        
        # Calculate energy potential at each grid point
        energy_grid = []
        
        for i in range(len(center_line)):
            # Calculate energy components
            energy_data = compute_segment_energy_potential(
                elevation=elevation[i],
                curvature=curvature[i],
                position=i,
                total_length=len(center_line)
            )
            
            energy_grid.append({
                "index": i,
                "position": center_line[i].tolist(),
                "elevation": float(elevation[i]),
                "curvature": float(curvature[i]),
                "potential_energy": energy_data["potential"],
                "kinetic_energy_estimate": energy_data["kinetic"],
                "total_energy": energy_data["total"]
            })
        
        grid_data = {
            "track_name": track_name,
            "resolution_meters": settings.ENERGY_GRID_RESOLUTION,
            "grid_points": len(energy_grid),
            "grid": energy_grid
        }
        
        self.energy_grids[track_name] = grid_data
        return grid_data
    
    async def generate_hologram_data(self, track_name: str) -> Dict:
        """
        Generate 3D holographic energy stream data for visualization
        STEP 8: Energy-Flow Holograms
        """
        geometry = await self.get_track_geometry(track_name)
        energy_grid = await self.get_energy_grid(track_name)
        
        # Create 3D coordinates with elevation
        center_line = np.array(geometry["center_line"])
        elevation = np.array(geometry["elevation_profile"])
        
        # Convert to 3D points (lon, lat, elevation)
        points_3d = np.column_stack([center_line, elevation])
        
        # Generate energy flow vectors
        flow_vectors = []
        for i in range(len(points_3d) - 1):
            direction = points_3d[i + 1] - points_3d[i]
            direction_norm = np.linalg.norm(direction)
            
            if direction_norm > 0:
                direction = direction / direction_norm
            
            energy_magnitude = energy_grid["grid"][i]["total_energy"]
            
            flow_vectors.append({
                "position": points_3d[i].tolist(),
                "direction": direction.tolist(),
                "magnitude": float(energy_magnitude)
            })
        
        return {
            "track_name": track_name,
            "points_3d": points_3d.tolist(),
            "flow_vectors": flow_vectors,
            "bounds": {
                "min": points_3d.min(axis=0).tolist(),
                "max": points_3d.max(axis=0).tolist()
            }
        }