## **File: backend/data/data_loader.py** (Data Management)

"""
Data Loader
Downloads and processes track data from TRD
Manages data ingestion pipeline
"""
import requests
import zipfile
import os
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import logging
import asyncio

from config import settings

logger = logging.getLogger(__name__)


class DataLoader:
    """
    Handles data downloading and processing
    """
    
    def __init__(self):
        self.base_url = settings.TRACK_DATA_BASE_URL
        self.raw_dir = Path(settings.RAW_DATA_DIR)
        self.processed_dir = Path(settings.PROCESSED_DATA_DIR)
        
        # Ensure directories exist
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
    
    async def download_track_data(self, track_name: str) -> Dict:
        """
        Download track data ZIP from TRD
        
        Args:
            track_name: Name of track (e.g., 'indianapolis')
        
        Returns:
            Status dictionary
        """
        logger.info(f"Downloading data for {track_name}")
        
        zip_filename = f"{track_name}.zip"
        zip_url = f"{self.base_url}{zip_filename}"
        local_path = self.raw_dir / zip_filename
        
        try:
            # Download ZIP file
            response = requests.get(zip_url, stream=True, timeout=60)
            response.raise_for_status()
            
            # Save to disk
            with open(local_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            logger.info(f"Downloaded {zip_filename} ({local_path.stat().st_size / 1024 / 1024:.2f} MB)")
            
            # Extract ZIP
            extract_dir = self.raw_dir / track_name
            with zipfile.ZipFile(local_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            
            logger.info(f"Extracted to {extract_dir}")
            
            return {
                "status": "success",
                "track": track_name,
                "zip_path": str(local_path),
                "extract_path": str(extract_dir)
            }
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to download {track_name}: {e}")
            return {
                "status": "error",
                "track": track_name,
                "error": str(e)
            }
    
    async def download_all_tracks(self) -> List[Dict]:
        """Download all track data"""
        results = []
        
        for track in settings.TRACKS:
            result = await self.download_track_data(track)
            results.append(result)
            await asyncio.sleep(1)  # Rate limiting
        
        return results
    
    async def process_data(self, processing_config: Dict) -> Dict:
        """
        Process raw data files
        
        Args:
            processing_config: Configuration for processing
                - track_name: Track to process
                - operations: List of operations (clean, normalize, merge)
        """
        track_name = processing_config.get('track_name')
        operations = processing_config.get('operations', ['clean', 'normalize'])
        
        logger.info(f"Processing data for {track_name}")
        
        # Load raw data
        extract_dir = self.raw_dir / track_name
        
        if not extract_dir.exists():
            return {
                "status": "error",
                "message": f"Data not found for {track_name}. Download first."
            }
        
        processed_files = []
        
        # Process each CSV file
        for csv_file in extract_dir.rglob('*.csv'):
            try:
                df = pd.read_csv(csv_file)
                
                # Apply operations
                if 'clean' in operations:
                    df = self._clean_data(df)
                
                if 'normalize' in operations:
                    df = self._normalize_data(df)
                
                # Save processed data
                output_path = self.processed_dir / f"{track_name}_{csv_file.stem}_processed.parquet"
                df.to_parquet(output_path, index=False)
                
                processed_files.append(str(output_path))
                logger.info(f"Processed: {csv_file.name} -> {output_path.name}")
                
            except Exception as e:
                logger.error(f"Failed to process {csv_file}: {e}")
        
        return {
            "status": "success",
            "track": track_name,
            "processed_files": processed_files,
            "count": len(processed_files)
        }
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean dataframe - remove nulls, fix types"""
        # Remove rows with all NaN
        df = df.dropna(how='all')
        
        # Fill numeric NaNs with 0
        numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
        df[numeric_columns] = df[numeric_columns].fillna(0)
        
        # Convert column names to lowercase and remove spaces
        df.columns = [col.lower().replace(' ', '_') for col in df.columns]
        
        return df
    
    def _normalize_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize column names and values"""
        # Standardize GPS column names
        rename_map = {}
        for col in df.columns:
            if 'vbox_long' in col.lower():
                rename_map[col] = 'gps_longitude'
            elif 'vbox_lat' in col.lower():
                rename_map[col] = 'gps_latitude'
        
        if rename_map:
            df = df.rename(columns=rename_map)
        
        return df
    
    async def get_status(self) -> Dict:
        """Get status of available data"""
        downloaded_tracks = []
        processed_tracks = []
        
        # Check raw data
        for track in settings.TRACKS:
            zip_path = self.raw_dir / f"{track}.zip"
            if zip_path.exists():
                downloaded_tracks.append(track)
        
        # Check processed data
        processed_files = list(self.processed_dir.glob('*.parquet'))
        processed_track_names = set([f.stem.split('_')[0] for f in processed_files])
        processed_tracks = list(processed_track_names)
        
        return {
            "total_tracks": len(settings.TRACKS),
            "downloaded_tracks": downloaded_tracks,
            "downloaded_count": len(downloaded_tracks),
            "processed_tracks": processed_tracks,
            "processed_count": len(processed_tracks),
            "processed_files_count": len(processed_files)
        }
