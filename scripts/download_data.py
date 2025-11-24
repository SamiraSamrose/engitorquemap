## **File: scripts/download_data.py** (Data Download Script)

"""
Script to download track data from TRD
"""
import asyncio
import argparse
from pathlib import Path
import sys

# Add backend to path
sys.path.append(str(Path(__file__).parent.parent / 'backend'))

from data.data_loader import DataLoader
from config import settings


async def main():
    parser = argparse.ArgumentParser(description='Download track data from TRD')
    parser.add_argument('--track', type=str, help='Specific track to download')
    parser.add_argument('--all', action='store_true', help='Download all tracks')
    parser.add_argument('--process', action='store_true', help='Process data after download')
    
    args = parser.parse_args()
    
    loader = DataLoader()
    
    if args.all:
        print("Downloading all tracks...")
        results = await loader.download_all_tracks()
        
        for result in results:
            if result['status'] == 'success':
                print(f"✓ Downloaded {result['track']}")
            else:
                print(f"✗ Failed to download {result['track']}: {result.get('error')}")
        
        if args.process:
            print("\nProcessing downloaded data...")
            for result in results:
                if result['status'] == 'success':
                    process_result = await loader.process_data({
                        'track_name': result['track'],
                        'operations': ['clean', 'normalize']
                    })
                    print(f"Processed {result['track']}: {process_result['count']} files")
    
    elif args.track:
        print(f"Downloading {args.track}...")
        result = await loader.download_track_data(args.track)
        
        if result['status'] == 'success':
            print(f"✓ Downloaded {args.track}")
            
            if args.process:
                print(f"Processing {args.track}...")
                process_result = await loader.process_data({
                    'track_name': args.track,
                    'operations': ['clean', 'normalize']
                })
                print(f"Processed: {process_result['count']} files")
        else:
            print(f"✗ Failed: {result.get('error')}")
    
    else:
        print("Please specify --track <name> or --all")
        parser.print_help()


if __name__ == '__main__':
    asyncio.run(main())
