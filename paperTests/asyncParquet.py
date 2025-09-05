import asyncio
import aiohttp
import os
import json
import sys
import argparse
import pandas as pd
from pathlib import Path
from tqdm.asyncio import tqdm

async def download_parquet_file(session, url, output_path, semaphore):
    """Download a single parquet file asynchronously"""
    async with semaphore:
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=300)) as response:
                if response.status == 200:
                    content = await response.read()
                    
                    # Ensure directory exists
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    
                    with open(output_path, 'wb') as f:
                        f.write(content)
                    
                    file_size = os.path.getsize(output_path)
                    return url, output_path, file_size, None
                else:
                    return url, None, 0, f"HTTP {response.status}"
        except Exception as e:
            return url, None, 0, str(e)

async def download_parquet_files(urls_and_paths, concurrent_downloads=5):
    """Download multiple parquet files concurrently"""
    semaphore = asyncio.Semaphore(concurrent_downloads)
    
    # Configure session
    connector = aiohttp.TCPConnector(
        limit=concurrent_downloads * 2,
        limit_per_host=10,
        ttl_dns_cache=300,
        use_dns_cache=True,
    )
    
    async with aiohttp.ClientSession(
        connector=connector,
        timeout=aiohttp.ClientTimeout(total=600),
        headers={'User-Agent': 'LDAWT-ParquetDownloader/1.0'}
    ) as session:
        
        tasks = [
            download_parquet_file(session, url, path, semaphore)
            for url, path in urls_and_paths
        ]
        
        results = []
        for future in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Downloading parquet files"):
            result = await future
            results.append(result)
            
            url, path, size, error = result
            if error:
                print(f"\n[ERROR] {url}: {error}")
            else:
                print(f"\n[SUCCESS] {path} ({size / 1e6:.2f} MB)")
        
        return results

def load_json_config(config_path):
    """
    Load configuration from JSON file and return as dictionary.
    """
    try:
        with open(config_path, 'r') as f:
            config_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file '{config_path}' not found.")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in configuration file '{config_path}': {e}")
        sys.exit(1)
    
    # Define required fields and defaults
    required_fields = ['input_parquet', 'url_column']
    defaults = {
        'output_directory': 'parquet_files',
        'filename_column': None,  # Optional: column for custom filenames
        'concurrent_downloads': 5,
        'timeout': 300
    }
    
    # Check required fields
    for field in required_fields:
        if field not in config_data:
            print(f"Error: Required field '{field}' not found in JSON configuration.")
            sys.exit(1)
    
    # Apply defaults for missing optional fields
    for field, default_value in defaults.items():
        if field not in config_data:
            config_data[field] = default_value
    
    # Validate data types
    type_validators = {
        'input_parquet': str,
        'url_column': str,
        'output_directory': str,
        'filename_column': (str, type(None)),
        'concurrent_downloads': int,
        'timeout': int
    }
    
    for field, expected_type in type_validators.items():
        if field in config_data:
            if not isinstance(config_data[field], expected_type):
                print(f"Error: Field '{field}' must be of type {expected_type.__name__ if not isinstance(expected_type, tuple) else ' or '.join(str(t.__name__) for t in expected_type if t is not type(None))}.")
                sys.exit(1)
    
    return config_data

def load_urls_from_parquet(config):
    """
    Load URLs from parquet file and generate (url, output_path) pairs.
    """
    try:
        # Load the parquet file
        df = pd.read_parquet(config['input_parquet'])
    except Exception as e:
        print(f"Error reading parquet file '{config['input_parquet']}': {e}")
        sys.exit(1)
    
    # Validate URL column exists
    if config['url_column'] not in df.columns:
        print(f"Error: Column '{config['url_column']}' not found in parquet file.")
        print(f"Available columns: {list(df.columns)}")
        sys.exit(1)
    
    # Validate filename column if specified
    if config['filename_column'] and config['filename_column'] not in df.columns:
        print(f"Error: Column '{config['filename_column']}' not found in parquet file.")
        print(f"Available columns: {list(df.columns)}")
        sys.exit(1)
    
    # Filter out rows with missing URLs
    initial_count = len(df)
    df = df.dropna(subset=[config['url_column']])
    filtered_count = len(df)
    
    if filtered_count < initial_count:
        print(f"Filtered out {initial_count - filtered_count} rows with missing URLs")
    
    if filtered_count == 0:
        print("No valid URLs found in parquet file")
        sys.exit(1)
    
    # Generate (url, output_path) pairs
    urls_and_paths = []
    output_dir = Path(config['output_directory'])
    
    for idx, row in df.iterrows():
        url = row[config['url_column']]
        
        # Generate filename
        if config['filename_column'] and pd.notna(row[config['filename_column']]):
            # Use custom filename from column
            filename = str(row[config['filename_column']])
            if not filename.endswith('.parquet'):
                filename += '.parquet'
        else:
            # Generate filename from URL or index
            if '/' in url:
                filename = url.split('/')[-1]
            else:
                filename = f"file_{idx}.parquet"
            
            if not filename.endswith('.parquet'):
                filename += '.parquet'
        
        output_path = output_dir / filename
        urls_and_paths.append((str(url), str(output_path)))
    
    print(f"Found {len(urls_and_paths)} URLs to download")
    return urls_and_paths

def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Download parquet files asynchronously from URLs in a parquet file.")
    parser.add_argument("--config", type=str, required=True, help="Path to JSON configuration file.")
    return parser.parse_args()

async def main():
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    config = load_json_config(args.config)
    print(f"Using configuration from: {args.config}")
    
    # Load URLs from parquet file
    urls_and_paths = load_urls_from_parquet(config)
    
    # Run downloads
    print(f"Starting downloads with {config['concurrent_downloads']} concurrent connections...")
    results = await download_parquet_files(urls_and_paths, concurrent_downloads=config['concurrent_downloads'])
    
    # Summary
    successful = sum(1 for _, _, _, error in results if error is None)
    failed = len(results) - successful
    total_size = sum(size for _, _, size, error in results if error is None)
    
    print(f"\nDownload Summary:")
    print(f"  - Successful downloads: {successful}")
    print(f"  - Failed downloads: {failed}")
    print(f"  - Total data downloaded: {total_size / 1e6:.2f} MB")
    
    if failed > 0:
        print(f"\nFailed downloads:")
        for url, _, _, error in results:
            if error:
                print(f"  - {url}: {error}")

if __name__ == '__main__':
    asyncio.run(main())