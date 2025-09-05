#!/usr/bin/env python3

import pandas as pd
from huggingface_hub import HfApi, hf_hub_url

def create_url_parquet(dataset_name, output_file="url_list.parquet"):
    """
    Create a parquet file containing URLs of all parquet files from a HuggingFace dataset.
    """
    print(f"Fetching parquet file list from dataset: {dataset_name}")
    
    # Get all files from the dataset
    api = HfApi()
    files = api.list_repo_files(repo_id=dataset_name, repo_type="dataset")
    
    # Filter for parquet files
    parquet_files = [f for f in files if f.endswith('.parquet')]
    print(f"Found {len(parquet_files)} parquet files")
    
    # Generate URLs and create dataframe
    url_data = []
    for file_path in parquet_files:
        url = hf_hub_url(repo_id=dataset_name, filename=file_path, repo_type="dataset")
        
        # Create a custom filename (remove path separators)
        custom_filename = file_path.replace('/', '_')
        
        url_data.append({
            'parquet_url': url,
            'original_path': file_path,
            'custom_filename': custom_filename,
            'dataset_name': dataset_name
        })
    
    # Create DataFrame and save
    df = pd.DataFrame(url_data)
    df.to_parquet(output_file, index=False)
    
    print(f"Created URL parquet file: {output_file}")
    print(f"Columns: {list(df.columns)}")
    print(f"Sample URLs:")
    for i, url in enumerate(df['parquet_url'].head(3)):
        print(f"  {i+1}. {url}")
    
    return output_file

if __name__ == '__main__':
    # Example usage
    dataset = "zkdeng/spiderTraining1000-1000"
    output_file = create_url_parquet(dataset, "spider_urls.parquet")
    
    print(f"\nTo download these files, use:")
    print(f"python asyncParquet.py --config parquet_download_config.json")
    print(f"\nMake sure your config JSON has:")
    print(f'  "input_parquet": "{output_file}"')
    print(f'  "url_column": "parquet_url"')
