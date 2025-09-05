import ndcctools.taskvine as vine
import json
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser(
        description="TaskVine script for distributed parquet file downloading",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Example usage:
  python TaskvineHF.py --config_file config/taskvine_config.json
  
JSON config file should contain:
  {
    "port_number": 9123,
    "parquets_directory": "input_parquets/",
    "url_col": "photo_url",
    "download_script": "paperTests/asyncParquet.py",
    "concurrent_downloads": 3,
    "timeout": 300,
    "output_directory": "downloaded_parquets"
  }"""
    )
    parser.add_argument(
        "--config_file", 
        type=str,
        default="data/json/taskvine.json",
        help="Path to the configuration JSON file.")
    return parser.parse_args()

def parse_json_config(file_path):
    """
    Load and parse the JSON configuration file.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Configuration file not found: {file_path}")
    
    with open(file_path, 'r') as config_file:
        return json.load(config_file)

def declare_parquet_files(manager, directory):
    """
    Declare input parquet files to TaskVine manager.
    """
    parquet_files = [os.path.join(directory, filename) for filename in os.listdir(directory) if filename.endswith(".parquet")]
    declared_files = {}

    # Loop over each parquet file and declare it
    for parquet_file in parquet_files:
        # Extract the file name without the directory and extension
        file_name = os.path.basename(parquet_file)
        
        # Declare the file with the manager
        declared_file = manager.declare_file(parquet_file)
        
        # Store the declared file in the dictionary with the file name as the key
        declared_files[file_name] = declared_file
    
    return declared_files

def declare_output_files(manager, directory, output_directory="downloaded_parquets"):
    """
    Declare output directory for downloaded parquet files for TaskVine manager.
    """
    parquet_files = [os.path.join(directory, filename) for filename in os.listdir(directory) if filename.endswith(".parquet")]
    declared_files = {}

    # Loop over each input parquet file and declare corresponding output directory
    for parquet_file in parquet_files:
        # Extract the file name without the directory and extension
        file_name = os.path.basename(parquet_file)
        base_name = os.path.splitext(file_name)[0]
        
        # Create output directory name based on input file
        output_dir = f"{output_directory}_{base_name}"
        
        # Declare the output directory with the manager
        declared_file = manager.declare_file(output_dir, cache=False)
        
        # Store the declared file in the dictionary with the file name as the key
        declared_files[file_name] = declared_file
    
    return declared_files

def create_json_config_for_task(file_name, url_col, output_directory="downloaded_parquets", concurrent_downloads=3, timeout=300):
    """
    Create a JSON configuration for asyncParquet.py task.
    """
    base_name = os.path.splitext(file_name)[0]
    config = {
        "input_parquet": file_name,
        "url_column": url_col,
        "output_directory": f"{output_directory}_{base_name}",
        "concurrent_downloads": concurrent_downloads,
        "timeout": timeout
    }
    
    config_filename = f"config_{base_name}.json"
    return config, config_filename

def submit_tasks(manager, download_script, parquet_files, output_files, configs):
    """
    Submit download tasks to TaskVine using asyncParquet.py.
    """
    download_script_vine = manager.declare_file(download_script)
    
    # Get configuration parameters
    url_col = configs['url_col']
    concurrent_downloads = configs.get('concurrent_downloads', 3)
    timeout = configs.get('timeout', 300)
    output_directory = configs.get('output_directory', 'downloaded_parquets')

    for file_name, declared_file in parquet_files.items():
        # Create JSON config for this task
        config_data, config_filename = create_json_config_for_task(
            file_name, url_col, output_directory, concurrent_downloads, timeout
        )
        
        # Write config to temporary file
        with open(config_filename, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        # Declare the config file
        config_file_vine = manager.declare_file(config_filename)
        
        # Create the TaskVine task
        download_task = vine.Task(
            f'python {download_script} --config {config_filename}'
        )
        
        inputs = declared_file
        outputs = output_files[file_name]
        
        # Add inputs and outputs to the task
        download_task.add_input(download_script_vine, os.path.basename(download_script))
        download_task.add_input(config_file_vine, config_filename)
        download_task.add_input(inputs, file_name)
        download_task.add_output(outputs, config_data['output_directory'])

        # Submit the task to the manager
        manager.submit(download_task)
        
        print(f"Submitted task for {file_name} with config {config_filename}")

def validate_config(configs):
    """
    Validate the configuration file has all required fields.
    """
    required_fields = ['port_number', 'parquets_directory', 'url_col']
    optional_fields = {
        'download_script': 'paperTests/asyncParquet.py',
        'concurrent_downloads': 3,
        'timeout': 300,
        'output_directory': 'downloaded_parquets'
    }
    
    # Check required fields
    for field in required_fields:
        if field not in configs:
            raise ValueError(f"Required field '{field}' not found in configuration.")
    
    # Add defaults for optional fields
    for field, default_value in optional_fields.items():
        if field not in configs:
            configs[field] = default_value
    
    # Validate paths exist
    if not os.path.exists(configs['parquets_directory']):
        raise FileNotFoundError(f"Parquets directory not found: {configs['parquets_directory']}")
    
    if not os.path.exists(configs['download_script']):
        raise FileNotFoundError(f"Download script not found: {configs['download_script']}")
    
    return configs

def main():
    # Parse command-line arguments
    args = parse_args()

    # Load and validate the configuration file
    configs = parse_json_config(args.config_file)
    configs = validate_config(configs)
    
    print(f"Configuration loaded:")
    print(f"  Parquets directory: {configs['parquets_directory']}")
    print(f"  Download script: {configs['download_script']}")
    print(f"  URL column: {configs['url_col']}")
    print(f"  Concurrent downloads: {configs['concurrent_downloads']}")
    print(f"  Output directory: {configs['output_directory']}")

    # Initialize the TaskVine manager
    manager = vine.Manager(configs['port_number'])
    print(f'\nTaskVine Manager listening on port {manager.port}')

    # Declare the input and output files to TaskVine
    directory = configs['parquets_directory']
    output_directory = configs['output_directory']
    
    parquet_files = declare_parquet_files(manager, directory)
    output_files = declare_output_files(manager, directory, output_directory)
    
    print(f"\nFound {len(parquet_files)} parquet files to process")

    # Submit the tasks
    submit_tasks(manager, configs['download_script'], parquet_files, output_files, configs)

    # Wait for tasks to complete
    print("\nWaiting for tasks to complete...")
    completed_tasks = 0
    total_tasks = len(parquet_files)
    
    while not manager.empty():
        task = manager.wait(5)
        if task:
            completed_tasks += 1
            if task.result == 0:
                print(f"✓ Task {task.id} completed successfully ({completed_tasks}/{total_tasks})")
            else:
                print(f"✗ Task {task.id} failed with exit code {task.result} ({completed_tasks}/{total_tasks})")
                if task.output:
                    print(f"  Output: {task.output[:200]}...")  # Show first 200 chars

    print(f"\nAll {total_tasks} tasks completed!")

if __name__ == '__main__':
    main()
