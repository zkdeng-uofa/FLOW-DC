#!/usr/bin/env python

import argparse
import pandas as pd
import numpy as np
import os
import sys
import json
from math import ceil
from dataclasses import dataclass, field
from enum import Enum

class GroupingMethod(Enum):
    """Enumeration of available grouping methods"""
    GREEDY = "greedy"
    SIMPLE = "simple"

@dataclass
class FilterArguments:
    """
    This class instantiates the user inputs of the script that they may input through command line or through a JSON
    """
    parquet: str = field(
        default = None,
        metadata = {"help": "name of input parquet"}
    )
    grouping_col: str = field(
        default = None,
        metadata = {"help": "name of col to group on (required for greedy method)"}
    )
    url_col: str = field(
        default = None,
        metadata = {"help": "name of URL column"}
    )
    groups: int = field(
        default = None,
        metadata = {"help": "number of groups to divide into"}
    )
    output_folder: str = field(
        default = None,
        metadata = {"help": "name of output folder"}
    )
    method: str = field(
        default = "greedy",
        metadata = {"help": "grouping method: 'greedy' or 'simple'"}
    )
    output_format: str = field(
        default = "parquet",
        metadata = {"help": "output format: 'parquet' or 'csv'"}
    )

def load_json_config(config_path):
    """
    Load configuration from JSON file and return as FilterArguments object.
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
    required_fields = ['parquet', 'url_col', 'groups', 'output_folder']
    conditional_required = {
        'greedy': ['grouping_col']  # grouping_col only required for greedy method
    }
    defaults = {
        'method': 'greedy',
        'output_format': 'parquet',
        'grouping_col': None
    }
    
    # Check required fields
    for field in required_fields:
        if field not in config_data:
            print(f"Error: Required field '{field}' not found in JSON configuration.")
            sys.exit(1)
    
    # Check conditional required fields based on method
    method = config_data.get('method', 'greedy')
    if method in conditional_required:
        for field in conditional_required[method]:
            if field not in config_data or config_data[field] is None:
                print(f"Error: Field '{field}' is required for method '{method}'.")
                sys.exit(1)
    
    # Apply defaults for missing optional fields
    for field, default_value in defaults.items():
        if field not in config_data:
            config_data[field] = default_value
    
    # Validate data types and values
    type_validators = {
        'parquet': str,
        'grouping_col': (str, type(None)),
        'url_col': str,
        'groups': int,
        'output_folder': str,
        'method': str,
        'output_format': str
    }
    
    for field, expected_type in type_validators.items():
        if field in config_data:
            if not isinstance(config_data[field], expected_type):
                print(f"Error: Field '{field}' must be of type {expected_type.__name__}.")
                sys.exit(1)
    
    # Validate method
    if config_data['method'] not in [m.value for m in GroupingMethod]:
        print(f"Error: Method must be one of: {[m.value for m in GroupingMethod]}")
        sys.exit(1)
    
    # Validate output format
    if config_data['output_format'] not in ['parquet', 'csv']:
        print(f"Error: Output format must be 'parquet' or 'csv'")
        sys.exit(1)
    
    # Validate groups is positive
    if config_data['groups'] <= 0:
        print(f"Error: Number of groups must be positive")
        sys.exit(1)
    
    return FilterArguments(**config_data)

def parse_args() -> FilterArguments:
    """
    Parse user input arguments from command line
    """
    parser = argparse.ArgumentParser(
        description="Split parquet files into groups using different algorithms",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  # Using command line arguments (simple method):
  python SplitParquet.py --parquet data.parquet --url_col photo_url --groups 5 --output_folder output --method simple
  
  # Using command line arguments (greedy method):
  python SplitParquet.py --parquet data.parquet --url_col photo_url --grouping_col species --groups 5 --output_folder output --method greedy
  
  # Using JSON configuration:
  python SplitParquet.py --config config.json"""
    )
    
    # JSON configuration option
    parser.add_argument(
        '--config',
        type=str,
        help="Path to JSON configuration file. If specified, other arguments are ignored."
    )
    
    parser.add_argument(
        '--parquet', 
        type=str, 
        help="name of input parquet"
    )
    parser.add_argument(
        '--grouping_col',
        type=str,
        help="name of col to group on (required for greedy method, ignored for simple method)"
    )
    parser.add_argument(
        '--url_col',
        type=str,
        help="name of URL column"
    )
    parser.add_argument(
        '--groups', 
        type=int, 
        help="number of groups to divide into"
    )
    parser.add_argument(
        '--output_folder', 
        type=str, 
        help="name of output folder"
    )
    parser.add_argument(
        '--method',
        type=str,
        choices=[m.value for m in GroupingMethod],
        default='greedy',
        help="grouping method: 'greedy' (balance by count) or 'simple' (equal rows per group)"
    )
    parser.add_argument(
        '--output_format',
        type=str,
        choices=['parquet', 'csv'],
        default='parquet',
        help="output file format"
    )
    
    args = parser.parse_args()
    
    # If JSON config is specified, load it and ignore other arguments
    if args.config:
        return load_json_config(args.config)
    
    # Validate required arguments when not using JSON config
    required_args = ['parquet', 'url_col', 'groups', 'output_folder']
    missing_args = [arg for arg in required_args if getattr(args, arg) is None]
    
    if missing_args:
        parser.error(f"The following arguments are required when not using --config: {', '.join('--' + arg for arg in missing_args)}")
    
    # Check method-specific requirements
    if args.method == 'greedy' and args.grouping_col is None:
        parser.error("--grouping_col is required when using greedy method")
    
    return FilterArguments(
        parquet=args.parquet,
        grouping_col=args.grouping_col,
        url_col=args.url_col,
        groups=args.groups, 
        output_folder=args.output_folder,
        method=args.method,
        output_format=args.output_format
    )

def greedy_grouping(num_partitions, df, count, name):
    """
    Performs greedy grouping given a dataframe and number of partitions.
    """
    # Sort the dataframe by count in descending order to handle larger groups first
    sorted_df = df.sort_values(by=count, ascending=False).reset_index(drop=True)

    # Initialize partition data structures
    partitions = [[] for _ in range(num_partitions)]
    partition_sums = [0 for _ in range(num_partitions)]
    
    for _, row in sorted_df.iterrows():
        # Find the partition with the minimum sum that can accommodate the current row
        min_partition_idx = np.argmin(partition_sums)
        partitions[min_partition_idx].append(row)
        partition_sums[min_partition_idx] += row[count]

    # Flatten the partitions and create a new dataframe
    new_rows = []
    for group_id, partition in enumerate(partitions, 1):
        for row in partition:
            new_rows.append({name: row[name], count: row[count], "group": group_id})

    output_df = pd.DataFrame(new_rows)

    return output_df

def simple_grouping(df, num_partitions):
    """
    Simple grouping that divides dataframe into equal-sized groups by row count.
    Last group may have fewer rows if total rows don't divide evenly.
    """
    total_rows = len(df)
    rows_per_group = total_rows // num_partitions
    remainder = total_rows % num_partitions
    
    # Create a copy to avoid modifying original
    df_copy = df.copy().reset_index(drop=True)
    
    # Assign group numbers
    group_assignments = []
    current_row = 0
    
    for group_id in range(1, num_partitions + 1):
        # Calculate group size (some groups get +1 row if there's remainder)
        group_size = rows_per_group + (1 if group_id <= remainder else 0)
        
        # Assign group number to these rows
        for _ in range(group_size):
            if current_row < total_rows:
                group_assignments.append(group_id)
                current_row += 1
    
    df_copy['group'] = group_assignments
    
    return df_copy

def partition_df(df, num_partitions, taxon_col):
    """
    Legacy function - partitions by taxon column (kept for backwards compatibility)
    """
    # Calculate the approximate size for each partition
    partition_size = ceil(len(df) / num_partitions)

    # Sort the dataframe by the taxon column
    sorted_df = df.sort_values(by=taxon_col).reset_index(drop=True)

    # Assign each row to a partition
    sorted_df['group_num'] = ((np.arange(len(sorted_df)) // partition_size) + 1)

    # Concatenate the group number to the taxon name
    sorted_df[taxon_col] = sorted_df[taxon_col] + "_" + sorted_df['group_num'].astype(str)

    # Drop the temporary 'group_num' column
    sorted_df = sorted_df.drop(columns=['group_num'])

    return sorted_df

def save_group_file(df, group_id, output_folder, output_format):
    """
    Save a group dataframe to file in specified format.
    """
    filename = f"group_{group_id}"
    
    if output_format == 'parquet':
        filepath = f"{output_folder}/{filename}.parquet"
        df.to_parquet(filepath, index=False)
    elif output_format == 'csv':
        filepath = f"{output_folder}/{filename}.csv"
        df.to_csv(filepath, index=False)
    
    return filepath

def print_grouping_summary(grouped_df, method):
    """
    Print summary statistics about the grouping.
    """
    group_stats = grouped_df.groupby('group').size().reset_index(name='count')
    
    print(f"\nGrouping Summary ({method} method):")
    print(f"  Total rows: {len(grouped_df)}")
    print(f"  Number of groups: {len(group_stats)}")
    print(f"  Rows per group:")
    
    for _, row in group_stats.iterrows():
        print(f"    Group {row['group']}: {row['count']} rows")
    
    print(f"  Min/Max/Mean rows per group: {group_stats['count'].min()}/{group_stats['count'].max()}/{group_stats['count'].mean():.1f}")

def main():
    inputs = parse_args()
    
    # Display configuration
    print(f"Configuration:")
    print(f"  Input parquet: {inputs.parquet}")
    print(f"  URL column: {inputs.url_col}")
    if inputs.method == 'greedy':
        print(f"  Grouping column: {inputs.grouping_col}")
    else:
        print(f"  Grouping column: Not used (simple method)")
    print(f"  Number of groups: {inputs.groups}")
    print(f"  Output folder: {inputs.output_folder}")
    print(f"  Grouping method: {inputs.method}")
    print(f"  Output format: {inputs.output_format}")
    
    # Validate input file
    if not os.path.exists(inputs.parquet):
        print(f"Error: Input parquet file '{inputs.parquet}' not found.")
        sys.exit(1)
    
    # Load data
    print(f"\nLoading data from {inputs.parquet}...")
    try:
        total_df = pd.read_parquet(inputs.parquet)
    except Exception as e:
        print(f"Error reading parquet file: {e}")
        sys.exit(1)
    
    print(f"Loaded {len(total_df)} rows with {len(total_df.columns)} columns")
    
    # Validate URL column exists
    if inputs.url_col not in total_df.columns:
        print(f"Error: URL column '{inputs.url_col}' not found in parquet file.")
        print(f"Available columns: {list(total_df.columns)}")
        sys.exit(1)
    
    # Validate grouping column exists (only for greedy method)
    if inputs.method == 'greedy':
        if inputs.grouping_col not in total_df.columns:
            print(f"Error: Grouping column '{inputs.grouping_col}' not found in parquet file.")
            print(f"Available columns: {list(total_df.columns)}")
            sys.exit(1)
    
    # Validate number of groups
    if inputs.groups > len(total_df):
        print(f"Warning: Number of groups ({inputs.groups}) is greater than number of rows ({len(total_df)}). Some groups will be empty.")
    
    # Apply grouping method
    print(f"\nApplying {inputs.method} grouping...")
    
    if inputs.method == GroupingMethod.SIMPLE.value:
        # Simple grouping - equal rows per group
        total_df_grouped = simple_grouping(total_df, inputs.groups)
        
    elif inputs.method == GroupingMethod.GREEDY.value:
        # Greedy grouping - balance by grouping column counts
        # First, modify the taxon column to create partitions
        total_df = partition_df(total_df, inputs.groups, inputs.grouping_col)
        
        # Group by specific row and count
        count_df = total_df.groupby(inputs.grouping_col).size().reset_index(name="Count").sort_values(by='Count', ascending=False)
        groups_df = greedy_grouping(inputs.groups, count_df, "Count", inputs.grouping_col)
        
        # Merge total_df with groups_df
        total_df = total_df.drop(columns=['group'], errors='ignore')
        total_df_grouped = total_df.merge(groups_df[[inputs.grouping_col, "group"]], on=inputs.grouping_col, how="left")
    
    else:
        print(f"Error: Unknown grouping method '{inputs.method}'")
        sys.exit(1)
    
    # Print summary
    print_grouping_summary(total_df_grouped, inputs.method)
    
    # Create output directory
    os.makedirs(inputs.output_folder, exist_ok=True)
    print(f"\nSaving groups to {inputs.output_folder}/...")
    
    # Save groups to files
    saved_files = []
    for group in sorted(total_df_grouped["group"].unique()):
        subset_df = total_df_grouped[total_df_grouped["group"] == group]
        
        # Remove the group column from output to match original format
        output_df = subset_df.drop(columns=['group'])
        
        filepath = save_group_file(output_df, group, inputs.output_folder, inputs.output_format)
        saved_files.append(filepath)
        print(f"  Saved {filepath} ({len(output_df)} rows)")
    
    print(f"\nCompleted! Created {len(saved_files)} group files in {inputs.output_folder}/")

if __name__ == "__main__":
    main()