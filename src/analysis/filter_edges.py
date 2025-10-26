"""
Filter edges based on similarity threshold.

This script loads all exhaustive predictions from CSV files and filters edges
based on a similarity threshold determined from the ground truth labels.
"""

import os
import glob
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch


def load_all_predictions(input_file):
    """
    Load all prediction CSV files from a directory.
    
    Args:
        directory (str): Directory containing prediction CSV files
        
    Returns:
        pd.DataFrame: Combined DataFrame of all predictions
    """
    print(f"Loading predictions from {input_file}...")
    arr = torch.load(input_file, weights_only=False).cpu().numpy()
    df = pd.DataFrame(arr, columns=['src', 'tgt', 'similarity', 'label', 'edge'])
    return df

def determine_threshold(df):
    """
    Determine the similarity threshold based on ground truth labels.
    
    Args:
        df (pd.DataFrame): DataFrame with predictions
        
    Returns:
        float: Minimum similarity value for positive labels
    """
    # Filter rows where label is 1 (positive)
    ground_truth_df = df[df['label'] == 1]
    
    if ground_truth_df.empty:
        raise ValueError("No positive labels found in the data")
    
    # Find the similarity value that covers 75% of positive labels
    threshold = ground_truth_df['similarity'].quantile(0.15)  # 90% of values are above this
    print(f"Determined threshold: {threshold:.6f}")
    return threshold

def filter_edges(df, threshold):
    """
    Filter edges based on similarity threshold.
    
    Args:
        df (pd.DataFrame): DataFrame with predictions
        threshold (float): Similarity threshold
        
    Returns:
        pd.DataFrame: Filtered DataFrame with edges above threshold
    """
    # Keep only edges with similarity >= threshold
    filtered_df = df[df['similarity'] >= threshold].copy()
    
    # Add edge column (1 = True)
    filtered_df['edge'] = 1

    # If label is 1, then edge should be 1
    filtered_df.loc[filtered_df['label'] == 1, 'edge'] = 1
    
    print(f"Filtered {len(filtered_df):,} edges from {len(df):,} predictions")
    return filtered_df

def save_filtered_edges(df, output_path):
    """
    Save filtered edges to CSV file.
    
    Args:
        df (pd.DataFrame): DataFrame with filtered edges
        output_path (str): Path to save the filtered edges
    """
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df):,} filtered edges to {output_path}")

def main(input_file="data/graph/all_exhaustive_predictions.pt", output_path=None):
    """
    Main function to load, filter, and save edges.
    
    Args:
        input_dir (str): Directory containing prediction CSV files
        output_path (str): Path to save the filtered edges
    """
    # Load all predictions
    all_predictions = load_all_predictions(input_file)
    
    # Determine threshold from ground truth
    threshold = determine_threshold(all_predictions)
    
    # Filter edges based on threshold
    filtered_edges = filter_edges(all_predictions, threshold)
    
    # Generate output path with threshold if not provided
    if output_path is None:
        output_path = f"data/graph/filtered_edges_threshold_{threshold:.3f}.csv"
    else:
        # Insert threshold into filename
        base, ext = os.path.splitext(output_path)
        output_path = f"{base}_threshold_{threshold:.2f}{ext}"
    
    # Save filtered edges
    save_filtered_edges(filtered_edges, output_path)
    
    return filtered_edges

if __name__ == "__main__":
    main(
        input_file="data/data/out/graph/all_exhaustive_predictions.pt",
        output_path="data/graph/filtered_edges.csv"
    )
