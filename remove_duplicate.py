"""
Remove duplicate edges from filtered_edges CSV files.

This script processes filtered_edges_threshold_*.csv files from an input folder,
detects and removes duplicate edges (treating the graph as undirected), and
outputs cleaned files to an output folder.
"""

import os
import glob
import pandas as pd


def normalize_edge(src, tgt):
    """
    Return canonical edge representation for undirected graph.
    
    Args:
        src: Source node ID
        tgt: Target node ID
        
    Returns:
        tuple: (min, max) pair representing the edge
    """
    return (min(src, tgt), max(src, tgt))


def find_duplicates(df, tolerance=1e-6):
    """
    Identify duplicate edges and calculate similarity differences.
    
    Args:
        df (pd.DataFrame): DataFrame with edge data
        tolerance (float): Tolerance for considering similarities as equal
        
    Returns:
        tuple: (duplicates_to_remove, errors)
            - duplicates_to_remove: List of indices to remove
            - errors: List of (edge, max_diff) tuples for edges with diff >= tolerance
    """
    # Create normalized edge column
    df_copy = df.copy()
    df_copy['normalized_edge'] = df_copy.apply(
        lambda row: normalize_edge(row['src'], row['tgt']), axis=1
    )
    
    # Group by normalized edge
    grouped = df_copy.groupby('normalized_edge')
    
    duplicates_to_remove = []
    errors = []
    
    for edge, group in grouped:
        if len(group) > 1:
            # Calculate max similarity difference
            similarities = group['similarity'].values
            max_diff = similarities.max() - similarities.min()
            
            if max_diff >= tolerance:
                # Report as error
                errors.append((edge, max_diff, len(group)))
            else:
                # Mark all but first occurrence for removal
                indices = group.index.tolist()
                duplicates_to_remove.extend(indices[1:])  # Keep first, remove rest
    
    return duplicates_to_remove, errors


def remove_duplicates(df, tolerance=1e-6):
    """
    Remove duplicate edges within tolerance and report errors.
    
    Args:
        df (pd.DataFrame): DataFrame with edge data
        tolerance (float): Tolerance for considering edges as duplicates
        
    Returns:
        pd.DataFrame: Cleaned DataFrame with duplicates removed
    """
    print(f"  Original edges: {len(df):,}")
    
    # Find duplicates
    duplicates_to_remove, errors = find_duplicates(df, tolerance)
    
    # Report findings
    if duplicates_to_remove:
        print(f"  Found {len(duplicates_to_remove):,} duplicate edges (within tolerance {tolerance})")
        print("  Keeping first occurrence, removing duplicates...")
    else:
        print(f"  No duplicates found within tolerance {tolerance}")
    
    # Report errors
    if errors:
        print(f"\n  ⚠️  ERROR: Found {len(errors)} edge(s) with duplicate entries having similarity difference >= {tolerance}:")
        for edge, max_diff, count in errors:
            print(f"    Edge {edge}: {count} occurrences, max similarity difference: {max_diff:.10f}")
        print("  These edges are kept in the output (not removed).\n")
    
    # Remove duplicates (only those within tolerance)
    df_cleaned = df.drop(duplicates_to_remove).reset_index(drop=True)
    
    print(f"  Final edges: {len(df_cleaned):,}")
    print(f"  Removed: {len(duplicates_to_remove):,} edges")
    
    return df_cleaned


def process_file(input_path, output_path, tolerance=1e-6):
    """
    Process a single CSV file to remove duplicates.
    
    Args:
        input_path (str): Path to input CSV file
        output_path (str): Path to output CSV file
        tolerance (float): Tolerance for duplicate detection
    """
    print(f"\nProcessing: {os.path.basename(input_path)}")
    
    # Load CSV
    df = pd.read_csv(input_path)
    
    # Remove duplicates
    df_cleaned = remove_duplicates(df, tolerance)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save cleaned data
    df_cleaned.to_csv(output_path, index=False)
    print(f"  Saved to: {output_path}")


def main(input_folder="data/graph.old", output_folder="data/graph", tolerance=1e-6):
    """
    Process all filtered_edges CSV files in the input folder.
    
    Args:
        input_folder (str): Directory containing input CSV files
        output_folder (str): Directory to save output CSV files
        tolerance (float): Tolerance for duplicate detection
    """
    print("=" * 80)
    print("Duplicate Edge Removal Script")
    print("=" * 80)
    print(f"Input folder: {input_folder}")
    print(f"Output folder: {output_folder}")
    print(f"Tolerance: {tolerance}")
    
    # Find all filtered_edges CSV files
    pattern = os.path.join(input_folder, "filtered_edges_threshold_*.csv")
    input_files = glob.glob(pattern)
    
    if not input_files:
        print(f"\n⚠️  No files matching pattern: {pattern}")
        return
    
    print(f"\nFound {len(input_files)} file(s) to process:")
    for f in input_files:
        print(f"  - {os.path.basename(f)}")
    
    # Process each file
    for input_path in sorted(input_files):
        filename = os.path.basename(input_path)
        output_path = os.path.join(output_folder, filename)
        
        try:
            process_file(input_path, output_path, tolerance)
        except Exception as e:  # noqa: BLE001
            print(f"\n❌ Error processing {filename}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("Processing complete!")
    print("=" * 80)


if __name__ == "__main__":
    main(
        input_folder="data/graph.old",
        output_folder="data/graph",
        tolerance=1e-6
    )

