"""
Check for ID overlaps in triplet datasets.

This script loads triplet datasets and checks for overlaps between anchor, positive,
and negative IDs across different splits.
"""

import json
import os
from collections import defaultdict

def load_triplets(file_path):
    """
    Load triplets from a JSONL file.
    
    Args:
        file_path (str): Path to the triplets JSONL file
        
    Returns:
        list: List of triplet dictionaries
    """
    triplets = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            triplets.append(json.loads(line.strip()))
    return triplets

def extract_all_ids(triplets):
    """
    Extract all IDs (anchor, positive, negatives) from triplets.
    
    Args:
        triplets (list): List of triplet dictionaries
        
    Returns:
        set: Set of all IDs
    """
    all_ids = set()
    for triplet in triplets:
        all_ids.add(triplet['anchor'])
        all_ids.add(triplet['positive'])
        all_ids.update(triplet['negatives'])
    return all_ids

def check_overlap(triplets_files):
    """
    Check for ID overlaps between different triplet files.
    
    Args:
        triplets_files (list): List of paths to triplet files
        
    Returns:
        dict: Dictionary with overlap statistics
    """
    # Load all triplets
    all_triplets = {}
    all_ids_by_file = {}
    
    for file_path in triplets_files:
        file_name = os.path.basename(file_path)
        triplets = load_triplets(file_path)
        all_triplets[file_name] = triplets
        all_ids_by_file[file_name] = extract_all_ids(triplets)
        print(f"Loaded {len(triplets)} triplets from {file_name} with {len(all_ids_by_file[file_name])} unique IDs")
    
    # Check overlaps
    overlaps = {}
    for i, (file1, ids1) in enumerate(all_ids_by_file.items()):
        for j, (file2, ids2) in enumerate(all_ids_by_file.items()):
            if i >= j:  # Skip duplicate comparisons and self-comparisons
                continue
                
            overlap = ids1.intersection(ids2)
            overlap_size = len(overlap)
            overlap_percentage = (overlap_size / len(ids1)) * 100
            
            key = f"{file1} vs {file2}"
            overlaps[key] = {
                "overlap_count": overlap_size,
                "overlap_percentage": f"{overlap_percentage:.2f}%",
                "sample_overlapping_ids": list(overlap)[:10] if overlap else []
            }
            
            print(f"Overlap between {file1} and {file2}: {overlap_size} IDs ({overlap_percentage:.2f}%)")
            if overlap:
                print(f"  Sample overlapping IDs: {', '.join(list(overlap)[:10])}")
    
    return overlaps

def main():
    # Paths to triplet files
    triplets_files = [
        "data/data/split_triplets/triplets_train.jsonl",
        "data/data/split_triplets/triplets_val.jsonl",
        "data/data/split_triplets/triplets_test.jsonl"
    ]
    
    # Check if files exist
    for file_path in triplets_files:
        if not os.path.exists(file_path):
            print(f"Warning: File {file_path} does not exist")
    
    # Check overlaps
    overlaps = check_overlap(triplets_files)
    
    # Print summary
    print("\nOverlap Summary:")
    for comparison, stats in overlaps.items():
        print(f"{comparison}: {stats['overlap_count']} IDs ({stats['overlap_percentage']})")

if __name__ == "__main__":
    main()
