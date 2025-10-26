"""
Compare edges_list_th0.6713.csv (filtered by 0.94) with filtered_edges_threshold_0.94.csv

This script checks if filtering edges_list_th0.6713 by similarity >= 0.94 produces
the same edge set as filtered_edges_threshold_0.94.csv, accounting for bidirectional edges.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def normalize_edge_set(df, threshold=None):
    """
    Normalize edges to account for bidirectionality.
    For each edge (src, tgt), store it as (min(src,tgt), max(src,tgt)) with similarity.
    
    Args:
        df: DataFrame with columns ['src', 'tgt', 'similarity']
        threshold: Optional similarity threshold to filter by
        
    Returns:
        Dict mapping (min_id, max_id) -> similarity
    """
    # Filter by threshold if provided
    if threshold is not None:
        df = df[df['similarity'] >= threshold].copy()
        print(f"After filtering by threshold >= {threshold}: {len(df)} edges")
    
    # Normalize edges: ensure src <= tgt for consistent comparison
    normalized_edges = {}
    for _, row in df.iterrows():
        src = row['src']
        tgt = row['tgt']
        sim = row['similarity']
        
        # Handle potential float IDs
        if isinstance(src, float):
            src = int(src) if not pd.isna(src) else src
        if isinstance(tgt, float):
            tgt = int(tgt) if not pd.isna(tgt) else tgt
            
        # Normalize: smaller ID first
        min_id = min(src, tgt)
        max_id = max(src, tgt)
        
        normalized_edges[(min_id, max_id)] = float(sim)
    
    return normalized_edges


def compare_edge_files(file1_path, file2_path, threshold=0.94, sim_tolerance=1e-5):
    """
    Compare two edge files, filtering file1 by threshold.
    
    Args:
        file1_path: Path to edges_list_th0.6713.csv
        file2_path: Path to filtered_edges_threshold_0.94.csv
        threshold: Similarity threshold to apply to file1
        sim_tolerance: Tolerance for similarity comparison (default: 1e-5)
    """
    print("=" * 80)
    print("EDGE FILE COMPARISON")
    print("=" * 80)
    print(f"\nFile 1: {file1_path}")
    print(f"File 2: {file2_path}")
    print(f"Threshold: {threshold}")
    print(f"Similarity tolerance: {sim_tolerance}")
    print()
    
    # Load file 1 (edges_list_th0.6713.csv)
    print(f"Loading {file1_path}...")
    df1 = pd.read_csv(file1_path)
    print(f"  Columns: {df1.columns.tolist()}")
    print(f"  Total edges: {len(df1):,}")
    print(f"  Similarity range: [{df1['similarity'].min():.6f}, {df1['similarity'].max():.6f}]")
    
    # Load file 2 (filtered_edges_threshold_0.94.csv)
    print(f"\nLoading {file2_path}...")
    df2 = pd.read_csv(file2_path)
    print(f"  Columns: {df2.columns.tolist()}")
    print(f"  Total edges: {len(df2):,}")
    print(f"  Similarity range: [{df2['similarity'].min():.6f}, {df2['similarity'].max():.6f}]")
    
    # Normalize both edge sets
    print(f"\n{'='*80}")
    print("NORMALIZING EDGE SETS (accounting for bidirectionality)")
    print(f"{'='*80}\n")
    
    print("Processing File 1 (with threshold filter)...")
    edges1 = normalize_edge_set(df1, threshold=threshold)
    print(f"File 1 normalized edges: {len(edges1):,}")
    
    print("\nProcessing File 2 (no additional filter)...")
    edges2 = normalize_edge_set(df2, threshold=None)
    print(f"File 2 normalized edges: {len(edges2):,}")
    
    # Compare the edge sets with tolerance
    print(f"\n{'='*80}")
    print("COMPARISON RESULTS (with similarity tolerance)")
    print(f"{'='*80}\n")
    
    # Find edges in common and differences
    edges1_keys = set(edges1.keys())
    edges2_keys = set(edges2.keys())
    
    # Edges only by node pairs
    only_in_file1_nodes = edges1_keys - edges2_keys
    only_in_file2_nodes = edges2_keys - edges1_keys
    common_nodes = edges1_keys & edges2_keys
    
    print(f"Common edge pairs (by nodes): {len(common_nodes):,}")
    print(f"Edge pairs only in File 1: {len(only_in_file1_nodes):,}")
    print(f"Edge pairs only in File 2: {len(only_in_file2_nodes):,}")
    
    # Among common edges, check similarity differences
    identical_edges = 0
    similar_edges = 0  # Within tolerance but not identical
    different_edges = 0  # Outside tolerance
    sim_differences = []
    
    for edge_pair in common_nodes:
        sim1 = edges1[edge_pair]
        sim2 = edges2[edge_pair]
        diff = abs(sim1 - sim2)
        
        if diff == 0:
            identical_edges += 1
        elif diff < sim_tolerance:
            similar_edges += 1
            sim_differences.append((edge_pair, sim1, sim2, diff))
        else:
            different_edges += 1
            sim_differences.append((edge_pair, sim1, sim2, diff))
    
    print(f"\nAmong {len(common_nodes):,} common edges:")
    print(f"  - Identical similarity: {identical_edges:,}")
    print(f"  - Similar within tolerance ({sim_tolerance}): {similar_edges:,}")
    print(f"  - Different (beyond tolerance): {different_edges:,}")
    
    # Overall comparison
    total_matching = identical_edges + similar_edges
    total_edges = len(edges1_keys | edges2_keys)
    
    print(f"\nTotal unique edges (union): {total_edges:,}")
    print(f"Matching edges (within tolerance): {total_matching:,}")
    print(f"Non-matching edges: {len(only_in_file1_nodes) + len(only_in_file2_nodes) + different_edges:,}")
    
    # Calculate similarity metrics
    if total_edges > 0:
        match_rate = total_matching / total_edges * 100
        print(f"\nMatch rate: {match_rate:.2f}%")
    
    # Determine if identical within tolerance
    print(f"\n{'='*80}")
    if len(only_in_file1_nodes) == 0 and len(only_in_file2_nodes) == 0 and different_edges == 0:
        print("✅ FILES ARE IDENTICAL (within tolerance)")
    else:
        print("❌ FILES ARE NOT IDENTICAL")
    print(f"{'='*80}\n")
    
    # Show sample differences if not identical
    if len(only_in_file1_nodes) > 0:
        print("\nSample edges ONLY in File 1 (first 10):")
        for i, edge_pair in enumerate(sorted(list(only_in_file1_nodes))[:10]):
            sim = edges1[edge_pair]
            print(f"  {i+1}. src={edge_pair[0]:05d}, tgt={edge_pair[1]:05d}, similarity={sim:.6f}")
    
    if len(only_in_file2_nodes) > 0:
        print("\nSample edges ONLY in File 2 (first 10):")
        for i, edge_pair in enumerate(sorted(list(only_in_file2_nodes))[:10]):
            sim = edges2[edge_pair]
            print(f"  {i+1}. src={edge_pair[0]:05d}, tgt={edge_pair[1]:05d}, similarity={sim:.6f}")
    
    # Show edges with different similarities (beyond tolerance)
    if different_edges > 0:
        print("\nSample edges with DIFFERENT similarities (beyond tolerance, first 10):")
        diff_samples = [x for x in sim_differences if x[3] >= sim_tolerance][:10]
        for i, (edge_pair, sim1, sim2, diff) in enumerate(diff_samples):
            print(f"  {i+1}. src={edge_pair[0]:05d}, tgt={edge_pair[1]:05d}")
            print(f"      File1: {sim1:.8f}, File2: {sim2:.8f}, diff={diff:.8f}")
    
    # Show edges with similar similarities (within tolerance but not identical)
    if similar_edges > 0 and similar_edges <= 20:
        print(f"\nEdges with SIMILAR similarities (within tolerance, showing all {similar_edges}):")
        sim_samples = [x for x in sim_differences if 0 < x[3] < sim_tolerance]
        for i, (edge_pair, sim1, sim2, diff) in enumerate(sim_samples):
            print(f"  {i+1}. src={edge_pair[0]:05d}, tgt={edge_pair[1]:05d}")
            print(f"      File1: {sim1:.8f}, File2: {sim2:.8f}, diff={diff:.8f}")
    elif similar_edges > 20:
        print(f"\nEdges with SIMILAR similarities (within tolerance, first 10 of {similar_edges}):")
        sim_samples = [x for x in sim_differences if 0 < x[3] < sim_tolerance][:10]
        for i, (edge_pair, sim1, sim2, diff) in enumerate(sim_samples):
            print(f"  {i+1}. src={edge_pair[0]:05d}, tgt={edge_pair[1]:05d}")
            print(f"      File1: {sim1:.8f}, File2: {sim2:.8f}, diff={diff:.8f}")
    
    # Additional statistics
    print(f"\n{'='*80}")
    print("ADDITIONAL STATISTICS")
    print(f"{'='*80}\n")
    
    # Check if any edges in file2 have similarity < threshold
    below_threshold = df2[df2['similarity'] < threshold]
    if len(below_threshold) > 0:
        print(f"⚠️  WARNING: File 2 contains {len(below_threshold)} edges with similarity < {threshold}")
        print(f"   Similarity range of these edges: [{below_threshold['similarity'].min():.6f}, {below_threshold['similarity'].max():.6f}]")
    else:
        print(f"✅ All edges in File 2 have similarity >= {threshold}")
    
    # Check for duplicate edges in original files
    print("\nChecking for duplicate edges (before normalization):")
    
    df1_pairs = df1[['src', 'tgt']].apply(lambda x: tuple(sorted([x['src'], x['tgt']])), axis=1)
    df1_duplicates = df1_pairs.duplicated().sum()
    print(f"  File 1 duplicates: {df1_duplicates:,}")
    
    df2_pairs = df2[['src', 'tgt']].apply(lambda x: tuple(sorted([x['src'], x['tgt']])), axis=1)
    df2_duplicates = df2_pairs.duplicated().sum()
    print(f"  File 2 duplicates: {df2_duplicates:,}")
    
    return {
        'identical': len(only_in_file1_nodes) == 0 and len(only_in_file2_nodes) == 0 and different_edges == 0,
        'common_edges': len(common_nodes),
        'identical_sim': identical_edges,
        'similar_sim': similar_edges,
        'different_sim': different_edges,
        'only_file1': len(only_in_file1_nodes),
        'only_file2': len(only_in_file2_nodes),
        'match_rate': match_rate if total_edges > 0 else 0,
    }


if __name__ == "__main__":
    # File paths
    base_dir = Path("/home/zhaomin/project/wikidbs/data/graph")
    file1 = base_dir / "edges_list_th0.6713.csv"
    file2 = base_dir / "filtered_edges_threshold_0.94.csv"
    
    # Run comparison
    results = compare_edge_files(file1, file2, threshold=0.943601, sim_tolerance=1e-5)
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Files Identical (within tolerance): {results['identical']}")
    print(f"Common edges: {results['common_edges']:,}")
    print(f"  - Identical similarity: {results['identical_sim']:,}")
    print(f"  - Similar (within tolerance): {results['similar_sim']:,}")
    print(f"  - Different (beyond tolerance): {results['different_sim']:,}")
    print(f"Edges only in File 1: {results['only_file1']:,}")
    print(f"Edges only in File 2: {results['only_file2']:,}")
    print(f"Match rate: {results['match_rate']:.2f}%")
    print("=" * 80)

