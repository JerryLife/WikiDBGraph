import pandas as pd
import os
import numpy as np
from collections import defaultdict
import json
import argparse
from pathlib import Path

from analysis.NodeProperties import Database, Table, Column


def load_edge_properties():
    """Load edge structural properties from CSV file"""
    structural_props_path = "data/graph/filtered_edges_threshold_0.94.csv"
    if os.path.exists(structural_props_path):
        return pd.read_csv(structural_props_path)
    else:
        print(f"Warning: {structural_props_path} not found")
        return pd.DataFrame()


def load_cluster_assignments():
    """Load cluster assignments"""
    cluster_path = "data/graph/cluster_assignments_dim2_sz100_msNone.csv"
    if os.path.exists(cluster_path):
        return pd.read_csv(cluster_path)
    else:
        print(f"Warning: {cluster_path} not found")
        return pd.DataFrame()


def get_database_row_counts(folder: str):
    """Get the maximum number of rows in each database"""
    row_counts = {}
    
    tables_dir = os.path.join(folder, "tables")
    if not os.path.exists(tables_dir):
        return 0
        
    for table_file in os.listdir(tables_dir):
        if table_file.endswith('.csv'):
            table_name = table_file.split(".")[0]
            try:
                # Count rows excluding header
                df = pd.read_csv(os.path.join(tables_dir, table_file))
                row_counts[table_name] = len(df)
            except Exception as e:
                print(f"Warning: Could not read {table_file}: {e}")
                continue
    
    return max(row_counts.values()) if row_counts else 0


def get_database_abstract(db_id: int, folder_path: str):
    """
    Extract database abstract including table names, columns, and other metadata
    
    Args:
        db_id (int): Database ID
        folder_path (str): Path to database folder
        
    Returns:
        dict: Database abstract information
    """
    schema_path = os.path.join(folder_path, "schema.json")
    
    if not os.path.exists(schema_path):
        return {"error": "Schema file not found"}
    
    try:
        with open(schema_path, "r") as f:
            schema = json.load(f)
        
        db = Database(db_id)
        db.load_from_schema(schema)
        
        # Extract table information
        tables_info = []
        all_columns = set()
        
        for table in db.tables:
            table_info = {
                "table_name": table.table_name,
                "columns": [{"name": col.column_name, "type": col.data_type} 
                           for col in table.columns]
            }
            tables_info.append(table_info)
            
            # Collect all normalized column names
            for col in table.columns:
                normalized_col = ''.join(c for c in col.column_name.lower() if c.isalnum())
                all_columns.add(normalized_col)
        
        max_rows = get_database_row_counts(folder_path)
        return {
            "db_id": db_id,
            "wikidata_topic_item_id": schema.get('wikidata_topic_item_id'),
            "database_name": schema.get('database_name', ''),
            "num_tables": len(tables_info),
            "tables": tables_info,
            "all_columns": list(all_columns),
            "max_rows": max_rows
        }
        
    except Exception as e:
        return {"error": f"Error processing schema: {str(e)}"}


def collect_similar_pairs(k=20, max_similarity=1.0, min_rows=100, by_cluster=False):
    """
    Collect top k similar database pairs globally or by cluster
    
    Args:
        k (int): Number of similar pairs (total if by_cluster=False, per cluster if by_cluster=True)
        max_similarity (float): Maximum similarity threshold
        min_rows (int): Minimum number of rows required for each database
        by_cluster (bool): If True, collect k pairs per cluster; if False, collect k pairs globally
        
    Returns:
        dict: Dictionary mapping cluster_id (or 'global') to list of similar pairs
    """
    # Load data
    edge_props = load_edge_properties()
    
    if edge_props.empty:
        print("Required data files not found")
        return {}
    
    # Filter by maximum similarity
    edge_props = edge_props[edge_props['similarity'] <= max_similarity]
    
    # Create database ID to folder mapping
    UNZIP_DIR = os.path.join("data/unzip")
    db_id_to_folder = {}
    
    if os.path.exists(UNZIP_DIR):
        for folder in os.listdir(UNZIP_DIR):
            if os.path.isdir(os.path.join(UNZIP_DIR, folder)):
                try:
                    id_str, _ = folder.split(" ", 1)
                    numeric_id = int(id_str)
                    db_id_to_folder[numeric_id] = folder
                except ValueError:
                    continue
    
    results = {}
    
    if by_cluster:
        # Cluster-based collection (original behavior)
        cluster_df = load_cluster_assignments()
        if cluster_df.empty:
            print("Cluster assignments not found")
            return {}
            
        # Get unique clusters
        clusters = cluster_df['cluster'].unique()
        
        for cluster_id in clusters:
            print(f"Processing cluster {cluster_id}...")
            
            # Get database IDs in this cluster
            cluster_db_ids = set(cluster_df[cluster_df['cluster'] == cluster_id]['db_id'].astype(int))
            
            # Filter edge properties to only include pairs where both databases are in the cluster
            cluster_edges = edge_props[
                (edge_props['src'].astype(int).isin(cluster_db_ids)) & 
                (edge_props['tgt'].astype(int).isin(cluster_db_ids))
            ]
            
            if cluster_edges.empty:
                continue
                
            cluster_results = process_pairs(cluster_edges, k, cluster_id, db_id_to_folder, UNZIP_DIR, min_rows)
            
            if cluster_results:
                results[cluster_id] = cluster_results
                print(f"  Found {len(cluster_results)} pairs in cluster {cluster_id}")
    else:
        # Global collection (new default behavior)
        print("Processing all pairs globally...")
        
        # Load cluster info for metadata (optional)
        cluster_df = load_cluster_assignments()
        db_to_cluster = {}
        if not cluster_df.empty:
            db_to_cluster = dict(zip(cluster_df['db_id'].astype(int), cluster_df['cluster']))
        
        # Sort all edges by similarity and process globally
        global_results = process_pairs(edge_props, k, 'global', db_id_to_folder, UNZIP_DIR, min_rows, db_to_cluster)
        
        if global_results:
            results['global'] = global_results
            print(f"  Found {len(global_results)} pairs globally")
    
    return results


def process_pairs(edges_df, k, group_id, db_id_to_folder, UNZIP_DIR, min_rows, db_to_cluster=None):
    """
    Process edge pairs to find k valid similar pairs
    
    Args:
        edges_df: DataFrame of edges to process
        k: Number of pairs to find
        group_id: Cluster ID or 'global'
        db_id_to_folder: Mapping of database IDs to folder names
        UNZIP_DIR: Path to unzip directory
        min_rows: Minimum rows requirement
        db_to_cluster: Optional mapping of database IDs to cluster IDs (for global mode)
    
    Returns:
        List of valid pairs
    """
    # Sort by similarity in descending order for processing
    sorted_edges = edges_df.sort_values(by='similarity', ascending=False)
    
    results = []
    
    # Process pairs until we have k valid results or run out of pairs
    for _, row in sorted_edges.iterrows():
        if len(results) >= k:
            break
            
        db_id1 = int(row['src'])
        db_id2 = int(row['tgt'])
        similarity = row['similarity']
        
        # Get folder paths
        folder1 = db_id_to_folder.get(db_id1)
        folder2 = db_id_to_folder.get(db_id2)
        
        if not folder1 or not folder2:
            continue
        
        folder_path1 = os.path.join(UNZIP_DIR, folder1)
        folder_path2 = os.path.join(UNZIP_DIR, folder2)
        
        # Get database abstracts
        abstract1 = get_database_abstract(db_id1, folder_path1)
        abstract2 = get_database_abstract(db_id2, folder_path2)
        
        # Check for errors in abstracts
        if 'error' in abstract1 or 'error' in abstract2:
            continue
        
        # Check minimum row requirement
        rows1 = abstract1.get('max_rows', 0)
        rows2 = abstract2.get('max_rows', 0)
        
        if rows1 < min_rows or rows2 < min_rows:
            continue
        
        # Find common columns
        columns1 = set(abstract1.get('all_columns', []))
        columns2 = set(abstract2.get('all_columns', []))
        common_columns = list(columns1 & columns2)
        
        # Get cluster information
        if db_to_cluster:
            cluster1 = db_to_cluster.get(db_id1, 'unknown')
            cluster2 = db_to_cluster.get(db_id2, 'unknown')
        else:
            cluster1 = cluster2 = group_id
        
        pair_info = {
            "db_id1": db_id1,
            "db_id2": db_id2,
            "similarity": similarity,
            "cluster1": cluster1,
            "cluster2": cluster2,
            "same_cluster": cluster1 == cluster2,
            "database1": abstract1,
            "database2": abstract2,
            "common_columns": common_columns,
            "num_common_columns": len(common_columns)
        }
        
        results.append(pair_info)
    
    return results


def save_similar_pairs(results, output_dir="results/db_pairs"):
    """
    Save similar pairs to individual files
    
    Args:
        results (dict): Results from collect_similar_pairs_by_cluster
        output_dir (str): Output directory path
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    saved_files = []
    
    def convert_numpy_types(obj):
        """Convert numpy types to native Python types for JSON serialization"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        return obj
    
    for cluster_id, pairs in results.items():
        for pair in pairs:
            # Convert numpy types to native Python types
            pair_json = convert_numpy_types(pair)
            
            db_id1 = pair_json['db_id1']
            db_id2 = pair_json['db_id2']
            similarity = pair_json['similarity']
            
            # Create filename with database IDs, similarity, and cluster info
            cluster1 = pair_json.get('cluster1', 'unknown')
            cluster2 = pair_json.get('cluster2', 'unknown')
            same_cluster = pair_json.get('same_cluster', True)
            
            if same_cluster and cluster1 != 'global':
                filename = f"pair_{db_id1:05d}_{db_id2:05d}_sim{similarity:.4f}_cluster{cluster1}.json"
            else:
                filename = f"pair_{db_id1:05d}_{db_id2:05d}_sim{similarity:.4f}_c{cluster1}_{cluster2}.json"
            
            file_path = output_path / filename
            
            # Save pair information
            with open(file_path, 'w') as f:
                json.dump(pair_json, f, indent=2)
            
            saved_files.append(str(file_path))
    
    return saved_files


def main():
    parser = argparse.ArgumentParser(description='Collect similar database pairs globally or by cluster')
    parser.add_argument('-k', type=int, default=20, 
                       help='Number of similar pairs (total if global, per cluster if by-cluster) (default: 20)')
    parser.add_argument('-s', '--max-similarity', type=float, default=1.0,
                       help='Maximum similarity threshold (default: 1.0)')
    parser.add_argument('-r', '--min-rows', type=int, default=100,
                       help='Minimum number of rows required for each database (default: 100)')
    parser.add_argument('--by-cluster', action='store_true',
                       help='Collect pairs within each cluster separately (default: False, collect globally)')
    parser.add_argument('--output-dir', type=str, default="results/db_pairs",
                       help='Output directory for pair files (default: results/db_pairs)')
    
    args = parser.parse_args()
    
    mode = "by cluster" if args.by_cluster else "globally"
    print(f"Collecting top {args.k} similar pairs {mode}...")
    print(f"Maximum similarity: {args.max_similarity}")
    print(f"Minimum rows: {args.min_rows}")
    print(f"Output directory: {args.output_dir}")
    print("-" * 60)
    
    # Collect similar pairs
    results = collect_similar_pairs(
        k=args.k, 
        max_similarity=args.max_similarity,
        min_rows=args.min_rows,
        by_cluster=args.by_cluster
    )
    
    if not results:
        print("No similar pairs found matching the criteria.")
        return
    
    # Save results
    mode_suffix = "cluster" if args.by_cluster else "global"
    output_dir = Path(args.output_dir) / f"{mode_suffix}_s{args.max_similarity:.4f}_r{args.min_rows}"
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_files = save_similar_pairs(results, output_dir)
    
    print(f"\nSummary:")
    if args.by_cluster:
        print(f"Total clusters processed: {len(results)}")
        total_pairs = sum(len(pairs) for pairs in results.values())
        print(f"Total pairs saved: {total_pairs}")
        # Print cluster summary
        for cluster_id, pairs in results.items():
            print(f"  Cluster {cluster_id}: {len(pairs)} pairs")
    else:
        print(f"Total pairs saved: {len(saved_files)}")
        if 'global' in results:
            # Print cross-cluster vs same-cluster breakdown
            same_cluster = sum(1 for pair in results['global'] if pair.get('same_cluster', True))
            cross_cluster = len(results['global']) - same_cluster
            print(f"  Same cluster pairs: {same_cluster}")
            print(f"  Cross cluster pairs: {cross_cluster}")
    
    print(f"Files saved to: {output_dir}")


if __name__ == "__main__":
    main()