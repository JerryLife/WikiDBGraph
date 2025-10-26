import pandas as pd
import os
import numpy as np
from collections import defaultdict
import json
import argparse
import sqlite3


from analysis.NodeProperties import Database, Table, Column


def load_edge_properties():
    """Load edge structural properties from CSV file"""
    structural_props_path = "data/graph/filtered_edges_threshold_0.94.csv"
    if os.path.exists(structural_props_path):
        return pd.read_csv(structural_props_path)
    else:
        print(f"Warning: {structural_props_path} not found")
        return pd.DataFrame()

def get_database_row_counts(folder: str):
    """Get the maximum number of rows in each database"""
    row_counts = {}
    
    tables_dir = os.path.join(folder, "tables")
    for table_file in os.listdir(tables_dir):
        table_name = table_file.split(".")[0]
        row_counts[table_name] = pd.read_csv(os.path.join(tables_dir, table_file)).shape[0]
    
    return max(row_counts.values())

def find_similar_non_matching_pairs(k=10, s=1.0, c=None):
    """
    Find k most similar database pairs that are not labeled as true matches.
    
    Args:
        k (int): Number of similar pairs to return
        s (float): The target similarity to print
        c (int, optional): The cluster ID to search in
    Returns:
        list: List of tuples containing (db_id1, db_id2, similarity_score, common_columns)
    """
    # Load edge properties data
    edge_props = load_edge_properties()
    
    if edge_props.empty:
        return []

    # If cluster ID is specified, filter by cluster
    if c is not None:
        # Load cluster assignments
        cluster_path = "data/graph/cluster_assignments_dim2_sz100_msNone.csv"
        if os.path.exists(cluster_path):
            cluster_df = pd.read_csv(cluster_path)
            # Get database IDs in the specified cluster
            cluster_db_ids = set(cluster_df[cluster_df['cluster'] == c]['db_id'].astype(int))
            # Filter edge properties to only include pairs where both databases are in the cluster
            edge_props = edge_props[
                (edge_props['src'].astype(int).isin(cluster_db_ids)) & 
                (edge_props['tgt'].astype(int).isin(cluster_db_ids))
            ]
        else:
            print(f"Warning: {cluster_path} not found, ignoring cluster filter")

    # Calculate the absolute difference between each similarity score and the target similarity s
    edge_props['similarity_diff'] = abs(edge_props['similarity'] - s)
    
    # Sort by the difference (smallest difference first)
    sorted_pairs = edge_props.sort_values(by='similarity_diff', ascending=True)
    
    # Get top k pairs closest to the target similarity s
    top_k_pairs = sorted_pairs.head(k)
    
    # Extract relevant information
    result = []
    for _, row in top_k_pairs.iterrows():
        db_id1 = int(row['src'])
        db_id2 = int(row['tgt'])
        similarity = row['similarity']
        
        # Get column names from the schema files
        UNZIP_DIR = os.path.join("data/unzip")
        
        # Create a mapping of database IDs to folder names
        db_id_to_folder = {}
        for folder in os.listdir(UNZIP_DIR):
            if os.path.isdir(os.path.join(UNZIP_DIR, folder)):
                id, _ = folder.split(" ", 1)
                numeric_id = int(id)
                db_id_to_folder[numeric_id] = folder
        
        # Get the actual folder names
        folder1 = db_id_to_folder.get(db_id1)
        folder2 = db_id_to_folder.get(db_id2)
        
        # Get the maximum number of rows in each database
        maxrow1 = get_database_row_counts(os.path.join(UNZIP_DIR, folder1))
        maxrow2 = get_database_row_counts(os.path.join(UNZIP_DIR, folder2))
        
        common_columns = []
        if folder1 and folder2:
            # Load schema files
            schema_path1 = os.path.join(UNZIP_DIR, folder1, "schema.json")
            schema_path2 = os.path.join(UNZIP_DIR, folder2, "schema.json")
            
            if os.path.exists(schema_path1) and os.path.exists(schema_path2):
                try:
                    with open(schema_path1, "r") as f:
                        schema1 = json.load(f)
                    with open(schema_path2, "r") as f:
                        schema2 = json.load(f)

                    db1 = Database(db_id1)
                    db2 = Database(db_id2)

                    db1.load_from_schema(schema1)
                    db2.load_from_schema(schema2)

                    wiki_topic_id1 = schema1['wikidata_topic_item_id']
                    wiki_topic_id2 = schema2['wikidata_topic_item_id']

                    # Extract column names and normalize by removing non-alphanumeric characters
                    columns1 = {''.join(c for c in c.column_name.lower() if c.isalnum()) 
                               for table in db1.tables for c in table.columns}
                    columns2 = {''.join(c for c in c.column_name.lower() if c.isalnum()) 
                               for table in db2.tables for c in table.columns}
                    
                    # Find common columns
                    common_columns = list(columns1 & columns2)
                except Exception as e:
                    common_columns = f"Error extracting columns: {str(e)}"
            else:
                common_columns = "Schema files not found"
        else:
            common_columns = "Database folders not found"
        
        result.append((db_id1, db_id2, similarity, common_columns, wiki_topic_id1, wiki_topic_id2, maxrow1, maxrow2))
    
    return result

def print_similar_non_matching_pairs(k=10, s=1.0, c=None):
    """Print k most similar database pairs that are not labeled as true matches"""
    similar_pairs = find_similar_non_matching_pairs(k, s, c)
    
    if not similar_pairs:
        print("No similar non-matching pairs found or data not available.")
        return
    
    print(f"Top {len(similar_pairs)} similar database pairs that are not labeled as true matches:")
    print("-" * 80)
    
    for i, (db_id1, db_id2, similarity, common_columns, wiki_topic_id1, wiki_topic_id2, maxrow1, maxrow2) in enumerate(similar_pairs, 1):
        if maxrow1 < 100 or maxrow2 < 100:
            continue
        print(f"Pair {i}:")
        print(f"  Database 1 ID: {db_id1} (Topic ID: {wiki_topic_id1})")
        print(f"  Database 2 ID: {db_id2} (Topic ID: {wiki_topic_id2})")
        print(f"  Similarity Score: {similarity:.4f}")
        print(f"  Max Rows - DB1: {maxrow1}, DB2: {maxrow2}")
        print(f"  Common Columns: {common_columns}")
        print("-" * 80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Print similar database pairs')
    parser.add_argument('-k', type=int, default=10, help='Number of similar pairs to print')
    parser.add_argument('-s', type=float, default=1.0, help='The target similarity to print')
    parser.add_argument('-c', type=int, default=None, help='The cluster ID to search in')
    args = parser.parse_args()

    # Default to showing top 10 similar non-matching pairs
    k = args.k
    s = args.s
    c = args.c
    print_similar_non_matching_pairs(k, s, c)
