import networkx as nx
import pandas as pd
import json
import random
import os
import glob

def load_schema(database_name, schema_dir='/home/wangzy/wikidbs/wikidbs/data/schema/'):
    """
    Load the schema information for the specified database.
    """
    db_prefix = database_name.split('_')[0]
    schema_pattern = os.path.join(schema_dir, f"{db_prefix}_*.json")
    matching_files = glob.glob(schema_pattern)
    
    if not matching_files:
        return None
    
    schema_path = matching_files[0]
    with open(schema_path, 'r', encoding='utf-8') as f:
        schema = json.load(f)
    return schema

def extract_schema_features(schema):
    """
    Extracts a dictionary of {table_name: [column_names]} from a schema.
    """
    features = {}
    if schema:
        for table in schema.get('tables', []):
            table_name = table.get('table_name', 'N/A')
            columns = [column.get('column_name', 'N/A') for column in table.get('columns', [])]
            features[table_name] = columns
    return features

def log_message(message, log_file):
    """
    Write log messages to a log file.
    """
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(message + '\n')
    print(message)

def display_schema_info_with_path_markers(schema, db_name, prev_schema, next_schema, log_file):
    """
    Display table and column information of the database schema.
    Mark columns that appear in the next database with * and in the previous database with ^.
    """
    if not schema:
        log_message(f"Schema not found for {db_name}", log_file)
        return
    log_message(f"Schema information for database {db_name}:", log_file)
    
    prev_features = extract_schema_features(prev_schema) if prev_schema else {}
    next_features = extract_schema_features(next_schema) if next_schema else {}
    
    for table in schema.get('tables', []):
        table_name = table.get('table_name', 'N/A')
        log_message(f"  Table Name: {table_name}", log_file)
        log_message("    Columns:", log_file)
        for column in table.get('columns', []):
            column_name = column.get('column_name', 'N/A')
            # Check if the column appears in any table of the previous or next schema
            is_in_next_schema = any(column_name in columns for columns in next_features.values())
            is_in_prev_schema = any(column_name in columns for columns in prev_features.values())
            
            if is_in_next_schema and is_in_prev_schema:
                column_display = f"{column_name.ljust(38)}*^"
            elif is_in_next_schema:
                column_display = f"{column_name.ljust(39)}*"
            elif is_in_prev_schema:
                column_display = f"{column_name.ljust(39)}^"
            else:
                column_display = column_name
            log_message(f"      - {column_display}", log_file)

if __name__ == "__main__":
    seed = 2
    max_hops_list = [2, 3]
    graph_path = f'out/graphs/overlap_graph_seed{seed}.gexf'
    G = nx.read_gexf(graph_path)
    results = {}

    for max_hops in max_hops_list:
        log_file = f'hop_{max_hops}_seed{seed}.log'
        log_message(f"\nProcessing {max_hops} hop(s)", log_file)
        paths = dict(nx.all_pairs_shortest_path(G, cutoff=max_hops))
        found = False  # Flag to indicate if a path has been found
        for source, target_paths in paths.items():
            if found:
                break  # Exit the outer loop if a path has been found
            for target, path in target_paths.items():
                if len(path) - 1 == max_hops:
                    log_message(f"Found path: {' -> '.join(path)}", log_file)
                    for i, db_name in enumerate(path):
                        schema = load_schema(db_name)
                        prev_schema = load_schema(path[i - 1]) if i > 0 else None
                        next_schema = load_schema(path[i + 1]) if i < len(path) - 1 else None
                        display_schema_info_with_path_markers(schema, db_name, prev_schema, next_schema, log_file)
                    found = True  # Set the flag to True after processing the path
                    break  # Exit the inner loop after processing the path
        if not found:
            log_message("No suitable paths found.", log_file)
