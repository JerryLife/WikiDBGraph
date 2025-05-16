import os
import math
import traceback
import pickle

# import orjson
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from tqdm import tqdm
import numpy as np
from collections import Counter, defaultdict

import networkx as nx


from src.analysis.NodeProperties import Database, Table, Column, ForeignKey


"""
Graph Attributes:

Node Properties:
* Structural Properties:
    - Number of tables
    - Number of columns
    - Proportion of data types (categorical, numerical etc.)
    - Foreign key density (number of foreign keys / number of columns)
    - Average table connectivity (number of potential joins / number of tables)
* Semantic Properties:
    - Database Embedding (we already have this)
    - Topic (e.g. medical, financial, etc., can be derived by clustering database embeddings)
* Statistical Properties:
    - Data volume (file size)
    - All-join size (number of rows when joining all tables with foreign keys)
    - Average/Median Column Cardinality (the average or median number of distinct values across columns)
    - Average/Median Column Sparsity (number of columns with nulls / total number of columns)
    - Average/Median Column Entropy (measure of randomness in the column)
    
Edge Properties:
* Structural Properties:
    - Jaccard index of set of table names (number of overlapping table names / total number of table names)
    - Jaccard index of set of columns (number of overlapping columns / total number of columns)
    - Jaccard index of set of data types (number of overlapping data types / total number of data types)
    - Similarity of internal graph structure (use graph matching; node: table, edge: foreign key)
* Semantic Properties:
    - Similarity of database embedding (we already have this)
    - Confidence of similarity
* Statistical Properties:
    - Divergence of distribution for shared columns
    - Ratio of overlap for shared columns
    
"""


class EdgeProperty:
    """
    Class to calculate and store edge properties between two database schemas.
    """
    
    def __init__(self, db_id1, db_id2):
        """Initialize with the two database IDs that form the edge."""
        self.db_id1 = db_id1
        self.db_id2 = db_id2
        self.properties = {}
    
    def calculate_properties(self, db1, db2):
        """Calculate structural properties between two Database objects."""
        # Extract table names and normalize by removing non-alphanumeric characters
        table_names1 = {''.join(c for c in table.table_name.lower() if c.isalnum()) for table in db1.tables}
        table_names2 = {''.join(c for c in table.table_name.lower() if c.isalnum()) for table in db2.tables}
        
        # Calculate Jaccard index for normalized table names
        table_intersection = len(table_names1.intersection(table_names2))
        table_union = len(table_names1.union(table_names2))
        jaccard_tables = table_intersection / table_union if table_union > 0 else 0
        
        # Extract column names and normalize by removing non-alphanumeric characters
        columns1 = {''.join(c for c in col.column_name.lower() if c.isalnum()) for table in db1.tables for col in table.columns}
        columns2 = {''.join(c for c in col.column_name.lower() if c.isalnum()) for table in db2.tables for col in table.columns}
        
        # Calculate Jaccard index for column names
        column_intersection = len(columns1.intersection(columns2))
        column_union = len(columns1.union(columns2))
        jaccard_columns = column_intersection / column_union if column_union > 0 else 0
        
        # Count occurrences of each data type
        dtype_counts1 = Counter()
        dtype_counts2 = Counter()
        
        for table in db1.tables:
            dtype_counts1.update(col.data_type for col in table.columns if col.data_type)
            
        for table in db2.tables:
            dtype_counts2.update(col.data_type for col in table.columns if col.data_type)
        
        # Get the set of data types
        data_types1 = set(dtype_counts1.keys())
        data_types2 = set(dtype_counts2.keys())
        
        # Calculate probability distributions
        total_cols1 = sum(dtype_counts1.values())
        total_cols2 = sum(dtype_counts2.values())
        
        dtype_dist1 = {dtype: count/total_cols1 for dtype, count in dtype_counts1.items()} if total_cols1 > 0 else {}
        dtype_dist2 = {dtype: count/total_cols2 for dtype, count in dtype_counts2.items()} if total_cols2 > 0 else {}
        
        # Calculate probability distance (using total variation distance)
        hellinger_dist_data_types = EdgeProperty.hellinger_distance(dtype_dist1, dtype_dist2)
        
        # Also calculate Jaccard index for data types
        dtype_intersection = len(data_types1.intersection(data_types2))
        dtype_union = len(data_types1.union(data_types2))
        jaccard_dtypes = dtype_intersection / dtype_union if dtype_union > 0 else 0
        
        # Extract foreign key structure for graph similarity using networkx
        G1 = self._build_fk_graph_from_db(db1)
        G2 = self._build_fk_graph_from_db(db2)
        
        # Calculate approximate graph edit distance
        graph_edit_distance = nx.graph_edit_distance(G1, G2, timeout=30)
        
        # Store properties
        self.properties = {
            "jaccard_table_names": jaccard_tables,
            "jaccard_columns": jaccard_columns,
            "jaccard_data_types": jaccard_dtypes,
            "hellinger_distance_data_types": hellinger_dist_data_types,
            "graph_edit_distance": graph_edit_distance,
            "common_tables": table_intersection,
            "common_columns": column_intersection,
            "common_data_types": dtype_intersection
        }
        
        return self.properties
    
    @staticmethod
    def _prepare_distributions_for_comparison(dist1_dict, dist2_dict):
        """
        Prepares two distributions (represented as dictionaries) for comparison.
        Ensures that both distributions cover the same set of categories,
        filling in zeros for missing categories.

        Args:
            dist1_dict (dict): First distribution {category: probability}.
            dist2_dict (dict): Second distribution {category: probability}.

        Returns:
            tuple: (p_values, q_values) where p_values and q_values are lists
                of probabilities for a common, sorted set of categories.
        """
        # Get all unique categories present in either distribution
        all_categories = sorted(list(set(dist1_dict.keys()) | set(dist2_dict.keys())))
        
        p_values = [dist1_dict.get(cat, 0.0) for cat in all_categories]
        q_values = [dist2_dict.get(cat, 0.0) for cat in all_categories]
        
        return p_values, q_values
    
    @staticmethod
    def hellinger_distance(dist1_dict, dist2_dict):
        """
        Calculates the Hellinger Distance between two discrete distributions.
        Ranges from 0 (identical) to 1 (orthogonal).
        """
        p_values, q_values = EdgeProperty._prepare_distributions_for_comparison(dist1_dict, dist2_dict)

        if not p_values and not q_values:
            return 0.0

        sum_sqrt_diff_sq = 0.0
        for p_val, q_val in zip(p_values, q_values):
            sum_sqrt_diff_sq += (math.sqrt(p_val) - math.sqrt(q_val))**2

        # The factor 1/sqrt(2) normalizes the distance to be between 0 and 1.
        return math.sqrt(sum_sqrt_diff_sq) / math.sqrt(2)
    
    def _build_fk_graph_from_db(self, db):
        """Build a graph representation of foreign key relationships from a Database object.
        
        Returns:
            networkx.DiGraph: A directed graph representing the database schema
        """
        # Create a directed graph
        G = nx.DiGraph()
        
        # Add all tables as nodes
        for table in db.tables:
            G.add_node(table.table_name)
            
        # Add foreign key relationships as edges
        for table in db.tables:
            table_name = table.table_name
            for fk in table.foreign_keys:
                source = table_name
                target = fk.target_table
                if target:
                    G.add_edge(source, target)
        
        return G
    
    def to_dict(self):
        """Convert to dictionary for DataFrame conversion."""
        result = {"db_id1": self.db_id1, "db_id2": self.db_id2}
        result.update(self.properties)
        return result
    
    def __str__(self):
        """String representation of the edge property."""
        return f"Edge({self.db_id1}, {self.db_id2}): {self.properties}"


def get_edge_structural_properties(raw_dir: str, edges: np.ndarray, save_path: str = None, num_threads: int = None, structural_path: str = None, debug: bool = False):
    """
    Extract structural properties for edges between database schemas.
    
    Args:
        raw_dir: Directory containing the database folders
        edges: List of edges (src, tgt)
        save_path: Path to save the results
        num_threads: Number of threads to use for parallel processing
        structural_path: Path to the structural.pkl file (optional)
        debug: Enable debug mode to record timing information
    
    Returns:
        List of EdgeProperty objects
    """
    import time
    start_time = time.time()
    timing_info = {}
    
    # Try to load structural data if provided
    structural_data = None
    if structural_path and os.path.exists(structural_path):
        load_start = time.time()
        with open(structural_path, 'rb') as f:
            structural_data = pickle.load(f)
        if debug:
            timing_info['load_structural_data'] = time.time() - load_start
        print(f"Loaded structural data from {structural_path}")
    else:
        print(f"Structural data not found at {structural_path}")
    
    scan_start = time.time()
    if structural_data is None:
        # First, scan the raw_dir to get a mapping of database IDs to folder names
        db_id_to_folder = {}
        for folder in os.listdir(raw_dir):
            if os.path.isdir(os.path.join(raw_dir, folder)):
                id, _ = folder.split(" ", 1)
                numeric_id = int(id)
                db_id_to_folder[numeric_id] = folder
        print(f"Found {len(db_id_to_folder)} database folders in {raw_dir}")
    
    if debug:
        timing_info['scan_folders'] = time.time() - scan_start
    
    def process_db_pair(db_pair):
        pair_start = time.time()
        pair_timing = {}
        
        db_id1, db_id2 = db_pair
        try:
            # Convert to integers for lookup in the mapping
            db_id1 = int(db_id1)
            db_id2 = int(db_id2)
            db_id1_str = f"{int(db_id1):5d}"
            db_id2_str = f"{int(db_id2):5d}"
            
            database1 = None
            database2 = None
            
            # Try to get databases from structural data first
            if structural_data is not None:
                struct_start = time.time()
                database1 = structural_data[db_id1_str]
                database2 = structural_data[db_id2_str]
                if debug:
                    pair_timing['get_from_structural'] = time.time() - struct_start
            
            # Load schema files if structural_path is None or databases not available from structural data
            if structural_path is None or database1 is None or database2 is None:
                import orjson   # import only when needed to avoid conflict with nogil
                
                schema_load_start = time.time()
                
                # Get the actual folder names
                folder1 = db_id_to_folder.get(db_id1)
                folder2 = db_id_to_folder.get(db_id2)
                
                if not folder1:
                    print(f"Database folder not found for ID {db_id1}")
                    return None
                if not folder2:
                    print(f"Database folder not found for ID {db_id2}")
                    return None
            
                schema_path1 = os.path.join(raw_dir, folder1, "schema.json")
                schema_path2 = os.path.join(raw_dir, folder2, "schema.json")
                
                if not os.path.exists(schema_path1):
                    print(f"Schema file not found: {schema_path1}")
                    return None
                if not os.path.exists(schema_path2):
                    print(f"Schema file not found: {schema_path2}")
                    return None
                
                try:
                    with open(schema_path1, "rb") as f:
                        schema1 = orjson.loads(f.read())
                    database1 = Database(db_id1)
                    database1.load_from_schema(schema1)
                except Exception as e:
                    print(f"Error loading schema for {db_id1}: {str(e)}")
                    return None
                    
                try:
                    with open(schema_path2, "rb") as f:
                        schema2 = orjson.loads(f.read())
                    database2 = Database(db_id2)
                    database2.load_from_schema(schema2)
                except Exception as e:
                    print(f"Error loading schema for {db_id2}: {str(e)}")
                    return None
                
                if debug:
                    pair_timing['load_schemas'] = time.time() - schema_load_start
            
            # Create and calculate edge properties
            calc_start = time.time()
            edge_prop = EdgeProperty(db_id1, db_id2)
            edge_prop.calculate_properties(database1, database2)
            if debug:
                pair_timing['calculate_properties'] = time.time() - calc_start
                edge_prop.properties['timing'] = pair_timing
                edge_prop.properties['total_time'] = time.time() - pair_start
            
            return edge_prop
        except Exception as e:
            print(f"Error processing pair {db_id1}, {db_id2}: {str(e)}")
            traceback.print_exc()
            return None

    
    # Process database pairs
    process_start = time.time()
    edge_properties = []
    
    if debug:
        # In debug mode, process the first 10 edges and report mean
        print("Debug mode: processing first 10 edges")
        debug_edges = edges[:10] if len(edges) >= 10 else edges
        edge_times = []
        
        for edge in debug_edges:
            edge_start = time.time()
            edge_prop = process_db_pair(edge)
            edge_time = time.time() - edge_start
            edge_times.append(edge_time)
            if edge_prop:
                edge_properties.append(edge_prop)
        
        if edge_times:
            mean_time = sum(edge_times) / len(edge_times)
            print(f"Mean processing time per edge: {mean_time:.4f} seconds")
            
        timing_info['process_debug_edges'] = time.time() - process_start
        timing_info['mean_edge_processing_time'] = mean_time if edge_times else 0
    else:
        # Normal mode: process in parallel
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(process_db_pair, pair) for pair in edges]
            for future in tqdm(futures, total=len(edges), desc="Processing database pairs"):
                edge_prop = future.result()
                if edge_prop:
                    edge_properties.append(edge_prop)
        timing_info['parallel_processing'] = time.time() - process_start
    
    print(f"Successfully processed {len(edge_properties)} out of {len(edges)} database pairs")
    
    # Save results if path is provided
    if save_path:
        save_start = time.time()
        # Convert to DataFrame
        rows = [edge_prop.to_dict() for edge_prop in edge_properties]
        df = pd.DataFrame(rows)
        df.to_csv(save_path, index=False)
        if debug:
            timing_info['save_results'] = time.time() - save_start
    
    if debug:
        timing_info['total_time'] = time.time() - start_time
        print(f"Timing information: {timing_info}")
        if edge_properties and 'timing' in edge_properties[0].properties:
            print(f"Detailed timing for first edge: {edge_properties[0].properties['timing']}")
    
    return edge_properties


if __name__ == "__main__":
    UNZIP_DIR = os.path.join("data/unzip")
    GRAPH_DIR = os.path.join("data/graph")
    STRUCTURAL_PATH = os.path.join(GRAPH_DIR, "structural.pkl")
    
    edges_df = pd.read_csv(os.path.join(GRAPH_DIR, "filtered_edges_threshold_0.96.csv"))
    edges = list(edges_df[["src", "tgt"]].values)
    
    get_edge_structural_properties(
        UNZIP_DIR, 
        edges,
        save_path=os.path.join(GRAPH_DIR, "edge_structural_properties.csv"), 
        num_threads=48,
        structural_path=STRUCTURAL_PATH,
        debug=False
    )