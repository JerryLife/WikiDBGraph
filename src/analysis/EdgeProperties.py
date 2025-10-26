import os
import math
import traceback
import pickle
import os
import tempfile
import shutil
import time

# import orjson
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from tqdm import tqdm
import numpy as np
from collections import Counter, defaultdict
import argparse
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
    
    def calculate_properties(self, db1, db2, debug: bool = False):
        """Calculate structural properties between two Database objects."""
        timing = {}
        start_time = time.time()
        
        # Extract table names and normalize by removing non-alphanumeric characters
        table_names1 = {''.join(c for c in table.table_name.lower() if c.isalnum()) for table in db1.tables}
        table_names2 = {''.join(c for c in table.table_name.lower() if c.isalnum()) for table in db2.tables}
        
        if debug:
            timing['extract_table_names'] = time.time() - start_time
            step_time = time.time()
        
        # Calculate Jaccard index for normalized table names
        table_intersection = len(table_names1 & table_names2)  # Using set intersection operator
        table_union = len(table_names1 | table_names2)  # Using set union operator
        jaccard_tables = table_intersection / table_union if table_union > 0 else 0
        
        if debug:
            timing['calculate_jaccard_tables'] = time.time() - step_time
            step_time = time.time()
        
        # Extract column names and normalize by removing non-alphanumeric characters
        columns1 = {''.join(c for c in col.column_name.lower() if c.isalnum()) for table in db1.tables for col in table.columns}
        columns2 = {''.join(c for c in col.column_name.lower() if c.isalnum()) for table in db2.tables for col in table.columns}
        
        if debug:
            timing['extract_column_names'] = time.time() - step_time
            step_time = time.time()
        
        # Calculate Jaccard index for column names
        column_intersection = len(columns1 & columns2)
        column_union = len(columns1 | columns2)
        jaccard_columns = column_intersection / column_union if column_union > 0 else 0
        
        if debug:
            timing['calculate_jaccard_columns'] = time.time() - step_time
            step_time = time.time()
        
        # Count occurrences of each data type - do this in one pass
        dtype_counts1 = Counter(col.data_type for table in db1.tables for col in table.columns if col.data_type)
        dtype_counts2 = Counter(col.data_type for table in db2.tables for col in table.columns if col.data_type)
        
        # Get the set of data types
        data_types1 = set(dtype_counts1.keys())
        data_types2 = set(dtype_counts2.keys())
        
        if debug:
            timing['extract_data_types'] = time.time() - step_time
            step_time = time.time()
        
        # Calculate probability distributions
        total_cols1 = sum(dtype_counts1.values())
        total_cols2 = sum(dtype_counts2.values())
        
        dtype_dist1 = {dtype: count/total_cols1 for dtype, count in dtype_counts1.items()} if total_cols1 > 0 else {}
        dtype_dist2 = {dtype: count/total_cols2 for dtype, count in dtype_counts2.items()} if total_cols2 > 0 else {}
        
        # Calculate probability distance
        hellinger_dist_data_types = self.hellinger_distance(dtype_dist1, dtype_dist2)
        
        if debug:
            timing['calculate_hellinger_distance'] = time.time() - step_time
            step_time = time.time()
        
        # Calculate Jaccard index for data types
        dtype_intersection = len(data_types1 & data_types2)
        dtype_union = len(data_types1 | data_types2)
        jaccard_dtypes = dtype_intersection / dtype_union if dtype_union > 0 else 0
        
        if debug:
            timing['calculate_jaccard_dtypes'] = time.time() - step_time
            step_time = time.time()
        
        # Extract foreign key structure - only build if needed for graph edit distance
        G1 = self._build_fk_graph_from_db(db1)
        G2 = self._build_fk_graph_from_db(db2)
        
        # Use a timeout for graph edit distance to prevent long-running calculations
        try:
            graph_edit_distance = nx.graph_edit_distance(G1, G2, timeout=5)  # Reduced timeout
        except (nx.NetworkXError, TimeoutError):
            # If timeout or error, use a fallback approximation
            graph_edit_distance = abs(G1.number_of_nodes() - G2.number_of_nodes()) + \
                               abs(G1.number_of_edges() - G2.number_of_edges())
        
        if debug:
            timing['calculate_graph_edit_distance'] = time.time() - step_time
            step_time = time.time()
        
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
        
        if debug:
            timing['store_properties'] = time.time() - step_time
            timing['total_time'] = time.time() - start_time
            # Sort timing dictionary by values in descending order
            sorted_timing = {k: v for k, v in sorted(timing.items(), key=lambda item: item[1], reverse=True)}
            self.properties['timing'] = sorted_timing
            print(f"Edge property calculation timing: {sorted_timing}")
        
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
        all_categories = list(set(dist1_dict.keys()) | set(dist2_dict.keys()))
        
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
        """Build a graph representation of foreign key relationships from a Database object."""
        # Create a directed graph
        G = nx.DiGraph()
        
        # Pre-compute table name lookup for faster access
        table_names = {table.table_name for table in db.tables}
        
        # Add all tables as nodes in one operation
        G.add_nodes_from(table_names)
        
        # Collect all edges first, then add them in one operation
        edges = []
        for table in db.tables:
            table_name = table.table_name
            for fk in table.foreign_keys:
                target = fk.target_table
                if target and target in table_names:  # Verify target exists
                    edges.append((table_name, target))
        
        # Add all edges at once
        G.add_edges_from(edges)
        
        return G
    
    def to_dict(self):
        """Convert to dictionary for DataFrame conversion."""
        result = {"db_id1": self.db_id1, "db_id2": self.db_id2}
        result.update(self.properties)
        return result
    
    def __str__(self):
        """String representation of the edge property."""
        return f"Edge({self.db_id1}, {self.db_id2}): {self.properties}"


def save_file(file_path, data, description=None):
    """
    Thread-safe function to save data to a file, preventing race conditions.
    
    Args:
        file_path: Path where the file should be saved
        data: DataFrame or data to be saved
        description: Description for logging purposes
    
    Returns:
        bool: True if this process saved the file, False if another process did
    """
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
    
    # Create a temporary file in the same directory
    temp_dir = os.path.dirname(os.path.abspath(file_path))
    temp_prefix = os.path.basename(file_path) + "_tmp_"
    
    with tempfile.NamedTemporaryFile(dir=temp_dir, prefix=temp_prefix, delete=False) as tmp_file:
        temp_path = tmp_file.name
    
    try:
        # Write data to the temporary file
        if isinstance(data, pd.DataFrame):
            data.to_csv(temp_path, index=False)
        else:
            with open(temp_path, 'wb') as f:
                f.write(data)
        
        # Attempt atomic rename (which is atomic on POSIX systems)
        if not os.path.exists(file_path):
            os.rename(temp_path, file_path)
            if description:
                print(f"{description} saved to {file_path}")
            return True
        else:
            # File already exists, another process beat us to it
            os.unlink(temp_path)  # Clean up the temp file
            return False
    except Exception as e:
        # Clean up on error
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        print(f"Error saving {description} to {file_path}: {str(e)}")
        return False


def get_edge_structural_properties(raw_dir: str, edges: np.ndarray, save_path: str = None, num_threads: int = None, structural_path: str = None, 
                                   save_every: int = None, save_first: int = None, debug: bool = False, batch_size: int = 1000):
    """
    Extract structural properties for edges between database schemas using parallel processing.
    
    Args:
        raw_dir: Directory containing the database folders
        edges: List of edges (src, tgt)
        save_path: Path to save the results
        num_threads: Number of threads to use for parallel processing
        structural_path: Path to the structural.pkl file (optional)
        save_every: Save results every N edges
        save_first: Save first N edges
        debug: Enable debug mode to record timing information
        batch_size: Number of edges to process in each thread
    
    Returns:
        List of EdgeProperty objects
    """
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
            
            database1 = None
            database2 = None
            
            # Try to get databases from structural data first
            if structural_data is not None:
                struct_start = time.time()
                database1 = structural_data[db_id1]
                database2 = structural_data[db_id2]
                if debug:
                    pair_timing['get_from_structural'] = time.time() - struct_start
            
            # Load schema files if structural_path is None or databases not available from structural data
            if structural_path is None or database1 is None or database2 is None:
                import json as json   # import only when needed to avoid conflict with nogil
                
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
                        schema1 = json.loads(f.read())
                    database1 = Database(db_id1)
                    database1.load_from_schema(schema1)
                except Exception as e:
                    print(f"Error loading schema for {db_id1}: {str(e)}")
                    return None
                    
                try:
                    with open(schema_path2, "rb") as f:
                        schema2 = json.loads(f.read())
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
            edge_prop.calculate_properties(database1, database2, debug=debug)
            if debug:
                pair_timing['calculate_properties'] = time.time() - calc_start
                edge_prop.properties['timing'] = pair_timing
                edge_prop.properties['total_time'] = time.time() - pair_start
            
            return edge_prop
        except Exception as e:
            print(f"Error processing pair {db_id1}, {db_id2}: {str(e)}")
            traceback.print_exc()
            return None

    def process_batch(batch_edges, batch_id):
        """Process a batch of edges and return the results"""
        batch_start_time = time.time()
        batch_results = []
        
        for i, edge in enumerate(batch_edges):
            edge_prop = process_db_pair(edge)
            if edge_prop:
                batch_results.append(edge_prop)
            
            # # Print progress occasionally
            # if (i + 1) % 100 == 0 or i == len(batch_edges) - 1:
            #    print(f"Batch {batch_id}: Processed {i+1}/{len(batch_edges)} edges")
        
        batch_time = time.time() - batch_start_time
        edges_per_second = len(batch_edges) / batch_time if batch_time > 0 else 0
        # print(f"Batch {batch_id} completed in {batch_time:.2f} seconds ({edges_per_second:.2f} edges/sec)")
        # print(f"Successfully processed {len(batch_results)} out of {len(batch_edges)} edges in batch {batch_id}")
        
        # Save batch results if path is provided
        if save_path:
            # Create a tmp directory for batch files
            tmp_dir = os.path.join(os.path.dirname(save_path), "tmp")
            os.makedirs(tmp_dir, exist_ok=True)
            batch_save_path = os.path.join(tmp_dir, f"{os.path.basename(os.path.splitext(save_path)[0])}_batch_{batch_id}.csv")
            if batch_results:
                batch_rows = [edge_prop.to_dict() for edge_prop in batch_results]
                batch_df = pd.DataFrame(batch_rows)
                save_file(batch_save_path, batch_df)
        
        return batch_results

    # Process database pairs in parallel with each thread handling a batch
    process_start = time.time()
    edge_properties = []
    
    if debug:
        # In debug mode, process just a small batch
        debug_batch_size = min(10, len(edges))
        debug_edges = edges[:debug_batch_size]
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
            estimated_total_time = mean_time * len(edges) / (num_threads or 1)
            print(f"Estimated total processing time: {estimated_total_time/3600:.2f} hours")
            
        timing_info['process_debug_edges'] = time.time() - process_start
        timing_info['mean_edge_processing_time'] = mean_time if edge_times else 0
    else:
        # Set default number of threads if not specified
        if num_threads is None:
            num_threads = os.cpu_count() * 2  # Nogil can handle more threads effectively
        
        print(f"Using {num_threads} threads, each processing a batch of edges")
        
        # Divide edges into batches for each thread
        total_edges = len(edges)
        edges_per_thread = (total_edges + num_threads - 1) // num_threads  # Ceiling division
        
        # Adjust batch size if it's too small
        if batch_size > edges_per_thread:
            batch_size = edges_per_thread
            print(f"Adjusted batch size to {batch_size} edges per thread")
        
        # Create batches for each thread
        batches = []
        for i in range(0, total_edges, batch_size):
            batches.append((edges[i:i+batch_size], len(batches) + 1))
        
        print(f"Created {len(batches)} batches from {total_edges} edges")
        
        # Process batches in parallel
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(process_batch, batch_edges, batch_id) for batch_edges, batch_id in batches]
            
            # Collect results as they complete
            for future in tqdm(as_completed(futures), total=len(batches), desc="Processing batches"):
                batch_results = future.result()
                edge_properties.extend(batch_results)
                
                # Save intermediate results if needed
                if save_path and save_every and len(edge_properties) >= save_every:
                    save_point = (len(edge_properties) // save_every) * save_every
                    if save_point > 0:
                        intermediate_save_path = f"{os.path.splitext(save_path)[0]}_intermediate_{save_point}.csv"
                        all_rows = [edge_prop.to_dict() for edge_prop in edge_properties[:save_point]]
                        all_df = pd.DataFrame(all_rows)
                        save_file(intermediate_save_path, all_df)
                        
                # Save first N results if needed
                if save_path and save_first and len(edge_properties) >= save_first and not hasattr(get_edge_structural_properties, '_saved_first'):
                    first_save_path = f"{os.path.splitext(save_path)[0]}_first_{save_first}.csv"
                    first_rows = [edge_prop.to_dict() for edge_prop in edge_properties[:save_first]]
                    first_df = pd.DataFrame(first_rows)
                    save_file(first_save_path, first_df)
                    # Mark that we've saved the first N results
                    setattr(get_edge_structural_properties, '_saved_first', True)
    
    print(f"Successfully processed {len(edge_properties)} out of {len(edges)} database pairs")
    
    # Save final results if path is provided
    if save_path:
        save_start = time.time()
        # Convert to DataFrame
        rows = [edge_prop.to_dict() for edge_prop in edge_properties]
        df = pd.DataFrame(rows)
        save_file(save_path, df, "Final results")
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

    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--threshold", type=str, default='0.94')
    args = parser.parse_args()
    
    edges_df = pd.read_csv(os.path.join(GRAPH_DIR, f"filtered_edges_threshold_{args.threshold}.csv"))
    edges = list(edges_df[["src", "tgt"]].values)
    
    get_edge_structural_properties(
        UNZIP_DIR, 
        edges,
        save_path=os.path.join(GRAPH_DIR, f"edge_structural_properties_GED_{args.threshold}.csv"), 
        num_threads=160,
        structural_path=STRUCTURAL_PATH,
        save_first=None,
        debug=False,
        batch_size=10000
    )
