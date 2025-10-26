import os
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from tqdm import tqdm
import numpy as np
import pickle
from collections import Counter, defaultdict


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



class Column:
    """
    Class representing a column in a table.
    column_name: str
    data_type: str
    wikidata_property_id: str
    """
    def __init__(self, column_name, data_type, wikidata_property_id=None):
        self.column_name = column_name
        self.data_type = data_type
        self.wikidata_property_id = wikidata_property_id
    
    def __repr__(self):
        return f"Column(name={self.column_name}, type={self.data_type})"


class ForeignKey:
    """
    Class representing a foreign key relationship between columns.
    source_column: str
    target_column: str
    target_table: str
    """
    def __init__(self, source_column, target_column, target_table):
        self.source_column = source_column
        self.target_column = target_column
        self.target_table = target_table
    
    def __repr__(self):
        return f"ForeignKey({self.source_column} -> {self.target_table}.{self.target_column})"


class Table:
    """
    Class representing a table in a database.
    table_name: str
    columns: list of Column objects
    foreign_keys: list of ForeignKey objects
    """
    def __init__(self, table_name):
        self.table_name = table_name
        self.columns = []
        self.foreign_keys = []
    
    def add_column(self, column):
        self.columns.append(column)
    
    def add_foreign_key(self, foreign_key):
        self.foreign_keys.append(foreign_key)
    
    def __repr__(self):
        return f"Table(name={self.table_name}, columns={len(self.columns)})"


class Database:
    """
    Class representing a database with tables, columns, and relationships.
    db_id: str
    tables: list of Table objects
    schema: dict
    """
    def __init__(self, db_id):
        self.db_id = db_id
        self.tables = []
        self.schema = None
    
    def add_table(self, table):
        self.tables.append(table)
    
    def load_from_schema(self, schema):
        """Load database structure from a schema dictionary."""
        self.schema = schema
        
        # Extract tables
        tables_data = schema.get("tables", [])
        if not tables_data and "table_name" in schema:
            # Handle the case where schema is a single table definition
            tables_data = [schema]
        
        # Process tables
        for table_data in tables_data:
            table_name = table_data.get("table_name", "")
            table = Table(table_name)
            
            # Process columns
            for column_data in table_data.get("columns", []):
                col_name = column_data.get("column_name", "")
                col_type = column_data.get("data_type", "").lower()
                wikidata_id = column_data.get("wikidata_property_id")
                column = Column(col_name, col_type, wikidata_id)
                table.add_column(column)
            
            # Process foreign keys
            if "foreign_keys" in table_data:
                for fk_data in table_data.get("foreign_keys", []):
                    if isinstance(fk_data, dict):
                        if "column_name" in fk_data and "reference_column_name" in fk_data:
                            fk = ForeignKey(
                                fk_data["column_name"],
                                fk_data["reference_column_name"],
                                fk_data.get("reference_table_name", "")
                            )
                            table.add_foreign_key(fk)
            
            self.add_table(table)
    
    def __repr__(self):
        return f"Database(id={self.db_id}, tables={len(self.tables)})"


class NodeProperty:
    """
    Class representing the structural properties of a database node in the graph.
    db_id: str
    database: Database object
    properties: dict
    """
    def __init__(self, database):
        """Initialize from a Database object."""
        self.db_id = database.db_id
        self.database = database  # Store the full database object for future reference
        self.properties = self._extract_properties(database)
    
    def _extract_properties(self, database):
        """Extract structural properties from a Database object."""
        # Count tables and columns
        num_tables = len(database.tables)
        columns = [(table.table_name, col.column_name) 
                  for table in database.tables 
                  for col in table.columns]
        num_columns = len(columns)
        
        # Count foreign keys
        foreign_keys = []
        for table in database.tables:
            foreign_keys.extend(table.foreign_keys)
        
        # Calculate foreign key density
        fk_density = len(foreign_keys) / num_columns if num_columns > 0 else 0
        
        # Count data types
        data_types = {}
        for table in database.tables:
            for col in table.columns:
                data_types[col.data_type] = data_types.get(col.data_type, 0) + 1
        
        # Calculate data type proportions
        type_proportions = {dtype: count/num_columns 
                           for dtype, count in data_types.items()} if num_columns > 0 else {}
        
        # Calculate table connectivity statistics
        table_connections = defaultdict(set)
        for table in database.tables:
            for fk in table.foreign_keys:
                table_connections[table.table_name].add(fk.target_table)
                table_connections[fk.target_table].add(table.table_name)
        
        # Calculate connectivity statistics
        connectivity_values = [len(connections) for connections in table_connections.values()]
        
        if connectivity_values:
            avg_connectivity = sum(connectivity_values) / len(connectivity_values)
            median_connectivity = np.median(connectivity_values) if connectivity_values else 0
            min_connectivity = min(connectivity_values) if connectivity_values else 0
            max_connectivity = max(connectivity_values) if connectivity_values else 0
        else:
            avg_connectivity = median_connectivity = min_connectivity = max_connectivity = 0
        
        # Count Wikidata properties
        wikidata_props = sum(1 for table in database.tables 
                            for col in table.columns 
                            if col.wikidata_property_id)
        
        # Get table and column lists for future edge calculations
        table_names = [table.table_name for table in database.tables]
        column_names = [col.column_name for table in database.tables for col in table.columns]
        
        return {
            "num_tables": num_tables,
            "num_columns": num_columns,
            "foreign_key_density": fk_density,
            "avg_table_connectivity": avg_connectivity,
            "median_table_connectivity": median_connectivity,
            "min_table_connectivity": min_connectivity,
            "max_table_connectivity": max_connectivity,
            "data_type_proportions": type_proportions,
            "data_types": data_types,
            "wikidata_properties": wikidata_props,
            "table_names": table_names,
            "column_names": column_names
        }
    
    def __getattr__(self, name):
        """Allow direct access to properties as attributes."""
        if name in self.properties:
            return self.properties[name]
        raise AttributeError(f"'{self.__class__.__name__}' has no attribute '{name}'")
    
    def __str__(self):
        """String representation for saving to file."""
        return (f"{self.db_id},{self.properties['num_tables']},"
                f"{self.properties['num_columns']},"
                f"{self.properties['foreign_key_density']:.4f},"
                f"{self.properties['avg_table_connectivity']:.4f},"
                f"{self.properties['median_table_connectivity']:.4f},"
                f"{self.properties['min_table_connectivity']:.4f},"
                f"{self.properties['max_table_connectivity']:.4f},"
                f"\"{str(self.properties['data_type_proportions'])}\"," 
                f"\"{str(self.properties['data_types'])}\"," 
                f"{self.properties['wikidata_properties']}")
    
    @staticmethod
    def get_csv_header():
        """Return the CSV header for saving a collection of NodeProperty objects."""
        return "db_id,num_tables,num_columns,foreign_key_density,avg_table_connectivity,median_table_connectivity,min_table_connectivity,max_table_connectivity,data_type_proportions,data_types,wikidata_properties"


def get_structural_properties(raw_dir: str, save_path: str = None, num_threads: int = None):
    """
    Extract structural properties from database schemas.
    
    Args:
        raw_dir: Directory containing database folders
        save_path: Path to save the extracted properties (optional)
        num_threads: Number of threads to use (default: None, uses CPU count)
        
    Returns:
        List of NodeProperty objects representing database structural properties
    """
    
    def process_single_db(db_path):
        """Process a single database folder to extract structural properties."""
        try:
            # Find schema.json in the database folder
            schema_path = os.path.join(db_path, "schema.json")
            if not os.path.exists(schema_path):
                return None
            
            # Extract database ID from path
            db_id = os.path.basename(db_path).split(" ", 1)[0]
            
            # Load schema
            with open(schema_path, "rb") as f:
                schema = orjson.loads(f.read())
            
            # Create and populate Database object
            database = Database(db_id)
            database.load_from_schema(schema)
            
            # Extract properties
            node_property = NodeProperty(database)
            
            return node_property
        except Exception as e:
            print(f"Error processing {db_path}: {str(e)}")
            return None
    
    # Get all database folders
    db_folders = [os.path.join(raw_dir, folder) for folder in os.listdir(raw_dir) 
                 if os.path.isdir(os.path.join(raw_dir, folder))]
    
    # Process databases in parallel using threads
    node_properties = []
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(process_single_db, folder) for folder in db_folders]
        for future in tqdm(futures, total=len(db_folders), desc="Processing databases"):
            node_prop = future.result()
            if node_prop:
                node_properties.append(node_prop)
    
    # Save results if path is provided
    if save_path:
        with open(save_path, 'w') as f:
            f.write(NodeProperty.get_csv_header() + '\n')
            for node_prop in node_properties:
                f.write(str(node_prop) + '\n')
    
    # Save the full database structure for future edge calculations
    structural_data = [None] * (max(int(node_prop.db_id) for node_prop in node_properties) + 1)
    for node_prop in node_properties:
        structural_data[int(node_prop.db_id)] = node_prop.database
    
    # Save to structural.pkl
    structural_path = os.path.join(os.path.dirname(save_path) if save_path else "data/graph", "structural.pkl")
    with open(structural_path, 'wb') as f:
        pickle.dump(structural_data, f)
    
    return node_properties

if __name__ == "__main__":
    import orjson
    
    UNZIP_DIR = os.path.join("data/unzip")
    GRAPH_DIR = os.path.join("data/graph")
    get_structural_properties(UNZIP_DIR, os.path.join(GRAPH_DIR, "node_structural_properties.csv"), num_threads=32)
