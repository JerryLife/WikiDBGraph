"""
Automated Data Preprocessing for Federated Learning Validation

This module handles automated data cleaning, label selection, and preprocessing
for database pairs sampled for FL experiments.
"""

import pandas as pd
import numpy as np
import os
import json
import sqlite3
import signal
import logging
import traceback
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Any
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')


class TimeoutError(Exception):
    """Custom timeout exception."""
    pass


def timeout_handler(signum, frame):
    """Signal handler for timeout."""
    raise TimeoutError("Operation timed out")

from analysis.NodeProperties import Database


class AutomatedDataPreprocessor:
    """
    Automated data preprocessing for FL validation experiments.
    
    This class handles:
    1. Data loading and table joining
    2. Common column identification and alignment
    3. Label selection for regression tasks
    4. Data cleaning and imputation
    5. Train/test splitting
    6. Data normalization
    """
    
    def __init__(self, 
                 test_size: float = 0.2,
                 random_state: int = 42,
                 min_label_variance: float = 0.01,
                 max_missing_ratio: float = 0.5,
                 join_timeout: int = 300,
                 max_rows: int = 1000000):
        """
        Initialize the data preprocessor.
        
        Args:
            test_size: Fraction of data for test set (default: 0.2)
            random_state: Random state for reproducibility (default: 42)
            min_label_variance: Minimum variance required for regression labels (default: 0.01)
            max_missing_ratio: Maximum missing value ratio allowed per column (default: 0.5)
            join_timeout: Timeout for table joins in seconds (default: 300)
            max_rows: Maximum number of rows allowed in joined table (default: 1,000,000)
        """
        self.test_size = test_size
        self.random_state = random_state
        self.min_label_variance = min_label_variance
        self.max_missing_ratio = max_missing_ratio
        self.join_timeout = join_timeout
        self.max_rows = max_rows
        
        self.unzip_dir = "data/unzip"
    
    def parse_failed_pairs_from_log(self, log_file_path: str) -> List[str]:
        """
        Parse failed pair IDs from error log.
        
        Args:
            log_file_path: Path to the error log file
            
        Returns:
            List of failed pair IDs in format "db1_id-db2_id"
        """
        failed_pairs = set()
        
        if not Path(log_file_path).exists():
            return []
        
        with open(log_file_path, 'r') as f:
            for line in f:
                if "Failed to process pair" in line:
                    # Extract pair ID from line like "Failed to process pair 32356-41801:"
                    import re
                    match = re.search(r'pair (\d+-\d+):', line)
                    if match:
                        # --- FIX: Format the ID with zero-padding ---
                        db_id1_str, db_id2_str = match.group(1).split('-')
                        pair_id = f"{int(db_id1_str):05d}-{int(db_id2_str):05d}"
                        failed_pairs.add(pair_id)
                        # --------------------------------------------
        
        return sorted(list(failed_pairs))
    
    def rotate_error_logs(self, base_dir: str):
        """
        Rename existing error log to backup and create new one.
        
        Args:
            base_dir: Base directory containing error logs
        """
        error_log_path = Path(base_dir) / "preprocessing_errors.log"
        
        if error_log_path.exists():
            # Create backup filename with timestamp
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = Path(base_dir) / f"preprocessing_errors_backup_{timestamp}.log"
            
            # Rename existing log
            error_log_path.rename(backup_path)
            print(f"Previous error log renamed to: {backup_path}")
        
        # Create new empty error log
        error_log_path.touch()
        print(f"Created new error log: {error_log_path}")
        
    def find_best_label_table(self, tables: Dict[str, pd.DataFrame]) -> Tuple[str, str, Dict]:
        """
        Find the best table and label column combination across all tables.
        
        Args:
            tables: Dictionary of table name to DataFrame
            
        Returns:
            Tuple of (best_table_name, best_label_column, label_metadata)
        """
        best_score = -1
        best_table = None
        best_label = None
        best_metadata = None
        
        print("Searching for best label column across all tables...")
        
        for table_name, df in tables.items():
            # Get all columns as potential candidates
            all_columns = list(df.columns)
            
            try:
                # Use the same label selection logic but on individual table
                label_col, label_metadata = self.select_label_column(df, all_columns)
                
                # Score the candidate based on task type and quality
                if label_metadata['task_type'] == 'classification':
                    # For classification, prefer fewer classes (easier to learn)
                    score = 1000 - label_metadata['n_classes']  # Higher score for fewer classes
                else:  # regression
                    # For regression, prefer higher variance (more informative)
                    score = label_metadata['variance']
                
                # Keep track of the best candidate
                if score > best_score:
                    best_score = score
                    best_table = table_name
                    best_label = label_col
                    best_metadata = label_metadata
                    
            except Exception as e:
                # Silently continue to next table if no suitable label found
                continue
        
        if best_table is None:
            raise ValueError("No suitable label column found in any table")
        
        # Only print the final selected label info
        print(f"Selected label: {best_table}.{best_label} (task: {best_metadata['task_type']})")
        if best_metadata['task_type'] == 'classification':
            print(f"  Classes: {best_metadata['n_classes']}")
            print(f"  Class distribution: {list(best_metadata['class_distribution'].keys())[:5]}{'...' if len(best_metadata['class_distribution']) > 5 else ''}")
        else:
            print(f"  Variance: {best_metadata['variance']:.4f}")
            print(f"  Unique values: {best_metadata['n_unique']}")
        
        return best_table, best_label, best_metadata

    def load_database_tables(self, db_id: int, folder_name: str) -> Tuple[pd.DataFrame, str, Dict]:
        """
        Load and join all tables from a database, starting from the table with the best label.
        
        This method:
        1. Loads all tables from the database
        2. Deduplicates columns in each table
        3. Removes constant columns from each table
        4. Finds the best label column across all tables
        5. Starts joining from the table containing the best label
        6. Joins other tables based on common columns or foreign keys
        7. Applies row limits to prevent memory issues
        
        Args:
            db_id: Database ID
            folder_name: Database folder name
            
        Returns:
            Tuple of (joined_dataframe, label_column, label_metadata)
        """
        folder_path = os.path.join(self.unzip_dir, folder_name)
        database_path = os.path.join(folder_path, "database.db")
        
        if not os.path.exists(database_path):
            raise FileNotFoundError(f"Database file not found: {database_path}")
        
        # Load schema information
        schema_path = os.path.join(folder_path, "schema.json")
        with open(schema_path, 'r') as f:
            schema = json.load(f)
        
        # Create Database object to understand structure
        db = Database(db_id)
        db.load_from_schema(schema)
        
        # Connect to SQLite database
        with sqlite3.connect(database_path) as conn:
            # Get all table names
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            table_names = [row[0] for row in cursor.fetchall()]
            
            # Load all tables
            tables = {}
            for table_name in table_names:
                try:
                    tables[table_name] = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
                    print(f"Loaded table {table_name}: {tables[table_name].shape}")
                except Exception as e:
                    print(f"Warning: Could not load table {table_name}: {e}")
                    continue
            
            if not tables:
                raise ValueError(f"No tables could be loaded from database {db_id}")
            
            # Deduplicate columns in each table
            for table_name in tables:
                original_shape = tables[table_name].shape
                tables[table_name] = self.deduplicate_columns_by_snapshot(tables[table_name])
                new_shape = tables[table_name].shape
                if original_shape != new_shape:
                    print(f"Deduplicated {table_name}: {original_shape} -> {new_shape}")
            
            # Remove constant columns from each table before joining
            tables_to_remove = []
            for table_name in tables:
                original_shape = tables[table_name].shape
                tables[table_name] = self.remove_constant_columns_single_table(tables[table_name])
                new_shape = tables[table_name].shape
                if original_shape != new_shape:
                    print(f"Removed constant columns from {table_name}: {original_shape} -> {new_shape}")
                
                # Check if table still has columns after removing constants
                if tables[table_name].shape[1] == 0:
                    print(f"Warning: Table {table_name} has no non-constant columns, marking for removal")
                    tables_to_remove.append(table_name)
            
            # Remove tables with no non-constant columns
            for table_name in tables_to_remove:
                del tables[table_name]
            
            # Check if we still have tables to work with
            if not tables:
                raise ValueError("All tables contain only constant columns")
            
            # Find the best label column across all tables
            best_table_name, best_label_col, label_metadata = self.find_best_label_table(tables)
            
            # If only one table, return it directly with label info
            if len(tables) == 1:
                return list(tables.values())[0], best_label_col, label_metadata
            
            # Try to join tables based on foreign key relationships (with timeout)
            # Start from the table containing the best label
            try:
                # Set up timeout signal
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(self.join_timeout)
                
                joined_df = self._join_tables_from_label_table(tables, db.tables, best_table_name)
                
                # Cancel timeout
                signal.alarm(0)
                
                return joined_df, best_label_col, label_metadata
                
            except TimeoutError:
                print(f"Warning: Table join timed out after {self.join_timeout}s, using label table only")
                # Cancel timeout
                signal.alarm(0)
                # Return the label table as fallback
                return tables[best_table_name], best_label_col, label_metadata
            except Exception as e:
                # Cancel timeout
                signal.alarm(0)
                print(f"Warning: Table join failed: {e}, using label table only")
                # Return the label table as fallback
                return tables[best_table_name], best_label_col, label_metadata
    
    def _join_tables_from_label_table(self, tables: Dict[str, pd.DataFrame], table_objects: List, 
                                      label_table_name: str) -> pd.DataFrame:
        """
        Join tables starting from the table containing the best label column.
        
        Args:
            tables: Dictionary of table name to DataFrame
            table_objects: List of Table objects with foreign key info
            label_table_name: Name of the table containing the best label
            
        Returns:
            Joined DataFrame
        """
        # Start with the label table as base
        joined_df = tables[label_table_name].copy()
        
        print(f"Starting join with label table: {label_table_name} ({joined_df.shape})")
        
        # Check if base table already exceeds row limit
        if len(joined_df) > self.max_rows:
            print(f"Warning: Label table {label_table_name} has {len(joined_df):,} rows, exceeding limit of {self.max_rows:,}")
            print(f"Truncating to first {self.max_rows:,} rows")
            joined_df = joined_df.head(self.max_rows)
            return joined_df
        
        # Try to join other tables
        for table_name, df in tables.items():
            if table_name == label_table_name:
                continue

            # --- FIX: Add this line to prevent complex suffix collisions ---
            joined_df = joined_df.loc[:,~joined_df.columns.duplicated()]
            # -----------------------------------------------------------------
            
            # --- CORRECTED LOGIC: ONLY use foreign keys from schema ---
            # Find valid join relationships using ONLY foreign keys from schema
            valid_joins = self._find_foreign_key_joins(joined_df, df, table_name, table_objects, label_table_name)
            
            # ONLY if a valid FK join exists, proceed with the merge
            if valid_joins:
                # Use the first valid foreign key relationship found
                join_info = valid_joins[0]
                join_col = join_info['join_column']
                
                # Check if the potential join key has only one unique value in the right table
                if df[join_col].nunique(dropna=True) <= 1:
                    print(f"Warning: Skipping join on constant foreign key '{join_col}' for table {table_name}")
                    continue # Skip this table entirely

                # Proceed with the foreign key join
                try:
                    # Get names of conflicting columns (excluding the join key)
                    conflicting_cols = set(joined_df.columns) & set(df.columns) - {join_col}
                    
                    # Rename conflicting columns in the right table before merging
                    df_renamed = df.rename(columns={
                        col: f"{col}_{table_name}" for col in conflicting_cols
                    })

                    # Perform the merge on the validated foreign key column
                    before_shape = joined_df.shape
                    joined_df = joined_df.merge(df_renamed, on=join_col, how='left')
                    after_shape = joined_df.shape

                    print(f"Joined {table_name} on FOREIGN KEY '{join_col}': {before_shape} -> {after_shape}")
                    
                    # Check if joined table exceeds row limit
                    if len(joined_df) > self.max_rows:
                        print(f"Warning: Joined table now has {len(joined_df):,} rows, exceeding limit of {self.max_rows:,}")
                        print(f"Stopping table joins and truncating to first {self.max_rows:,} rows")
                        joined_df = joined_df.head(self.max_rows)
                        break  # Stop joining more tables
                        
                except Exception as e:
                    print(f"Warning: Could not join table {table_name} on foreign key '{join_col}': {e}")
            else:
                # If no FK is found, DO NOT JOIN - this is the key fix
                print(f"Warning: No valid foreign key relationship found to join table {table_name}. SKIPPING table.")
        
        return joined_df

    def _find_foreign_key_joins(self, left_df: pd.DataFrame, right_df: pd.DataFrame, 
                               right_table_name: str, table_objects: List, 
                               base_table_name: str) -> List[Dict]:
        """
        Find valid join relationships using foreign key constraints from schema.
        
        Args:
            left_df: Left DataFrame (already joined tables)
            right_df: Right DataFrame (table to join)
            right_table_name: Name of the right table
            table_objects: List of Table objects with foreign key info
            base_table_name: Name of the base/label table
            
        Returns:
            List of valid join information dictionaries
        """
        valid_joins = []
        
        # Find the table object for the right table
        right_table_obj = None
        for table_obj in table_objects:
            if table_obj.table_name == right_table_name:
                right_table_obj = table_obj
                break
        
        if not right_table_obj:
            print(f"Warning: Table object not found for {right_table_name}")
            return []
        
        # Check foreign keys from right table to any table in the left side
        for fk in right_table_obj.foreign_keys:
            # Check if the foreign key column exists in right table
            if fk.column_name not in right_df.columns:
                continue
                
            # Check if the referenced column exists in left dataframe
            if fk.reference_column_name not in left_df.columns:
                continue
            
            # For pandas merge, we need the same column name in both dataframes
            # If foreign key column name != reference column name, we need to handle it
            if fk.column_name == fk.reference_column_name:
                join_column = fk.column_name
            else:
                # Need to rename one of the columns for the join
                join_column = fk.column_name  # Use the foreign key column name
                print(f"Warning: Foreign key column '{fk.column_name}' != reference column '{fk.reference_column_name}' - manual column alignment may be needed")
                continue  # Skip for now - would need column renaming logic
            
            # Valid foreign key relationship found
            join_info = {
                'join_column': join_column,
                'reference_column': fk.reference_column_name,
                'reference_table': fk.reference_table_name
            }
            valid_joins.append(join_info)
            print(f"Found foreign key: {right_table_name}.{fk.column_name} -> {fk.reference_table_name}.{fk.reference_column_name}")
        
        # Also check if any table in the left side has foreign keys pointing to right table
        for table_obj in table_objects:
            # Skip if this table is not represented in the left dataframe
            # (We check by looking for table-specific column prefixes or base table)
            if table_obj.table_name != base_table_name:
                # Check if any columns from this table exist in left_df
                table_cols_in_left = any(col.startswith(f"{table_obj.table_name}_") or col in [c.column_name for c in table_obj.columns] 
                                       for col in left_df.columns)
                if not table_cols_in_left:
                    continue
            
            for fk in table_obj.foreign_keys:
                # Check if this foreign key points to the right table
                if fk.reference_table_name != right_table_name:
                    continue
                    
                # Check if foreign key column exists in left dataframe
                if fk.column_name not in left_df.columns:
                    continue
                    
                # Check if referenced column exists in right table
                if fk.reference_column_name not in right_df.columns:
                    continue
                
                # For reverse foreign key, check column name consistency
                if fk.column_name == fk.reference_column_name:
                    join_column = fk.reference_column_name
                else:
                    print(f"Warning: Reverse foreign key column '{fk.column_name}' != reference column '{fk.reference_column_name}' - manual column alignment may be needed")
                    continue  # Skip for now
                
                # Valid reverse foreign key relationship found
                join_info = {
                    'join_column': join_column,  # Column to join on
                    'reference_column': fk.column_name,  # Column in left table (foreign key)
                    'reference_table': table_obj.table_name
                }
                valid_joins.append(join_info)
                print(f"Found reverse foreign key: {table_obj.table_name}.{fk.column_name} -> {right_table_name}.{fk.reference_column_name}")
        
        return valid_joins

    def _join_tables(self, tables: Dict[str, pd.DataFrame], table_objects: List) -> pd.DataFrame:
        """
        Join tables based on foreign key relationships or common columns.
        
        Args:
            tables: Dictionary of table name to DataFrame
            table_objects: List of Table objects with foreign key info
            
        Returns:
            Joined DataFrame
        """
        # Start with the largest table as base
        base_table_name = max(tables.keys(), key=lambda k: len(tables[k]))
        joined_df = tables[base_table_name].copy()
        
        print(f"Starting join with base table: {base_table_name} ({joined_df.shape})")
        
        # Check if base table already exceeds row limit
        if len(joined_df) > self.max_rows:
            print(f"Warning: Base table {base_table_name} has {len(joined_df):,} rows, exceeding limit of {self.max_rows:,}")
            print(f"Truncating to first {self.max_rows:,} rows")
            joined_df = joined_df.head(self.max_rows)
            return joined_df
        
        # Try to join other tables
        for table_name, df in tables.items():
            if table_name == base_table_name:
                continue
            
            # --- CORRECTED LOGIC: ONLY use foreign keys from schema ---
            # Find valid join relationships using ONLY foreign keys from schema
            valid_joins = self._find_foreign_key_joins(joined_df, df, table_name, table_objects, base_table_name)
            
            # ONLY if a valid FK join exists, proceed with the merge
            if valid_joins:
                # Use the first valid foreign key relationship found
                join_info = valid_joins[0]
                join_col = join_info['join_column']
                
                # Check if the potential join key has only one unique value in the right table
                if df[join_col].nunique(dropna=True) <= 1:
                    print(f"Warning: Skipping join on constant foreign key '{join_col}' for table {table_name}")
                    continue # Skip this table entirely

                # Proceed with the foreign key join
                try:
                    # Get names of conflicting columns (excluding the join key)
                    conflicting_cols = set(joined_df.columns) & set(df.columns) - {join_col}
                    
                    # Rename conflicting columns in the right table before merging
                    df_renamed = df.rename(columns={
                        col: f"{col}_{table_name}" for col in conflicting_cols
                    })

                    # Perform the merge on the validated foreign key column
                    before_shape = joined_df.shape
                    joined_df = joined_df.merge(df_renamed, on=join_col, how='left')
                    after_shape = joined_df.shape

                    print(f"Joined {table_name} on FOREIGN KEY '{join_col}': {before_shape} -> {after_shape}")
                    
                    # Check if joined table exceeds row limit
                    if len(joined_df) > self.max_rows:
                        print(f"Warning: Joined table now has {len(joined_df):,} rows, exceeding limit of {self.max_rows:,}")
                        print(f"Stopping table joins and truncating to first {self.max_rows:,} rows")
                        joined_df = joined_df.head(self.max_rows)
                        break  # Stop joining more tables
                        
                except Exception as e:
                    print(f"Warning: Could not join table {table_name} on foreign key '{join_col}': {e}")
            else:
                # If no FK is found, DO NOT JOIN - this is the key fix
                print(f"Warning: No valid foreign key relationship found to join table {table_name}. SKIPPING table.")
        
        return joined_df
    
    def create_column_snapshot(self, column_name: str) -> str:
        """Create column snapshot by removing non-alpha symbols and converting to lowercase."""
        return ''.join(c for c in column_name.lower() if c.isalnum())
    
    def deduplicate_columns_by_snapshot(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove duplicate columns based on column snapshots, keeping only the first occurrence.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with deduplicated columns
        """
        seen_snapshots = set()
        columns_to_keep = []
        
        for col in df.columns:
            snapshot = self.create_column_snapshot(col)
            if snapshot not in seen_snapshots and snapshot:  # Skip empty snapshots
                seen_snapshots.add(snapshot)
                columns_to_keep.append(col)
            else:
                print(f"Removing duplicate column: {col} (snapshot: {snapshot})")
        
        return df[columns_to_keep]
    
    def identify_common_columns(self, df1: pd.DataFrame, df2: pd.DataFrame) -> Tuple[List[str], pd.DataFrame]:
        """
        Identify common columns between two DataFrames using column snapshots.
        Returns aligned column names and renames df2 to match df1.
        
        Args:
            df1: First DataFrame
            df2: Second DataFrame
            
        Returns:
            Tuple of (common column names, df2 with renamed columns)
        """
        # Create snapshot mappings (handle multiple columns with same snapshot)
        df1_snapshots = {}
        for col in df1.columns:
            snapshot = self.create_column_snapshot(col)
            if snapshot:  # Skip empty snapshots
                if snapshot in df1_snapshots:
                    print(f"Warning: Column snapshot collision in df1 - '{col}' -> '{snapshot}' (keeping '{df1_snapshots[snapshot]}')")
                else:
                    df1_snapshots[snapshot] = col
        
        df2_snapshots = {}
        for col in df2.columns:
            snapshot = self.create_column_snapshot(col)
            if snapshot:  # Skip empty snapshots
                if snapshot in df2_snapshots:
                    print(f"Warning: Column snapshot collision in df2 - '{col}' -> '{snapshot}' (keeping '{df2_snapshots[snapshot]}')")
                else:
                    df2_snapshots[snapshot] = col
        
        # Find common snapshots
        common_snapshots = set(df1_snapshots.keys()) & set(df2_snapshots.keys())
        
        # Create rename mapping for df2 to match df1 column names
        rename_map = {df2_snapshots[snapshot]: df1_snapshots[snapshot] for snapshot in common_snapshots}
        df2_aligned = df2.rename(columns=rename_map)
        
        # Get common column names (using df1 names as reference)
        common_columns = [df1_snapshots[snapshot] for snapshot in common_snapshots]
        
        print(f"Found {len(common_columns)} common columns: {common_columns[:5]}{'...' if len(common_columns) > 5 else ''}")
        print(f"Renamed {len(rename_map)} columns in df2 to align with df1")
        
        return common_columns, df2_aligned
    
    def select_label_column(self, df: pd.DataFrame, common_columns: List[str]) -> Tuple[str, Dict]:
        """
        Automatically select a suitable column for classification or regression tasks.
        Prioritizes classification, falls back to regression.
        
        Args:
            df: DataFrame to analyze
            common_columns: List of common columns to consider
            
        Returns:
            Tuple of (selected_column, metadata_dict)
        """
        # First, try to find classification candidates
        classification_candidates = []
        regression_candidates = []
        
        for col in common_columns:
            if col not in df.columns:
                continue
                
            # Skip if too many missing values
            missing_ratio = df[col].isnull().sum() / len(df)
            if missing_ratio > self.max_missing_ratio:
                continue
            
            # Get unique values (excluding NaN)
            unique_values = df[col].dropna().unique()
            n_unique = len(unique_values)
            
            # Skip constant columns - must have at least 2 unique values
            if n_unique <= 1:
                continue
            
            # Double-check for constant columns using the same logic as our cleaning process
            try:
                unique_vals_robust = df[col].nunique(dropna=True)
                if unique_vals_robust <= 1:
                    continue  # Skip truly constant columns
            except:
                pass  # If check fails, continue with original n_unique check
            
            # Check for classification (categorical with reasonable number of classes)
            if 2 <= n_unique <= 50:
                # Try to determine if it's categorical
                is_categorical = False
                
                # Check if values are strings or mixed types
                sample_values = df[col].dropna().head(100)
                if sample_values.dtype == 'object' or any(isinstance(v, str) for v in sample_values):
                    is_categorical = True
                else:
                    # Check if numeric values are integers and reasonable for categories
                    try:
                        numeric_values = pd.to_numeric(sample_values, errors='coerce')
                        if not numeric_values.isnull().any():
                            # If all are integers and range is reasonable, could be categorical
                            if all(float(v).is_integer() for v in numeric_values) and n_unique <= 20:
                                is_categorical = True
                    except:
                        pass
                
                if is_categorical:
                    classification_candidates.append({
                        'column': col,
                        'task_type': 'classification',
                        'n_classes': n_unique,
                        'missing_ratio': missing_ratio,
                        'class_distribution': df[col].value_counts().to_dict()
                    })
                    continue
            
            # Try for regression (continuous numeric variables)
            try:
                numeric_series = pd.to_numeric(df[col], errors='coerce')
                
                # Skip if too many non-numeric values after conversion
                numeric_missing_ratio = numeric_series.isnull().sum() / len(numeric_series)
                if numeric_missing_ratio > self.max_missing_ratio:
                    continue
                
                # Calculate variance (for non-constant columns)
                variance = numeric_series.var()
                if pd.isna(variance) or variance < self.min_label_variance:
                    continue
                
                # For regression, prefer columns with many unique values
                if n_unique > 50:  # Good for regression
                    mean_val = numeric_series.mean()
                    std_val = numeric_series.std()
                    
                    regression_candidates.append({
                        'column': col,
                        'task_type': 'regression',
                        'variance': variance,
                        'n_unique': n_unique,
                        'mean': mean_val,
                        'std': std_val,
                        'missing_ratio': missing_ratio,
                        'numeric_missing_ratio': numeric_missing_ratio
                    })
                    
            except Exception as e:
                continue
        
        # Prioritize classification
        if classification_candidates:
            # Select classification column with minimum number of classes (but > 1)
            best_candidate = min(classification_candidates, key=lambda x: x['n_classes'])
            selected_col = best_candidate['column']
            return selected_col, best_candidate
        
        # Fall back to regression
        elif regression_candidates:
            # Select regression column with highest variance (most informative)
            best_candidate = max(regression_candidates, key=lambda x: x['variance'])
            selected_col = best_candidate['column']
            return selected_col, best_candidate
        
        else:
            raise ValueError("No suitable label column found for either classification or regression")
    
    def remove_constant_columns_single_table(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove columns that have constant values in a single table.
        Robust handling of both numeric and categorical (string) features.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with constant columns removed
        """
        columns_to_keep = []
        
        for col in df.columns:
            try:
                # First check unique values (works for all data types)
                unique_vals = df[col].nunique(dropna=True)  # Exclude NaN from count
                
                # --- REFINED LOGIC ---
                if unique_vals > 1:
                    # More than one unique value, definitely keep
                    columns_to_keep.append(col)
                elif unique_vals == 1:
                    # Check if there are also NaNs, which would make it a two-category feature
                    if df[col].isnull().any():
                        columns_to_keep.append(col)
                        # print(f"  Keeping column with missing values: {col} (1 unique value + NaNs)")
                    else:
                        print(f"  Removing constant column: {col} (1 unique value, no NaNs)")
                else: # unique_vals == 0 (all NaN)
                    print(f"  Removing empty column: {col} (all NaN)")
                # --- END REFINED LOGIC ---
                    
            except Exception as e:
                print(f"  Warning: Could not analyze column {col}: {e}")
                # Keep the column if we can't analyze it (be conservative)
                columns_to_keep.append(col)
        
        return df[columns_to_keep].copy() if columns_to_keep else df
    
    def remove_constant_columns(self, df1: pd.DataFrame, df2: pd.DataFrame, 
                              common_columns: List[str], label_col: str = None) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
        """
        Remove columns that have constant values across both datasets.
        Protects the label column from removal.
        
        Args:
            df1: First DataFrame
            df2: Second DataFrame
            common_columns: List of common columns
            label_col: Label column to protect from removal
            
        Returns:
            Tuple of (filtered_df1, filtered_df2, filtered_columns)
        """
        columns_to_keep = []
        
        for col in common_columns:
            # Always keep the label column, even if it appears constant
            if label_col and col == label_col:
                columns_to_keep.append(col)
                continue
                
            # Check if column has variation in either dataset using robust method
            try:
                # Use the same robust method as single table check
                unique1 = df1[col].nunique(dropna=True)
                unique2 = df2[col].nunique(dropna=True)
                
                # --- REFINED LOGIC ---
                # Condition 1: Keep if either dataframe has more than 1 unique non-NaN value.
                has_variance = unique1 > 1 or unique2 > 1
                
                # Condition 2: Keep if a column has one value in one df and NaNs in the other,
                # creating a combined feature with variance.
                is_mixed_nan = (unique1 == 1 and df2[col].isnull().all()) or \
                               (unique2 == 1 and df1[col].isnull().all()) or \
                               (unique1 == 1 and unique2 == 1 and 
                                len(df1[col].dropna()) > 0 and len(df2[col].dropna()) > 0 and 
                                df1[col].dropna().iloc[0] != df2[col].dropna().iloc[0])

                if has_variance or is_mixed_nan:
                    columns_to_keep.append(col)
                else:
                    print(f"Removing constant column across both dataframes: {col}")
                # --- END REFINED LOGIC ---
                    
            except Exception as e:
                # If we can't analyze the column, keep it to be safe
                print(f"  Warning: Could not analyze column {col}: {e}")
                columns_to_keep.append(col)
        
        print(f"Kept {len(columns_to_keep)} non-constant columns out of {len(common_columns)}")
        
        return df1[columns_to_keep].copy(), df2[columns_to_keep].copy(), columns_to_keep
    
    def clean_and_process_data(self, df1: pd.DataFrame, df2: pd.DataFrame, 
                             common_columns: List[str], label_col: str, task_type: str) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
        """
        Clean and process DataFrames with type-aware constant detection.
        Keep categorical features as strings, apply label encoding only at the end.
        
        Args:
            df1: First database DataFrame
            df2: Second database DataFrame
            common_columns: List of common columns
            label_col: Selected label column
            task_type: Task type ('classification' or 'regression')
            
        Returns:
            Tuple of (cleaned_df1, cleaned_df2, column_types)
        """
        # Keep only common columns
        df1_processed = df1[common_columns].copy()
        df2_processed = df2[common_columns].copy()
        
        print(f"After selecting common columns: df1={df1_processed.shape}, df2={df2_processed.shape}")
        
        # 1. Drop rows with missing labels first
        df1_processed = df1_processed.dropna(subset=[label_col])
        df2_processed = df2_processed.dropna(subset=[label_col])

        if df1_processed.empty or df2_processed.empty:
            raise ValueError("DataFrame became empty after removing missing labels.")

        print(f"After removing missing labels: df1={df1_processed.shape}, df2={df2_processed.shape}")

        columns_to_keep = [label_col]
        column_types = {}
        feature_cols = [col for col in common_columns if col != label_col]
        
        # 2. Process each feature column based on its inferred type
        for col in feature_cols:
            # Combine both dataframes to analyze the column type
            combined_series = pd.concat([df1_processed[col], df2_processed[col]], ignore_index=True)
            
            # Try to infer if this is numeric
            numeric_series = pd.to_numeric(combined_series, errors='coerce')
            
            if numeric_series.notna().sum() > 0 and (numeric_series.notna().sum() / len(combined_series)) > 0.5:
                # --- NUMERIC COLUMN LOGIC ---
                # Convert to numeric in both dataframes
                df1_processed[col] = pd.to_numeric(df1_processed[col], errors='coerce')
                df2_processed[col] = pd.to_numeric(df2_processed[col], errors='coerce')
                
                # Check for non-zero variance (constant detection for numeric)
                combined_numeric = pd.concat([df1_processed[col], df2_processed[col]], ignore_index=True)
                if combined_numeric.var(skipna=True) > 1e-6:  # Non-constant
                    columns_to_keep.append(col)
                    column_types[col] = 'numeric'
                else:
                    print(f"Removing constant numeric column: {col}")
            else:
                # --- CATEGORICAL (STRING) COLUMN LOGIC ---
                # Keep as string/object, don't apply label encoding yet
                df1_processed[col] = df1_processed[col].astype(str)
                df2_processed[col] = df2_processed[col].astype(str)
                
                # For categorical, check unique values including NaN representation
                combined_categorical = pd.concat([df1_processed[col], df2_processed[col]], ignore_index=True)
                # nunique() will count 'nan' string as a category if it exists
                if combined_categorical.nunique() > 1:
                    columns_to_keep.append(col)
                    column_types[col] = 'categorical'
                else:
                    print(f"Removing constant categorical column: {col}")

        # 3. Process the label column
        if task_type == 'classification':
            df1_processed[label_col] = df1_processed[label_col].astype(str)
            df2_processed[label_col] = df2_processed[label_col].astype(str)
            column_types[label_col] = 'categorical_label'
        else:  # Regression
            df1_processed[label_col] = pd.to_numeric(df1_processed[label_col], errors='coerce')
            df2_processed[label_col] = pd.to_numeric(df2_processed[label_col], errors='coerce')
            column_types[label_col] = 'numeric_label'

        # Filter dataframes to only keep the processed, non-constant columns
        df1_final = df1_processed[columns_to_keep].copy()
        df2_final = df2_processed[columns_to_keep].copy()
        
        print(f"Final cleaned shapes: df1={df1_final.shape}, df2={df2_final.shape}")
        print(f"Column types: {column_types}")
        
        return df1_final, df2_final, column_types
    
    def apply_label_encoding_for_save(self, df: pd.DataFrame, column_types: Dict) -> pd.DataFrame:
        """
        Apply label encoding to categorical columns just before saving to CSV.
        Keep the original dataframe unchanged, return encoded copy.
        
        Args:
            df: DataFrame to encode
            column_types: Dictionary mapping column names to their types

        Returns:
            DataFrame with categorical columns label-encoded
        """
        from sklearn.preprocessing import LabelEncoder
        
        df_encoded = df.copy()
        
        for col, col_type in column_types.items():
            if col_type in ['categorical', 'categorical_label']:
                # Apply label encoding
                encoder = LabelEncoder()
                df_encoded[col] = encoder.fit_transform(df_encoded[col].astype(str))
        
        return df_encoded
    
    def process_labels(self, df1: pd.DataFrame, df2: pd.DataFrame, label_col: str, 
                      task_type: str) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
        """
        Process label columns based on task type (classification or regression).
        
        Args:
            df1: First DataFrame
            df2: Second DataFrame  
            label_col: Label column name
            task_type: 'classification' or 'regression'
            
        Returns:
            Tuple of (processed_df1, processed_df2, processing_params)
        """
        df1_processed = df1.copy()
        df2_processed = df2.copy()
        
        if task_type == 'classification':
            # For classification, encode labels as integers
            from sklearn.preprocessing import LabelEncoder
            
            # Combine all labels to get consistent encoding
            combined_labels = pd.concat([df1[label_col], df2[label_col]])
            combined_labels_clean = combined_labels.dropna().astype(str)
            
            label_encoder = LabelEncoder()
            label_encoder.fit(combined_labels_clean)
            
            # Encode labels for both datasets
            df1_processed[label_col] = label_encoder.transform(df1[label_col].astype(str))
            df2_processed[label_col] = label_encoder.transform(df2[label_col].astype(str))
            
            processing_params = {
                'task_type': 'classification',
                'label_col': label_col,
                'n_classes': len(label_encoder.classes_),
                'class_names': label_encoder.classes_.tolist(),
                'label_encoder': label_encoder.classes_.tolist()  # Store class names for decoding
            }
            
            print(f"Encoded classification labels: {len(label_encoder.classes_)} classes")
            print(f"  Classes: {label_encoder.classes_[:10]}{'...' if len(label_encoder.classes_) > 10 else ''}")
            
        else:  # regression
            # For regression, normalize labels to [0, 1] range
            combined_labels = pd.concat([df1[label_col], df2[label_col]])
            
            # Convert to numeric first
            df1_processed[label_col] = pd.to_numeric(df1[label_col], errors='coerce')
            df2_processed[label_col] = pd.to_numeric(df2[label_col], errors='coerce')
            combined_labels = pd.to_numeric(combined_labels, errors='coerce')
            
            label_min = combined_labels.min()
            label_max = combined_labels.max()
            label_range = label_max - label_min
            
            if label_range == 0:
                raise ValueError(f"Label column {label_col} has zero variance")
            
            # Normalize to [0, 1]
            df1_processed[label_col] = (df1_processed[label_col] - label_min) / label_range
            df2_processed[label_col] = (df2_processed[label_col] - label_min) / label_range
            
            processing_params = {
                'task_type': 'regression',
                'label_col': label_col,
                'label_min': float(label_min),
                'label_max': float(label_max),
                'label_range': float(label_range)
            }
            
            print(f"Normalized regression labels to [0, 1]: original range [{label_min:.4f}, {label_max:.4f}]")
        
        return df1_processed, df2_processed, processing_params
    
    def split_data(self, df1: pd.DataFrame, df2: pd.DataFrame, label_col: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train/test sets.
        
        Args:
            df1: First DataFrame
            df2: Second DataFrame
            label_col: Label column name
            
        Returns:
            Tuple of (df1_train, df1_test, df2_train, df2_test)
        """
        # Split each dataset independently
        df1_train, df1_test = train_test_split(
            df1, test_size=self.test_size, random_state=self.random_state, stratify=None
        )
        
        df2_train, df2_test = train_test_split(
            df2, test_size=self.test_size, random_state=self.random_state, stratify=None
        )
        
        print(f"Train/test split:")
        print(f"  DB1 train: {df1_train.shape}, test: {df1_test.shape}")
        print(f"  DB2 train: {df2_train.shape}, test: {df2_test.shape}")
        
        return df1_train, df1_test, df2_train, df2_test
    
    def process_pair(self, pair_metadata: Dict, output_dir: str) -> Dict:
        """
        Process a single database pair for FL training.
        
        Args:
            pair_metadata: Metadata dictionary for the database pair
            output_dir: Output directory for processed data
            
        Returns:
            Processing configuration and results
        """
        db_id1 = pair_metadata['db_id1']
        db_id2 = pair_metadata['db_id2']
        folder1 = pair_metadata['folder1']
        folder2 = pair_metadata['folder2']
        
        print(f"\nProcessing pair: {db_id1} - {db_id2}")
        print(f"Similarity: {pair_metadata['similarity']:.4f}")
        
        try:
            # Load database tables
            print("Loading database tables...")
            df1, _, _ = self.load_database_tables(db_id1, folder1)
            df2, _, _ = self.load_database_tables(db_id2, folder2)
            
            # Identify common columns and align df2 column names with df1
            common_columns, df2_aligned = self.identify_common_columns(df1, df2)
            
            if len(common_columns) < 3:  # Need at least 2 features + 1 label
                raise ValueError(f"Insufficient common columns: {len(common_columns)} (minimum: 3)")
            
            # Select label from the common columns using combined data
            combined_df = pd.concat([df1[common_columns], df2_aligned[common_columns]], ignore_index=True)
            label_col, label_metadata = self.select_label_column(combined_df, common_columns)
            task_type = label_metadata['task_type']
            
            print(f"Selected label from common columns: {label_col} (Task: {task_type})")
            
            # Clean and align data (df2_aligned already has matching column names)
            df1_clean, df2_clean, column_types = self.clean_and_process_data(df1, df2_aligned, common_columns, label_col, task_type)
            
            # --- Add check for empty dataframes ---
            if df1_clean.empty or df2_clean.empty:
                raise ValueError("DataFrame became empty after cleaning, no valid labels found.")
            # ---------------------------------------------
            
            # --- Update common_columns to reflect the actual columns after constant removal ---
            # The clean_and_process_data function handles constant removal internally
            # So we need to update our common_columns list to match the actual cleaned data
            actual_common_columns = list(df1_clean.columns)
            print(f"Updated common columns after constant removal: {len(common_columns)} -> {len(actual_common_columns)}")
            common_columns = actual_common_columns
            # -----------------------------------------------------------------------------------
            
            # Validate minimum feature requirements
            # After cleaning, we need at least 2 feature columns (plus 1 label = 3 total minimum)
            available_columns = list(df1_clean.columns)
            feature_columns = [col for col in available_columns if col != label_col]
            
            if len(feature_columns) < 2:
                raise ValueError(f"Insufficient feature columns after cleaning: {len(feature_columns)} (minimum: 2). "
                               f"Available columns: {available_columns}, Label: {label_col}")
            
            print(f"Validation passed: {len(feature_columns)} feature columns available")
            
            # The label processing is now handled in clean_and_process_data
            # Extract processing parameters from column_types
            if task_type == 'classification':
                # Get unique classes for classification
                combined_labels = pd.concat([df1_clean[label_col], df2_clean[label_col]], ignore_index=True)
                unique_classes = sorted(combined_labels.unique())
                processing_params = {
                    'task_type': task_type,
                    'label_col': label_col,
                    'n_classes': len(unique_classes),
                    'class_names': unique_classes
                }
            else:
                processing_params = {
                    'task_type': task_type,
                    'label_col': label_col
                }
            
            df1_processed = df1_clean
            df2_processed = df2_clean
            
            # Split data
            df1_train, df1_test, df2_train, df2_test = self.split_data(df1_processed, df2_processed, label_col)
            
            # Create output directory only after successful processing
            pair_dir = Path(output_dir) / f"{db_id1:05d}_{db_id2:05d}"
            pair_dir.mkdir(parents=True, exist_ok=True)
            
            # Apply label encoding before saving (categorical features become numeric)
            df1_train_encoded = self.apply_label_encoding_for_save(df1_train, column_types)
            df1_test_encoded = self.apply_label_encoding_for_save(df1_test, column_types)
            df2_train_encoded = self.apply_label_encoding_for_save(df2_train, column_types)
            df2_test_encoded = self.apply_label_encoding_for_save(df2_test, column_types)
            
            # Save processed data (now with categorical features encoded as numbers)
            df1_train_encoded.to_csv(pair_dir / f"{db_id1:05d}_train.csv", index=False)
            df1_test_encoded.to_csv(pair_dir / f"{db_id1:05d}_test.csv", index=False)
            df2_train_encoded.to_csv(pair_dir / f"{db_id2:05d}_train.csv", index=False)
            df2_test_encoded.to_csv(pair_dir / f"{db_id2:05d}_test.csv", index=False)
            
            # Create configuration
            config = {
                'pair_id': f"{db_id1:05d}_{db_id2:05d}",
                'db_id1': db_id1,
                'db_id2': db_id2,
                'similarity': pair_metadata['similarity'],
                'task_type': task_type,
                'label_column': label_col,
                'label_metadata': label_metadata,
                'processing_params': processing_params,
                'common_columns': common_columns,
                'num_common_columns': len(common_columns),
                'data_shapes': {
                    'db1_train': df1_train.shape,
                    'db1_test': df1_test.shape,
                    'db2_train': df2_train.shape,
                    'db2_test': df2_test.shape
                },
                'feature_columns': [col for col in common_columns if col != label_col],
                'column_types': column_types,
                'preprocessing_config': {
                    'test_size': self.test_size,
                    'random_state': self.random_state,
                    'min_label_variance': self.min_label_variance,
                    'max_missing_ratio': self.max_missing_ratio,
                    'max_rows': self.max_rows
                },
                'status': 'success'
            }
            
            # Save configuration
            with open(pair_dir / 'config.json', 'w') as f:
                json.dump(config, f, indent=2, default=str)
            
            print(f"Successfully processed pair {db_id1}-{db_id2}")
            print(f"  Task type: {task_type}")
            print(f"  Label: {label_col}")
            print(f"  Features: {len(config['feature_columns'])}")
            if task_type == 'classification':
                print(f"  Classes: {processing_params['n_classes']}")
            print(f"  Output: {pair_dir}")
            
            return config
            
        except Exception as e:
            # Log full error stack trace to logger
            error_msg = f"Failed to process pair {db_id1}-{db_id2}: {str(e)}"
            logging.error(f"{error_msg}\n{traceback.format_exc()}")
            
            error_config = {
                'pair_id': f"{db_id1:05d}_{db_id2:05d}",
                'db_id1': db_id1,
                'db_id2': db_id2,
                'similarity': pair_metadata['similarity'],
                'error': str(e),
                'status': 'failed'
            }
            
            print(f"Failed to process pair {db_id1}-{db_id2}: {e}")
            print(f"  No output directory created for failed pair")
            
            return error_config


def main():
    """Main function for data preprocessing."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Preprocess database pairs for FL validation")
    parser.add_argument("--input", type=str, required=True,
                       help="Input JSON file with sampled pairs")
    parser.add_argument("--output-dir", type=str, default="data/auto",
                       help="Output directory for processed data (default: data/auto)")
    parser.add_argument("--test-size", type=float, default=0.2,
                       help="Test set size (default: 0.2)")
    parser.add_argument("--random-state", type=int, default=42,
                       help="Random state (default: 42)")
    parser.add_argument("--min-label-variance", type=float, default=0.01,
                       help="Minimum label variance (default: 0.01)")
    parser.add_argument("--max-missing-ratio", type=float, default=0.5,
                       help="Maximum missing ratio (default: 0.5)")
    parser.add_argument("--max-rows", type=int, default=1000000,
                       help="Maximum number of rows in joined table (default: 1,000,000)")
    parser.add_argument("--retry", action="store_true",
                       help="Retry only failed pairs from error log")
    parser.add_argument("--retry-log", type=str, default=None,
                       help="Path to error log for retry mode (default: output-dir/preprocessing_errors.log)")
    
    args = parser.parse_args()
    
    # Set up logging to file
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / 'preprocessing_errors.log'
    
    # Handle retry mode
    if args.retry:
        # Determine error log path
        retry_log_path = args.retry_log if args.retry_log else log_file
        
        # Create preprocessor to access retry methods
        temp_preprocessor = AutomatedDataPreprocessor()
        
        # Parse failed pairs from error log
        failed_pair_ids = temp_preprocessor.parse_failed_pairs_from_log(retry_log_path)
        
        if not failed_pair_ids:
            print(f"No failed pairs found in error log: {retry_log_path}")
            return 0
        
        print(f"Found {len(failed_pair_ids)} failed pairs to retry: {failed_pair_ids}")
        
        # Rotate error logs
        temp_preprocessor.rotate_error_logs(args.output_dir)
        
        # Load original pairs and filter to only failed ones
        with open(args.input, 'r') as f:
            sampling_data = json.load(f)
        
        all_pairs = sampling_data['pairs']
        pairs = []
        
        for pair in all_pairs:
            pair_id = f"{pair['db_id1']:05d}-{pair['db_id2']:05d}"
            if pair_id in failed_pair_ids:
                pairs.append(pair)
        
        print(f"Filtered to {len(pairs)} pairs for retry (out of {len(all_pairs)} total pairs)")
        
        if len(pairs) != len(failed_pair_ids):
            print(f"Warning: Could only find {len(pairs)} pairs to retry out of {len(failed_pair_ids)} failed pairs")
    else:
        # Normal mode: process all pairs
        with open(args.input, 'r') as f:
            sampling_data = json.load(f)
        
        pairs = sampling_data['pairs']
        print(f"Loaded {len(pairs)} pairs for preprocessing")
    
    logging.basicConfig(
        level=logging.ERROR,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w'),  # Overwrite each run
        ]
    )
    
    # Create preprocessor
    preprocessor = AutomatedDataPreprocessor(
        test_size=args.test_size,
        random_state=args.random_state,
        min_label_variance=args.min_label_variance,
        max_missing_ratio=args.max_missing_ratio,
        max_rows=args.max_rows
    )
    
    # Process all pairs
    results = []
    successful = 0
    failed = 0
    task_type_counts = {'classification': 0, 'regression': 0}
    
    for i, pair in enumerate(pairs):
        print(f"\n{'='*60}")
        print(f"Processing pair {i+1}/{len(pairs)}")
        
        config = preprocessor.process_pair(pair, args.output_dir)
        results.append(config)
        
        if config['status'] == 'failed':
            failed += 1
        else:
            successful += 1
            task_type = config['task_type']
            task_type_counts[task_type] += 1
    
    # Generate detailed statistics
    classification_pairs = [r for r in results if r.get('task_type') == 'classification']
    regression_pairs = [r for r in results if r.get('task_type') == 'regression']
    
    # Classification statistics
    class_counts = {}
    for pair in classification_pairs:
        n_classes = pair.get('processing_params', {}).get('n_classes', 0)
        if n_classes > 0:
            if n_classes not in class_counts:
                class_counts[n_classes] = 0
            class_counts[n_classes] += 1
    
    # Regression statistics
    reg_stats = []
    for pair in regression_pairs:
        variance = pair.get('label_metadata', {}).get('variance', 0)
        n_unique = pair.get('label_metadata', {}).get('n_unique', 0)
        if variance > 0:
            reg_stats.append({'variance': variance, 'n_unique': n_unique})
    
    # Save overall results
    summary = {
        'preprocessing_config': {
            'test_size': args.test_size,
            'random_state': args.random_state,
            'min_label_variance': args.min_label_variance,
            'max_missing_ratio': args.max_missing_ratio,
            'max_rows': args.max_rows
        },
        'summary_stats': {
            'total_pairs': len(pairs),
            'successful': successful,
            'failed': failed,
            'success_rate': (successful / len(pairs)) * 100 if pairs else 0,
            'task_type_distribution': task_type_counts,
            'classification_stats': {
                'total': task_type_counts['classification'],
                'class_distribution': class_counts
            },
            'regression_stats': {
                'total': task_type_counts['regression'],
                'avg_variance': sum(s['variance'] for s in reg_stats) / len(reg_stats) if reg_stats else 0,
                'avg_unique_values': sum(s['n_unique'] for s in reg_stats) / len(reg_stats) if reg_stats else 0
            }
        },
        'results': results
    }
    
    # Save main summary (merge with existing if retry mode)
    summary_file = Path(args.output_dir) / 'preprocessing_summary.json'
    
    if args.retry and summary_file.exists():
        # Load existing summary and merge retry results
        with open(summary_file, 'r') as f:
            existing_summary = json.load(f)
        
        # Create a lookup for retry results by pair_id
        retry_results_lookup = {r['pair_id']: r for r in results}
        
        # Update existing results with retry results
        updated_results = []
        for existing_result in existing_summary.get('results', []):
            pair_id = existing_result['pair_id']
            if pair_id in retry_results_lookup:
                # Replace with retry result
                updated_results.append(retry_results_lookup[pair_id])
                print(f"Updated result for pair {pair_id}: {existing_result.get('status', 'unknown')} -> {retry_results_lookup[pair_id]['status']}")
            else:
                # Keep existing result
                updated_results.append(existing_result)
        
        # Recalculate summary statistics
        successful_total = sum(1 for r in updated_results if r['status'] != 'failed')
        failed_total = sum(1 for r in updated_results if r['status'] == 'failed')
        task_type_counts_total = {'classification': 0, 'regression': 0}
        
        for r in updated_results:
            if r['status'] != 'failed':
                task_type = r.get('task_type', '')
                if task_type in task_type_counts_total:
                    task_type_counts_total[task_type] += 1
        
        # Update summary with merged results
        # --- FIX: Use 'summary_stats' instead of 'summary' ---
        existing_summary['summary_stats']['total_pairs'] = len(updated_results)
        existing_summary['summary_stats']['successful'] = successful_total
        existing_summary['summary_stats']['failed'] = failed_total
        existing_summary['summary_stats']['success_rate'] = successful_total / len(updated_results) * 100 if updated_results else 0
        existing_summary['summary_stats']['task_type_distribution'] = task_type_counts_total
        # -----------------------------------------------------------
        existing_summary['results'] = updated_results
        
        # Add retry information
        existing_summary['retry_info'] = {
            'retry_performed': True,
            'retry_pairs_count': len(pairs),
            'retry_successful': successful,
            'retry_failed': failed
        }
        
        final_summary = existing_summary
        print(f"\nMerged retry results with existing summary:")
        print(f"  Total pairs: {len(updated_results)}")
        print(f"  Retry updated: {len(retry_results_lookup)} pairs")
        print(f"  New success rate: {successful_total/len(updated_results)*100:.1f}%")
    else:
        # Normal mode or first run
        final_summary = summary
    
    with open(summary_file, 'w') as f:
        json.dump(final_summary, f, indent=2, default=str)
    
    # Create detailed log file
    log_file = Path(args.output_dir) / 'preprocessing_log.txt'
    with open(log_file, 'w') as f:
        f.write("FEDERATED LEARNING DATA PREPROCESSING SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"Total database pairs processed: {len(pairs)}\n")
        f.write(f"Successful: {successful} ({successful/len(pairs)*100:.1f}%)\n")
        f.write(f"Failed: {failed} ({failed/len(pairs)*100:.1f}%)\n\n")
        
        f.write("TASK TYPE DISTRIBUTION:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Classification tasks: {task_type_counts['classification']}\n")
        f.write(f"Regression tasks: {task_type_counts['regression']}\n\n")
        
        if task_type_counts['classification'] > 0:
            f.write("CLASSIFICATION DETAILS:\n")
            f.write("-" * 30 + "\n")
            for n_classes, count in sorted(class_counts.items()):
                f.write(f"  {n_classes} classes: {count} pairs\n")
            f.write("\n")
        
        if task_type_counts['regression'] > 0:
            f.write("REGRESSION DETAILS:\n")
            f.write("-" * 30 + "\n")
            avg_var = sum(s['variance'] for s in reg_stats) / len(reg_stats) if reg_stats else 0
            avg_unique = sum(s['n_unique'] for s in reg_stats) / len(reg_stats) if reg_stats else 0
            f.write(f"  Average label variance: {avg_var:.6f}\n")
            f.write(f"  Average unique values: {avg_unique:.1f}\n\n")
        
        if failed > 0:
            f.write("FAILED PAIRS:\n")
            f.write("-" * 30 + "\n")
            for result in results:
                if result['status'] == 'failed':
                    f.write(f"  {result['pair_id']}: {result['error']}\n")
            f.write("\n")
        
        f.write("SUCCESSFUL PAIRS BY TYPE:\n")
        f.write("-" * 30 + "\n")
        for result in results:
            if result['status'] == 'success':
                task_type = result['task_type']
                pair_id = result['pair_id']
                if task_type == 'classification':
                    n_classes = result['processing_params']['n_classes']
                    f.write(f"  {pair_id}: {task_type} ({n_classes} classes)\n")
                else:
                    variance = result['label_metadata']['variance']
                    f.write(f"  {pair_id}: {task_type} (variance: {variance:.4f})\n")
    
    print(f"\n{'='*60}")
    print(f"PREPROCESSING SUMMARY:")
    print(f"{'='*60}")
    print(f"Total pairs: {len(pairs)}")
    print(f"Successful: {successful} ({successful/len(pairs)*100:.1f}%)")
    print(f"Failed: {failed} ({failed/len(pairs)*100:.1f}%)")
    print(f"")
    print(f"Task type distribution:")
    print(f"  Classification: {task_type_counts['classification']}")
    print(f"  Regression: {task_type_counts['regression']}")
    print(f"")
    print(f"Files saved:")
    print(f"  Summary: {summary_file}")
    print(f"  Detailed log: {log_file}")
    if failed > 0:
        print(f"  Error log: {output_dir / 'preprocessing_errors.log'}")
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    exit(main())