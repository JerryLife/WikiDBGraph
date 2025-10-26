"""
Calculate statistical properties for database nodes.

This module computes statistical properties for databases including:
- Data volume (file size)
- Column cardinality (distinct values per column)
- Column sparsity (NULL ratio per column)
- Column entropy (Shannon entropy per column)

Uses multi-threading for efficient parallel processing in nogil environment.
"""

import os
import sqlite3
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class NodeStatistical:
    """
    Class for computing statistical properties of database nodes.
    """
    
    def __init__(self, data_dir: str = "data/unzip", output_dir: str = "data/graph", num_threads: int = 32, force_recompute: bool = False):
        """
        Initialize the statistical property calculator.
        
        Args:
            data_dir: Directory containing database folders
            output_dir: Directory to save output CSV files
            num_threads: Number of threads for parallel processing
            force_recompute: If True, recompute all statistics even if output files exist
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.num_threads = num_threads
        self.force_recompute = force_recompute
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get all database folders
        self.db_folders = sorted([
            f for f in self.data_dir.iterdir() 
            if f.is_dir() and f.name.split()[0].isdigit()
        ])
        logger.info(f"Found {len(self.db_folders)} database folders")
    
    def _get_db_id(self, db_folder: Path) -> str:
        """Extract database ID from folder name (normalized to 5-digit zero-padded)."""
        db_id = db_folder.name.split()[0]
        # Normalize to 5-digit zero-padded format
        return f"{int(db_id):05d}"
    
    def _get_database_path(self, db_folder: Path) -> Path:
        """Get the path to database.db file."""
        return db_folder / "database.db"
    
    def compute_data_volume(self) -> pd.DataFrame:
        """
        Compute data volume (file size in bytes) for all databases.
        
        Returns:
            DataFrame with columns: db_id, volume_bytes
        """
        logger.info("Computing data volume...")
        output_path = self.output_dir / "data_volume.csv"
        
        # Check if file exists and we're not forcing recompute
        if output_path.exists() and not self.force_recompute:
            logger.info(f"File {output_path} already exists. Use --force to recompute.")
            return pd.read_csv(output_path)
        
        # Process all databases
        folders_to_process = self.db_folders
        logger.info(f"Processing {len(folders_to_process)} databases")
        
        def get_volume(db_folder: Path) -> Optional[Tuple[str, int]]:
            """Get volume for a single database."""
            try:
                db_id = self._get_db_id(db_folder)
                db_path = self._get_database_path(db_folder)
                
                if not db_path.exists():
                    logger.warning(f"Database file not found: {db_path}")
                    return None
                
                # Use os.path.getsize for file size
                volume_bytes = os.path.getsize(db_path)
                return (db_id, volume_bytes)
            except Exception as e:
                logger.error(f"Error processing {db_folder.name}: {e}")
                return None
        
        # Process databases in parallel
        results = []
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            futures = [executor.submit(get_volume, folder) for folder in folders_to_process]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Data volume"):
                result = future.result()
                if result:
                    results.append(result)
        
        # Create DataFrame
        df = pd.DataFrame(results, columns=['db_id', 'volume_bytes'])
        
        # Sort by db_id
        df = df.sort_values('db_id').reset_index(drop=True)
        
        # Save to CSV (overwrite completely)
        df.to_csv(output_path, index=False)
        logger.info(f"Saved data volume to {output_path} ({len(df)} databases processed)")
        
        return df
    
    def compute_column_sparsity(self) -> pd.DataFrame:
        """
        Compute column sparsity (NULL ratio) for all databases.
        
        Returns:
            DataFrame with columns: db_id, table_name, column_name, sparsity
        """
        logger.info("Computing column sparsity...")
        output_path = self.output_dir / "column_sparsity.csv"
        
        # Check if file exists and we're not forcing recompute
        if output_path.exists() and not self.force_recompute:
            logger.info(f"File {output_path} already exists. Use --force to recompute.")
            return pd.read_csv(output_path)
        
        # Process all databases
        folders_to_process = self.db_folders
        logger.info(f"Processing {len(folders_to_process)} databases")
        
        def get_sparsity(db_folder: Path) -> List[Tuple[str, str, str, float]]:
            """Get sparsity for all columns in a database."""
            results = []
            db_id = self._get_db_id(db_folder)
            db_path = self._get_database_path(db_folder)
            
            if not db_path.exists():
                return results
            
            try:
                # Connect to database
                source_conn = sqlite3.connect(str(db_path))
                
                # Copy to in-memory database for faster queries
                mem_conn = sqlite3.connect(":memory:")
                source_conn.backup(mem_conn)
                source_conn.close()
                
                cursor = mem_conn.cursor()
                
                # Get all tables
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                tables = [row[0] for row in cursor.fetchall()]
                
                for table_name in tables:
                    try:
                        # Get all columns for this table
                        cursor.execute(f'PRAGMA table_info("{table_name}");')
                        columns = [row[1] for row in cursor.fetchall()]
                        
                        # Get total row count
                        cursor.execute(f'SELECT COUNT(*) FROM "{table_name}";')
                        total_count = cursor.fetchone()[0]
                        
                        if total_count == 0:
                            # Empty table, all columns have 0 sparsity
                            for column_name in columns:
                                results.append((db_id, table_name, column_name, 0.0))
                            continue
                        
                        # Calculate sparsity for each column
                        for column_name in columns:
                            try:
                                # Count non-NULL values
                                cursor.execute(f'SELECT COUNT("{column_name}") FROM "{table_name}";')
                                non_null_count = cursor.fetchone()[0]
                                
                                # Calculate sparsity (ratio of NULL values)
                                null_count = total_count - non_null_count
                                sparsity = null_count / total_count
                                
                                results.append((db_id, table_name, column_name, sparsity))
                            except sqlite3.Error as e:
                                logger.warning(f"Error calculating sparsity for {db_id}.{table_name}.{column_name}: {e}")
                    
                    except sqlite3.Error as e:
                        logger.warning(f"Error processing table {db_id}.{table_name}: {e}")
                
                mem_conn.close()
                
            except Exception as e:
                logger.error(f"Error processing database {db_folder.name}: {e}")
            
            return results
        
        # Process databases in parallel
        all_results = []
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            futures = [executor.submit(get_sparsity, folder) for folder in folders_to_process]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Column sparsity"):
                results = future.result()
                all_results.extend(results)
        
        # Create DataFrame
        df = pd.DataFrame(all_results, columns=['db_id', 'table_name', 'column_name', 'sparsity'])
        
        # Save to CSV (overwrite completely)
        df.to_csv(output_path, index=False)
        logger.info(f"Saved column sparsity to {output_path} ({len(df)} columns processed)")
        
        return df
    
    def compute_column_cardinality(self) -> pd.DataFrame:
        """
        Compute column cardinality (distinct values) for all databases.
        
        Returns:
            DataFrame with columns: db_id, table_name, column_name, n_distinct
        """
        logger.info("Computing column cardinality...")
        output_path = self.output_dir / "column_cardinality.csv"
        
        # Check if file exists and we're not forcing recompute
        if output_path.exists() and not self.force_recompute:
            logger.info(f"File {output_path} already exists. Use --force to recompute.")
            return pd.read_csv(output_path)
        
        # Process all databases
        folders_to_process = self.db_folders
        logger.info(f"Processing {len(folders_to_process)} databases")
        
        def get_cardinality(db_folder: Path) -> List[Tuple[str, str, str, int]]:
            """Get cardinality for all columns in a database."""
            results = []
            db_id = self._get_db_id(db_folder)
            db_path = self._get_database_path(db_folder)
            
            if not db_path.exists():
                return results
            
            try:
                # Connect to database
                source_conn = sqlite3.connect(str(db_path))
                
                # Copy to in-memory database for faster queries
                mem_conn = sqlite3.connect(":memory:")
                source_conn.backup(mem_conn)
                source_conn.close()
                
                cursor = mem_conn.cursor()
                
                # Get all tables
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                tables = [row[0] for row in cursor.fetchall()]
                
                for table_name in tables:
                    try:
                        # Get all columns for this table
                        cursor.execute(f'PRAGMA table_info("{table_name}");')
                        columns = [row[1] for row in cursor.fetchall()]
                        
                        # Calculate cardinality for each column
                        for column_name in columns:
                            try:
                                # Count distinct non-NULL values
                                cursor.execute(
                                    f'SELECT COUNT(DISTINCT "{column_name}") FROM "{table_name}" WHERE "{column_name}" IS NOT NULL;'
                                )
                                n_distinct = cursor.fetchone()[0]
                                
                                results.append((db_id, table_name, column_name, n_distinct))
                            except sqlite3.Error as e:
                                logger.warning(f"Error calculating cardinality for {db_id}.{table_name}.{column_name}: {e}")
                    
                    except sqlite3.Error as e:
                        logger.warning(f"Error processing table {db_id}.{table_name}: {e}")
                
                mem_conn.close()
                
            except Exception as e:
                logger.error(f"Error processing database {db_folder.name}: {e}")
            
            return results
        
        # Process databases in parallel
        all_results = []
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            futures = [executor.submit(get_cardinality, folder) for folder in folders_to_process]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Column cardinality"):
                results = future.result()
                all_results.extend(results)
        
        # Create DataFrame
        df = pd.DataFrame(all_results, columns=['db_id', 'table_name', 'column_name', 'n_distinct'])
        
        # Save to CSV (overwrite completely)
        df.to_csv(output_path, index=False)
        logger.info(f"Saved column cardinality to {output_path} ({len(df)} columns processed)")
        
        return df
    
    def compute_column_entropy(self) -> pd.DataFrame:
        """
        Compute column entropy (Shannon entropy) for all databases.
        
        Returns:
            DataFrame with columns: db_id, table_name, column_name, entropy
        """
        logger.info("Computing column entropy...")
        output_path = self.output_dir / "column_entropy.csv"
        
        # Check if file exists and we're not forcing recompute
        if output_path.exists() and not self.force_recompute:
            logger.info(f"File {output_path} already exists. Use --force to recompute.")
            return pd.read_csv(output_path)
        
        # Process all databases
        folders_to_process = self.db_folders
        logger.info(f"Processing {len(folders_to_process)} databases")
        
        def calculate_entropy(value_counts: np.ndarray) -> float:
            """
            Calculate Shannon entropy from value counts.
            
            Args:
                value_counts: Array of counts for each distinct value
                
            Returns:
                Shannon entropy in bits
            """
            if len(value_counts) == 0 or value_counts.sum() == 0:
                return 0.0
            
            # Calculate probabilities
            probabilities = value_counts / value_counts.sum()
            
            # Remove zero probabilities to avoid log(0)
            probabilities = probabilities[probabilities > 0]
            
            # Calculate entropy: H = -Σ(p * log2(p))
            entropy = -np.sum(probabilities * np.log2(probabilities))
            
            return float(entropy)
        
        def get_entropy(db_folder: Path) -> List[Tuple[str, str, str, float]]:
            """Get entropy for all columns in a database."""
            results = []
            db_id = self._get_db_id(db_folder)
            db_path = self._get_database_path(db_folder)
            
            if not db_path.exists():
                return results
            
            try:
                # Connect to database
                source_conn = sqlite3.connect(str(db_path))
                
                # Copy to in-memory database for faster queries
                mem_conn = sqlite3.connect(":memory:")
                source_conn.backup(mem_conn)
                source_conn.close()
                
                cursor = mem_conn.cursor()
                
                # Get all tables
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                tables = [row[0] for row in cursor.fetchall()]
                
                for table_name in tables:
                    try:
                        # Get all columns for this table
                        cursor.execute(f'PRAGMA table_info("{table_name}");')
                        columns = [row[1] for row in cursor.fetchall()]
                        
                        # Calculate entropy for each column
                        for column_name in columns:
                            try:
                                # Get value distribution (counts for each distinct value)
                                cursor.execute(
                                    f'SELECT COUNT(*) as cnt FROM "{table_name}" '
                                    f'WHERE "{column_name}" IS NOT NULL '
                                    f'GROUP BY "{column_name}";'
                                )
                                counts = cursor.fetchall()
                                
                                if counts:
                                    # Extract counts as numpy array
                                    value_counts = np.array([row[0] for row in counts])
                                    entropy = calculate_entropy(value_counts)
                                else:
                                    entropy = 0.0
                                
                                results.append((db_id, table_name, column_name, entropy))
                            except sqlite3.Error as e:
                                logger.warning(f"Error calculating entropy for {db_id}.{table_name}.{column_name}: {e}")
                    
                    except sqlite3.Error as e:
                        logger.warning(f"Error processing table {db_id}.{table_name}: {e}")
                
                mem_conn.close()
                
            except Exception as e:
                logger.error(f"Error processing database {db_folder.name}: {e}")
            
            return results
        
        # Process databases in parallel
        all_results = []
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            futures = [executor.submit(get_entropy, folder) for folder in folders_to_process]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Column entropy"):
                results = future.result()
                all_results.extend(results)
        
        # Create DataFrame
        df = pd.DataFrame(all_results, columns=['db_id', 'table_name', 'column_name', 'entropy'])
        
        # Save to CSV (overwrite completely)
        df.to_csv(output_path, index=False)
        logger.info(f"Saved column entropy to {output_path} ({len(df)} columns processed)")
        
        return df
    
    def run_all_steps(self) -> Dict[str, pd.DataFrame]:
        """
        Run all statistical property calculations.
        
        Returns:
            Dictionary mapping step name to resulting DataFrame
        """
        results = {}
        
        logger.info("=" * 80)
        logger.info("Running all statistical property calculations")
        logger.info("=" * 80)
        
        results['volume'] = self.compute_data_volume()
        results['sparsity'] = self.compute_column_sparsity()
        results['cardinality'] = self.compute_column_cardinality()
        results['entropy'] = self.compute_column_entropy()
        
        logger.info("=" * 80)
        logger.info("All steps completed!")
        logger.info("=" * 80)
        
        return results
    
    def run_specific_steps(self, steps: List[str]) -> Dict[str, pd.DataFrame]:
        """
        Run specific statistical property calculations.
        
        Args:
            steps: List of step names to run ('volume', 'sparsity', 'cardinality', 'entropy')
            
        Returns:
            Dictionary mapping step name to resulting DataFrame
        """
        results = {}
        
        step_mapping = {
            'volume': self.compute_data_volume,
            'sparsity': self.compute_column_sparsity,
            'cardinality': self.compute_column_cardinality,
            'entropy': self.compute_column_entropy
        }
        
        logger.info("=" * 80)
        logger.info(f"Running steps: {', '.join(steps)}")
        logger.info("=" * 80)
        
        for step in steps:
            if step not in step_mapping:
                logger.error(f"Unknown step: {step}. Valid steps: {list(step_mapping.keys())}")
                continue
            
            results[step] = step_mapping[step]()
        
        logger.info("=" * 80)
        logger.info("Selected steps completed!")
        logger.info("=" * 80)
        
        return results


def main():
    """Main function for CLI interface."""
    parser = argparse.ArgumentParser(
        description="Calculate statistical properties for database nodes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all steps
  python src/analysis/NodeStatistical.py --steps all
  
  # Run specific steps
  python src/analysis/NodeStatistical.py --steps volume,sparsity
  
  # Use more threads
  python src/analysis/NodeStatistical.py --steps all --threads 64
  
  # Specify custom directories
  python src/analysis/NodeStatistical.py --data-dir data/unzip --output-dir data/graph
        """
    )
    
    parser.add_argument(
        '--steps',
        type=str,
        default='all',
        help='Comma-separated list of steps to run: all, volume, sparsity, cardinality, entropy (default: all)'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/unzip',
        help='Directory containing database folders (default: data/unzip)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/graph',
        help='Directory to save output CSV files (default: data/graph)'
    )
    parser.add_argument(
        '--threads',
        type=int,
        default=32,
        help='Number of threads for parallel processing (default: 1)'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force recomputation of all statistics, even if output files already exist'
    )
    
    args = parser.parse_args()
    
    # Create calculator instance
    calculator = NodeStatistical(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        num_threads=args.threads,
        force_recompute=args.force
    )
    
    # Parse steps
    if args.steps.lower() == 'all':
        calculator.run_all_steps()
    else:
        steps = [s.strip() for s in args.steps.split(',')]
        calculator.run_specific_steps(steps)


if __name__ == "__main__":
    main()

