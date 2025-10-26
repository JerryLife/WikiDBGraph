"""
Automated Database Pair Sampling for Federated Learning Validation

This module samples database pairs within a specified similarity range for FL experiments.
"""

import pandas as pd
import numpy as np
import os
import json
import sqlite3
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import random
from collections import defaultdict

from analysis.NodeProperties import Database, Table, Column


class DatabasePairSampler:
    """
    Sample database pairs for automated federated learning validation.
    
    This class filters database pairs by similarity threshold and meaningful data requirements,
    then samples a specified number of pairs for FL experiments.
    """
    
    def __init__(self, 
                 min_similarity: float = 0.98,
                 max_similarity: float = 1.0,
                 min_table_rows: int = 100,
                 sample_size: int = 100,
                 seed: int = 42):
        """
        Initialize the pair sampler.
        
        Args:
            min_similarity: Minimum similarity threshold (default: 0.98)
            max_similarity: Maximum similarity threshold (default: 1.0)
            min_table_rows: Minimum number of rows in largest table (default: 100)
            sample_size: Number of pairs to sample (default: 100)
            seed: Random seed for reproducibility (default: 42)
        """
        self.min_similarity = min_similarity
        self.max_similarity = max_similarity
        self.min_table_rows = min_table_rows
        self.sample_size = sample_size
        self.seed = seed
        
        # Set random seed
        random.seed(seed)
        np.random.seed(seed)
        
        self.edges_file = "data/graph/filtered_edges_threshold_0.94.csv"
        self.unzip_dir = "data/unzip"
        
    def load_edge_properties(self) -> pd.DataFrame:
        """Load edge properties from CSV file."""
        if not os.path.exists(self.edges_file):
            raise FileNotFoundError(f"Edge properties file not found: {self.edges_file}")
        
        print(f"Loading edge properties from {self.edges_file}")
        return pd.read_csv(self.edges_file)
    
    def get_database_folder_mapping(self) -> Dict[int, str]:
        """Create mapping from database ID to folder name."""
        db_id_to_folder = {}
        
        if not os.path.exists(self.unzip_dir):
            raise FileNotFoundError(f"Unzip directory not found: {self.unzip_dir}")
        
        for folder in os.listdir(self.unzip_dir):
            folder_path = os.path.join(self.unzip_dir, folder)
            if os.path.isdir(folder_path):
                try:
                    id_str, _ = folder.split(" ", 1)
                    numeric_id = int(id_str)
                    db_id_to_folder[numeric_id] = folder
                except (ValueError, IndexError):
                    continue
        
        print(f"Found {len(db_id_to_folder)} database folders")
        return db_id_to_folder
    
    def get_database_row_counts(self, folder_path: str) -> int:
        """Get the maximum number of rows across all tables in a database."""
        tables_dir = os.path.join(folder_path, "tables")
        if not os.path.exists(tables_dir):
            return 0
        
        max_rows = 0
        for table_file in os.listdir(tables_dir):
            if table_file.endswith('.csv'):
                try:
                    file_path = os.path.join(tables_dir, table_file)
                    df = pd.read_csv(file_path)
                    max_rows = max(max_rows, len(df))
                except Exception as e:
                    print(f"Warning: Could not read {table_file}: {e}")
                    continue
        
        return max_rows
    
    def get_database_schema_info(self, db_id: int, folder_path: str) -> Optional[Dict]:
        """Extract database schema information."""
        schema_path = os.path.join(folder_path, "schema.json")
        
        if not os.path.exists(schema_path):
            return None
        
        try:
            with open(schema_path, "r") as f:
                schema = json.load(f)
            
            db = Database(db_id)
            db.load_from_schema(schema)
            
            # Extract basic info
            tables_info = []
            all_columns = set()
            
            for table in db.tables:
                table_info = {
                    "table_name": table.table_name,
                    "num_columns": len(table.columns),
                    "columns": [col.column_name for col in table.columns]
                }
                tables_info.append(table_info)
                
                # Collect normalized column names
                for col in table.columns:
                    normalized_col = ''.join(c for c in col.column_name.lower() if c.isalnum())
                    all_columns.add(normalized_col)
            
            max_rows = self.get_database_row_counts(folder_path)
            
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
            print(f"Error processing schema for database {db_id}: {e}")
            return None
    
    def filter_pairs_by_similarity(self, edges_df: pd.DataFrame) -> pd.DataFrame:
        """Filter pairs by similarity range."""
        filtered = edges_df[
            (edges_df['similarity'] >= self.min_similarity) & 
            (edges_df['similarity'] <= self.max_similarity)
        ]
        
        print(f"Found {len(filtered)} pairs with similarity in range [{self.min_similarity}, {self.max_similarity}]")
        return filtered
    
    def validate_pair_data_quality(self, db_id1: int, db_id2: int, 
                                 folder1: str, folder2: str) -> Tuple[bool, Dict]:
        """
        Validate that a database pair has meaningful data for FL.
        
        Returns:
            Tuple of (is_valid, metadata_dict)
        """
        folder_path1 = os.path.join(self.unzip_dir, folder1)
        folder_path2 = os.path.join(self.unzip_dir, folder2)
        
        # Get schema information
        schema1 = self.get_database_schema_info(db_id1, folder_path1)
        schema2 = self.get_database_schema_info(db_id2, folder_path2)
        
        if not schema1 or not schema2:
            return False, {"error": "Could not load schema information"}
        
        # Check minimum row requirement
        if schema1["max_rows"] < self.min_table_rows or schema2["max_rows"] < self.min_table_rows:
            return False, {
                "error": "Insufficient rows",
                "rows1": schema1["max_rows"],
                "rows2": schema2["max_rows"],
                "min_required": self.min_table_rows
            }
        
        # Find common columns
        columns1 = set(schema1["all_columns"])
        columns2 = set(schema2["all_columns"])
        common_columns = list(columns1 & columns2)
        
        if len(common_columns) < 2:  # Need at least 2 common columns (features + potential target)
            return False, {
                "error": "Too few common columns",
                "common_columns": len(common_columns),
                "min_required": 2
            }
        
        metadata = {
            "db_id1": db_id1,
            "db_id2": db_id2,
            "database1": schema1,
            "database2": schema2,
            "common_columns": common_columns,
            "num_common_columns": len(common_columns)
        }
        
        return True, metadata
    
    def sample_pairs(self) -> List[Dict]:
        """
        Sample database pairs for FL validation using efficient streaming top-k approach.
        
        Returns:
            List of sampled pair metadata dictionaries
        """
        print("Starting database pair sampling...")
        
        # Load edge properties
        edges_df = self.load_edge_properties()
        
        # Filter by similarity
        filtered_pairs = self.filter_pairs_by_similarity(edges_df)
        
        if len(filtered_pairs) == 0:
            raise ValueError(f"No pairs found with similarity in range [{self.min_similarity}, {self.max_similarity}]")
        
        # Get database folder mapping
        db_id_to_folder = self.get_database_folder_mapping()
        
        # Sort by similarity (descending) for streaming top-k
        filtered_pairs = filtered_pairs.sort_values('similarity', ascending=False)
        
        # Streaming top-k with tie buffer
        sampled_pairs = []
        tie_buffer = []
        current_similarity = -1.0
        processed = 0
        
        print(f"Processing pairs in similarity order for top-{self.sample_size} selection...")
        
        for _, row in filtered_pairs.iterrows():
            # Early exit if we have enough samples
            if len(sampled_pairs) >= self.sample_size:
                break
                
            processed += 1
            if processed % 100 == 0:
                print(f"Processed {processed} pairs, collected {len(sampled_pairs)} samples (buffer: {len(tie_buffer)})")
            
            db_id1 = int(row['src'])
            db_id2 = int(row['tgt'])
            similarity = row['similarity']
            rounded_similarity = round(similarity, 4)
            
            # Check if folders exist
            folder1 = db_id_to_folder.get(db_id1)
            folder2 = db_id_to_folder.get(db_id2)
            if not folder1 or not folder2:
                continue
            
            # Validate data quality
            is_valid, metadata = self.validate_pair_data_quality(db_id1, db_id2, folder1, folder2)
            if not is_valid:
                continue
            
            # Process tie buffer when similarity drops
            if rounded_similarity < current_similarity and tie_buffer:
                remaining_slots = self.sample_size - len(sampled_pairs)
                
                if len(tie_buffer) <= remaining_slots:
                    # All pairs in buffer fit
                    sampled_pairs.extend(tie_buffer)
                    print(f"Added {len(tie_buffer)} pairs at similarity {current_similarity:.4f}")
                else:
                    # Buffer won't fit, randomly sample and finish
                    selected_pairs = random.sample(tie_buffer, remaining_slots)
                    sampled_pairs.extend(selected_pairs)
                    print(f"Randomly sampled {remaining_slots} from {len(tie_buffer)} pairs at similarity {current_similarity:.4f}")
                    break  # We're done - exit main loop
                
                tie_buffer = []  # Clear buffer for new similarity level
            
            # Add current valid pair to buffer
            current_similarity = rounded_similarity
            metadata['similarity'] = similarity
            metadata['folder1'] = folder1
            metadata['folder2'] = folder2
            tie_buffer.append(metadata)
        
        # Handle any remaining pairs in the buffer after loop finishes
        if tie_buffer and len(sampled_pairs) < self.sample_size:
            remaining_slots = self.sample_size - len(sampled_pairs)
            
            if len(tie_buffer) <= remaining_slots:
                sampled_pairs.extend(tie_buffer)
                print(f"Added final {len(tie_buffer)} pairs at similarity {current_similarity:.4f}")
            else:
                selected_pairs = random.sample(tie_buffer, remaining_slots)
                sampled_pairs.extend(selected_pairs)
                print(f"Randomly sampled final {remaining_slots} from {len(tie_buffer)} pairs at similarity {current_similarity:.4f}")
        
        print(f"Efficient top-{self.sample_size} sampling completed: {len(sampled_pairs)} pairs selected from {processed} processed")
        
        return sampled_pairs
    
    def save_sampled_pairs(self, pairs: List[Dict], output_file: str) -> None:
        """Save sampled pairs to JSON file."""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Add sampling metadata
        sampling_info = {
            "sampling_params": {
                "min_similarity": self.min_similarity,
                "max_similarity": self.max_similarity,
                "min_table_rows": self.min_table_rows,
                "sample_size": self.sample_size,
                "seed": self.seed
            },
            "total_sampled": len(pairs),
            "pairs": pairs
        }
        
        with open(output_path, 'w') as f:
            json.dump(sampling_info, f, indent=2)
        
        print(f"Saved {len(pairs)} sampled pairs to {output_file}")


def main():
    """Main function for running pair sampling."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Sample database pairs for FL validation")
    parser.add_argument("--min-similarity", type=float, default=0.98,
                       help="Minimum similarity threshold (default: 0.98)")
    parser.add_argument("--max-similarity", type=float, default=1.0,
                       help="Maximum similarity threshold (default: 1.0)")
    parser.add_argument("--min-rows", type=int, default=100,
                       help="Minimum table rows requirement (default: 100)")
    parser.add_argument("--sample-size", type=int, default=200,
                       help="Number of pairs to sample (default: 100)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed (default: 42)")
    parser.add_argument("--output", type=str, default="out/autorun/sampled_pairs.json",
                       help="Output file path (default: out/autorun/sampled_pairs.json)")
    
    args = parser.parse_args()
    
    # Create sampler and run
    sampler = DatabasePairSampler(
        min_similarity=args.min_similarity,
        max_similarity=args.max_similarity,
        min_table_rows=args.min_rows,
        sample_size=args.sample_size,
        seed=args.seed
    )
    
    try:
        pairs = sampler.sample_pairs()
        sampler.save_sampled_pairs(pairs, args.output)
        
        print(f"\nSampling Summary:")
        print(f"- Similarity range: [{args.min_similarity}, {args.max_similarity}]")
        print(f"- Minimum rows: {args.min_rows}")
        print(f"- Requested samples: {args.sample_size}")
        print(f"- Actually sampled: {len(pairs)}")
        print(f"- Output file: {args.output}")
        
    except Exception as e:
        print(f"Error during sampling: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())