"""
Table Extractor for GitTables

Extracts tables from GitTables parquet files and filters them based on
minimum column/row requirements for synthetic partitioning.
"""

import os
import json
import argparse
import pyarrow.parquet as pq
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from typing import Optional, Dict, List, Tuple
import random


def extract_tables_from_parquet(
    parquet_path: str,
    min_columns: int = 6,
    min_rows: int = 50,
) -> Optional[Dict]:
    """
    Extract a table from a parquet file if it meets size requirements.
    
    Args:
        parquet_path: Path to parquet file
        min_columns: Minimum columns required for vertical split
        min_rows: Minimum rows required for horizontal split
        
    Returns:
        Dict with table data and metadata, or None if doesn't meet requirements
    """
    try:
        # Read parquet file
        table = pq.read_table(parquet_path)
        df = table.to_pandas()
        
        # Check size requirements
        num_cols = len(df.columns)
        num_rows = len(df)
        
        # Determine which split types are valid
        valid_vertical = num_cols >= min_columns
        valid_horizontal = num_rows >= min_rows
        
        if not valid_vertical and not valid_horizontal:
            return None
        
        # Extract metadata from parquet schema
        metadata = {}
        if table.schema.metadata:
            for key, value in table.schema.metadata.items():
                try:
                    if isinstance(key, bytes):
                        key = key.decode('utf-8')
                    if isinstance(value, bytes):
                        value = value.decode('utf-8')
                    # Try to parse JSON metadata
                    try:
                        metadata[key] = json.loads(value)
                    except json.JSONDecodeError:
                        metadata[key] = value
                except Exception:
                    pass
        
        return {
            'df': df,
            'num_columns': num_cols,
            'num_rows': num_rows,
            'valid_vertical': valid_vertical,
            'valid_horizontal': valid_horizontal,
            'source_path': parquet_path,
            'metadata': metadata,
        }
        
    except Exception as e:
        print(f"Error reading {parquet_path}: {e}")
        return None


def extract_tables_from_topic(
    topic_dir: str,
    min_columns: int = 6,
    min_rows: int = 50,
    max_tables: Optional[int] = None,
) -> List[Dict]:
    """
    Extract all valid tables from a topic directory.
    
    Args:
        topic_dir: Path to extracted topic directory (contains parquet files)
        min_columns: Minimum columns for vertical split
        min_rows: Minimum rows for horizontal split
        max_tables: Maximum tables to extract (for testing)
        
    Returns:
        List of table dictionaries
    """
    topic_path = Path(topic_dir)
    parquet_files = list(topic_path.glob("*.parquet"))
    
    if max_tables:
        parquet_files = parquet_files[:max_tables]
    
    tables = []
    for pf in tqdm(parquet_files, desc=f"Extracting tables from {topic_path.name}"):
        table_data = extract_tables_from_parquet(
            str(pf), min_columns=min_columns, min_rows=min_rows
        )
        if table_data:
            table_data['topic'] = topic_path.name
            table_data['table_id'] = pf.stem
            tables.append(table_data)
    
    return tables


def scan_gittables_directory(
    gittables_dir: str,
    min_columns: int = 6,
    min_rows: int = 50,
    max_tables_per_topic: Optional[int] = None,
    topics: Optional[List[str]] = None,
) -> Tuple[List[Dict], Dict]:
    """
    Scan the entire GitTables directory and extract valid tables.
    
    Args:
        gittables_dir: Path to GitTables data directory
        min_columns: Minimum columns for vertical split
        min_rows: Minimum rows for horizontal split
        max_tables_per_topic: Maximum tables per topic (for testing)
        topics: List of specific topics to process (None = all)
        
    Returns:
        Tuple of (list of tables, statistics dict)
    """
    gittables_path = Path(gittables_dir)
    
    # Find all topic directories (extracted from zips)
    topic_dirs = [d for d in gittables_path.iterdir() if d.is_dir()]
    
    if topics:
        topic_dirs = [d for d in topic_dirs if d.name in topics]
    
    all_tables = []
    stats = {
        'total_topics': len(topic_dirs),
        'total_tables': 0,
        'valid_vertical': 0,
        'valid_horizontal': 0,
        'valid_both': 0,
    }
    
    for topic_dir in tqdm(topic_dirs, desc="Processing topics"):
        tables = extract_tables_from_topic(
            str(topic_dir),
            min_columns=min_columns,
            min_rows=min_rows,
            max_tables=max_tables_per_topic,
        )
        
        for t in tables:
            stats['total_tables'] += 1
            if t['valid_vertical']:
                stats['valid_vertical'] += 1
            if t['valid_horizontal']:
                stats['valid_horizontal'] += 1
            if t['valid_vertical'] and t['valid_horizontal']:
                stats['valid_both'] += 1
        
        all_tables.extend(tables)
    
    return all_tables, stats


def save_table_index(tables: List[Dict], output_path: str):
    """Save table index (metadata without actual data) to JSON."""
    index = []
    for t in tables:
        index.append({
            'table_id': t['table_id'],
            'topic': t['topic'],
            'source_path': t['source_path'],
            'num_columns': t['num_columns'],
            'num_rows': t['num_rows'],
            'valid_vertical': t['valid_vertical'],
            'valid_horizontal': t['valid_horizontal'],
        })
    
    with open(output_path, 'w') as f:
        json.dump(index, f, indent=2)
    
    print(f"Saved table index with {len(index)} tables to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Extract tables from GitTables")
    parser.add_argument("--gittables-dir", type=str, default="data/GitTables",
                        help="Path to GitTables directory")
    parser.add_argument("--output", type=str, default="data/GitTables/table_index.json",
                        help="Output path for table index")
    parser.add_argument("--min-columns", type=int, default=6,
                        help="Minimum columns for vertical split")
    parser.add_argument("--min-rows", type=int, default=50,
                        help="Minimum rows for horizontal split")
    parser.add_argument("--max-tables-per-topic", type=int, default=None,
                        help="Maximum tables per topic (for testing)")
    parser.add_argument("--topics", type=str, nargs="+", default=None,
                        help="Specific topics to process")
    
    args = parser.parse_args()
    
    # Scan and extract tables
    tables, stats = scan_gittables_directory(
        args.gittables_dir,
        min_columns=args.min_columns,
        min_rows=args.min_rows,
        max_tables_per_topic=args.max_tables_per_topic,
        topics=args.topics,
    )
    
    # Print statistics
    print("\n=== Extraction Statistics ===")
    print(f"Total topics processed: {stats['total_topics']}")
    print(f"Total valid tables: {stats['total_tables']}")
    print(f"Valid for vertical split: {stats['valid_vertical']}")
    print(f"Valid for horizontal split: {stats['valid_horizontal']}")
    print(f"Valid for both splits: {stats['valid_both']}")
    
    # Save table index
    save_table_index(tables, args.output)


if __name__ == "__main__":
    main()
