"""
Semantic Column Matcher for Horizontal Federated Learning.

This module performs plain text matching of columns across databases
and generates a validation report for human review.
"""

import os
import sys
import argparse
import json
import re
from typing import Dict, List, Set, Tuple
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.WKDataset import WKDataset


def normalize_column_name(col_name: str) -> str:
    """
    Normalize column name for matching:
    - Convert to lowercase
    - Replace underscores, spaces, hyphens with empty string
    - Remove common prefixes/suffixes
    """
    normalized = col_name.lower()
    # Remove common separators
    normalized = re.sub(r'[_\s\-]', '', normalized)
    return normalized


def get_all_columns(wk: WKDataset, db_id: str) -> Dict[str, List[str]]:
    """
    Get all columns from a database, organized by table.
    
    Returns:
        Dict mapping table_name -> list of column names
    """
    schema = wk.load_database(db_id)
    table_columns = {}
    
    for table in schema.get("tables", []):
        table_name = table["table_name"]
        columns = [col["column_name"] for col in table.get("columns", [])]
        table_columns[table_name] = columns
    
    return table_columns


def get_flat_columns(wk: WKDataset, db_id: str) -> Set[str]:
    """Get all column names from a database as a flat set."""
    table_columns = get_all_columns(wk, db_id)
    all_cols = set()
    for cols in table_columns.values():
        all_cols.update(cols)
    return all_cols


def match_columns_across_databases(
    wk: WKDataset, 
    db_ids: List[str]
) -> Tuple[Dict[str, Dict[str, str]], Dict[str, List[str]], Dict[str, List[bool]]]:
    """
    Match columns across multiple databases using plain text matching.
    Uses UNION approach - all columns from all databases are included.
    
    Returns:
        union_columns: Dict mapping normalized_name -> {db_id: original_col_name or None}
        omitted_columns: Dict mapping db_id -> list of columns only in that database
        feature_masks: Dict mapping db_id -> list of booleans (True if db has that column)
    """
    # Collect all columns from all databases
    db_columns = {}  # db_id -> {normalized_name: original_name}
    db_original = {}  # db_id -> set of original names
    
    for db_id in db_ids:
        columns = get_flat_columns(wk, db_id)
        db_columns[db_id] = {normalize_column_name(col): col for col in columns}
        db_original[db_id] = columns
    
    # Build UNION of all normalized column names
    all_normalized = set()
    for db_id in db_ids:
        all_normalized.update(db_columns[db_id].keys())
    
    # Build union columns mapping (one entry per normalized name)
    union_columns = {}
    for norm_name in sorted(all_normalized):
        union_columns[norm_name] = {
            db_id: db_columns[db_id].get(norm_name, None)
            for db_id in db_ids
        }
    
    # Find omitted columns per database (columns not in that db but in union)
    omitted_columns = {}
    for db_id in db_ids:
        present = set(db_columns[db_id].keys())
        omitted = all_normalized - present
        omitted_columns[db_id] = sorted(list(omitted))
    
    # Build feature masks: for each db, True if column exists
    feature_masks = {}
    sorted_cols = sorted(all_normalized)
    for db_id in db_ids:
        mask = [norm_name in db_columns[db_id] for norm_name in sorted_cols]
        feature_masks[db_id] = mask
    
    return union_columns, omitted_columns, feature_masks


def print_matching_report(
    union_columns: Dict[str, Dict[str, str]],
    omitted_columns: Dict[str, List[str]],
    feature_masks: Dict[str, List[bool]],
    db_ids: List[str]
):
    """Print a human-readable matching report for union schema."""
    print("\n" + "=" * 70)
    print("COLUMN MATCHING REPORT (UNION SCHEMA)")
    print("=" * 70)
    
    print(f"\nDatabases analyzed: {', '.join(db_ids)}")
    print(f"Total union columns: {len(union_columns)}")
    
    # Count shared vs unique columns
    shared_count = sum(1 for mapping in union_columns.values() 
                       if all(v is not None for v in mapping.values()))
    print(f"Columns present in ALL databases: {shared_count}")
    
    # Print feature coverage per database
    print("\n" + "-" * 70)
    print("FEATURE COVERAGE (per database):")
    print("-" * 70)
    
    for db_id in db_ids:
        mask = feature_masks[db_id]
        present = sum(mask)
        total = len(mask)
        pct = 100 * present / total if total > 0 else 0
        print(f"  DB{db_id}: {present}/{total} columns ({pct:.1f}%)")
    
    # Print columns shared by all (first 10)
    print("\n" + "-" * 70)
    print(f"SHARED COLUMNS (present in all {len(db_ids)} databases):")
    print("-" * 70)
    
    shared = [(n, m) for n, m in union_columns.items() 
              if all(v is not None for v in m.values())]
    
    for norm_name, db_mapping in shared[:10]:
        print(f"  {norm_name}")
        for db_id in db_ids:
            print(f"    - DB{db_id}: {db_mapping[db_id]}")
    if len(shared) > 10:
        print(f"  ... and {len(shared) - 10} more shared columns")
    
    # Print missing columns per database
    print("\n" + "-" * 70)
    print("MISSING COLUMNS (per database - will use mask=0):")
    print("-" * 70)
    
    for db_id in db_ids:
        omitted = omitted_columns[db_id]
        print(f"\n  DB{db_id}: {len(omitted)} columns missing")
        if omitted:
            for col in omitted[:10]:
                print(f"    - {col}")
            if len(omitted) > 10:
                print(f"    ... and {len(omitted) - 10} more")
    
    print("\n" + "=" * 70)


def save_column_mapping(
    union_columns: Dict[str, Dict[str, str]],
    omitted_columns: Dict[str, List[str]],
    feature_masks: Dict[str, List[bool]],
    output_path: str
):
    """Save the column mapping with feature masks to a JSON file."""
    sorted_cols = sorted(union_columns.keys())
    
    result = {
        "union_columns": union_columns,
        "omitted_columns": omitted_columns,
        "feature_masks": feature_masks,
        "column_order": sorted_cols,
        "union_size": len(union_columns)
    }
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\nColumn mapping saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Match columns across databases for horizontal FL"
    )
    parser.add_argument(
        "--db_ids",
        type=str,
        default="54379,37176,85770,50469",
        help="Comma-separated database IDs (default: 54379,37176,85770,50469)"
    )
    parser.add_argument(
        "--schema_dir",
        type=str,
        default="data/schema",
        help="Schema directory (default: data/schema)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/clean/column_matching.json",
        help="Output JSON file path (default: data/clean/column_matching.json)"
    )
    
    args = parser.parse_args()
    
    # Parse database IDs
    db_ids = [db_id.strip() for db_id in args.db_ids.split(",")]
    
    # Pad db_ids to 5 digits
    db_ids = [db_id.zfill(5) for db_id in db_ids]
    
    print(f"Matching columns across databases: {db_ids}")
    
    # Initialize WKDataset
    wk = WKDataset(schema_dir=args.schema_dir)
    
    # Perform matching (union approach)
    union_columns, omitted_columns, feature_masks = match_columns_across_databases(wk, db_ids)
    
    # Print report
    print_matching_report(union_columns, omitted_columns, feature_masks, db_ids)
    
    # Save mapping
    save_column_mapping(union_columns, omitted_columns, feature_masks, args.output)
    
    return union_columns, omitted_columns, feature_masks


if __name__ == "__main__":
    main()
