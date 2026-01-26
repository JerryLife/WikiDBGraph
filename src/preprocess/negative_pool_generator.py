"""
Generate negative candidate pool for triplet generation.

Creates negative_candidates.csv containing database IDs that can be used
as negative samples. Excludes databases that appear in positive pairs.
"""

import os
import csv
import argparse
from pathlib import Path
from typing import Set


def load_positive_db_ids(qid_pairs_path: str) -> Set[str]:
    """
    Load all database IDs that appear in positive pairs.
    
    Args:
        qid_pairs_path: Path to qid_pairs.csv
    
    Returns:
        Set of database IDs in positive pairs
    """
    positive_ids = set()
    
    if not os.path.exists(qid_pairs_path):
        print(f"Warning: {qid_pairs_path} not found, using empty positive set")
        return positive_ids
    
    with open(qid_pairs_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            positive_ids.add(row['db_1'].zfill(5))
            positive_ids.add(row['db_2'].zfill(5))
    
    return positive_ids


def get_all_db_ids(schema_dir: str) -> Set[str]:
    """
    Get all database IDs from schema directory.
    
    Args:
        schema_dir: Path to schema directory containing JSON files
    
    Returns:
        Set of all database IDs
    """
    all_ids = set()
    schema_path = Path(schema_dir)
    
    if not schema_path.exists():
        raise FileNotFoundError(f"Schema directory not found: {schema_dir}")
    
    for json_file in schema_path.glob("*.json"):
        # Extract db_id from filename (format: 00000_dbname.json)
        db_id = json_file.stem.split('_')[0]
        all_ids.add(db_id.zfill(5))
    
    return all_ids


def generate_negative_pool(
    schema_dir: str,
    qid_pairs_path: str,
    output_path: str,
    exclude_positives: bool = True
) -> int:
    """
    Generate negative candidate pool.
    
    Args:
        schema_dir: Path to schema directory
        qid_pairs_path: Path to qid_pairs.csv
        output_path: Output path for negative_candidates.csv
        exclude_positives: Whether to exclude DBs in positive pairs
    
    Returns:
        Number of negative candidates generated
    """
    # Get all database IDs
    all_ids = get_all_db_ids(schema_dir)
    print(f"Found {len(all_ids)} total databases in {schema_dir}")
    
    # Get positive IDs to exclude
    if exclude_positives:
        positive_ids = load_positive_db_ids(qid_pairs_path)
        print(f"Found {len(positive_ids)} unique DBs in positive pairs")
        negative_ids = all_ids - positive_ids
    else:
        negative_ids = all_ids
    
    print(f"Negative pool size: {len(negative_ids)}")
    
    # Sort for reproducibility
    sorted_negatives = sorted(negative_ids)
    
    # Save to CSV
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        f.write("db_id\n")  # header
        for db_id in sorted_negatives:
            f.write(f"{db_id}\n")
    
    print(f"✅ Saved {len(sorted_negatives)} negative candidates to {output_path}")
    return len(sorted_negatives)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate negative candidate pool")
    parser.add_argument("--schema-dir", type=str, required=True,
                        help="Path to schema directory")
    parser.add_argument("--qid-pairs", type=str, required=True,
                        help="Path to qid_pairs.csv")
    parser.add_argument("--output", type=str, required=True,
                        help="Output path for negative_candidates.csv")
    parser.add_argument("--include-positives", action="store_true",
                        help="Include DBs from positive pairs (not recommended)")
    
    args = parser.parse_args()
    
    generate_negative_pool(
        schema_dir=args.schema_dir,
        qid_pairs_path=args.qid_pairs,
        output_path=args.output,
        exclude_positives=not args.include_positives
    )
