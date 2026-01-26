"""
Triplet Generator for GitTables

Generates triplets for contrastive learning from synthetic table splits.
Outputs JSONL format compatible with the existing trainer.
"""

import os
import json
import argparse
from pathlib import Path
from tqdm import tqdm
from typing import Optional, List
import random

from .table_extractor import scan_gittables_directory, extract_tables_from_parquet
from .synthetic_splitter import SplitGenerator
from .table_serializer import serialize_triplet


def generate_triplets_jsonl(
    gittables_dir: str,
    output_dir: str,
    split_type: str = "both",
    mode: str = "full",
    sample_size: int = 3,
    num_negatives: int = 2,
    min_columns: int = 6,
    min_rows: int = 50,
    max_tables_per_topic: Optional[int] = None,
    topics: Optional[List[str]] = None,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
):
    """
    Generate triplets from GitTables and save to JSONL files.
    
    Args:
        gittables_dir: Path to GitTables directory
        output_dir: Output directory for triplet files
        split_type: "vertical", "horizontal", or "both"
        mode: Serialization mode - "schema_only", "data_only", "full"
        sample_size: Sample values per column
        num_negatives: Number of negatives per triplet
        min_columns: Minimum columns for vertical split
        min_rows: Minimum rows for horizontal split
        max_tables_per_topic: Maximum tables per topic (for testing)
        topics: Specific topics to process
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set
        seed: Random seed
    """
    random.seed(seed)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("=== GitTables Triplet Generation ===")
    print(f"Split type: {split_type}")
    print(f"Serialization mode: {mode}")
    print(f"Num negatives: {num_negatives}")
    print(f"Sample size: {sample_size}")
    print()
    
    # Extract tables
    print("Step 1: Extracting tables from parquet files...")
    tables, stats = scan_gittables_directory(
        gittables_dir,
        min_columns=min_columns,
        min_rows=min_rows,
        max_tables_per_topic=max_tables_per_topic,
        topics=topics,
    )
    
    print(f"\nExtracted {len(tables)} valid tables")
    print(f"  Valid for vertical split: {stats['valid_vertical']}")
    print(f"  Valid for horizontal split: {stats['valid_horizontal']}")
    print()
    
    if not tables:
        print("No valid tables found!")
        return
    
    # Generate triplets
    print("Step 2: Generating synthetic splits and triplets...")
    split_generator = SplitGenerator(
        tables,
        split_type=split_type,
        num_negatives=num_negatives,
        seed=seed,
    )
    
    all_triplets = []
    for triplet in tqdm(split_generator.generate_triplets(), desc="Generating triplets"):
        serialized = serialize_triplet(triplet, mode=mode, sample_size=sample_size)
        all_triplets.append(serialized)
    
    print(f"\nGenerated {len(all_triplets)} triplets")
    
    # Shuffle and split
    random.shuffle(all_triplets)
    
    n_total = len(all_triplets)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    train_triplets = all_triplets[:n_train]
    val_triplets = all_triplets[n_train:n_train + n_val]
    test_triplets = all_triplets[n_train + n_val:]
    
    print(f"\nSplit sizes:")
    print(f"  Train: {len(train_triplets)}")
    print(f"  Val: {len(val_triplets)}")
    print(f"  Test: {len(test_triplets)}")
    
    # Save triplets
    print("\nStep 3: Saving triplets to JSONL files...")
    
    def save_jsonl(triplets, filepath):
        with open(filepath, 'w') as f:
            for t in triplets:
                # Convert to format expected by trainer
                jsonl_entry = {
                    'anchor': t['anchor'],
                    'positive': t['positive'],
                    'negatives': t['negatives'],
                    'metadata': {
                        'split_type': t['split_type'],
                        'table_id': t['table_id'],
                        'topic': t['topic'],
                    }
                }
                f.write(json.dumps(jsonl_entry) + '\n')
    
    train_path = os.path.join(output_dir, 'triplets_train.jsonl')
    val_path = os.path.join(output_dir, 'triplets_val.jsonl')
    test_path = os.path.join(output_dir, 'triplets_test.jsonl')
    
    save_jsonl(train_triplets, train_path)
    save_jsonl(val_triplets, val_path)
    save_jsonl(test_triplets, test_path)
    
    print(f"\nSaved triplets to:")
    print(f"  Train: {train_path}")
    print(f"  Val: {val_path}")
    print(f"  Test: {test_path}")
    
    # Save config
    config = {
        'gittables_dir': gittables_dir,
        'split_type': split_type,
        'mode': mode,
        'sample_size': sample_size,
        'num_negatives': num_negatives,
        'min_columns': min_columns,
        'min_rows': min_rows,
        'total_tables': len(tables),
        'total_triplets': len(all_triplets),
        'train_size': len(train_triplets),
        'val_size': len(val_triplets),
        'test_size': len(test_triplets),
        'seed': seed,
    }
    
    config_path = os.path.join(output_dir, 'triplet_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"  Config: {config_path}")
    print("\n=== Triplet Generation Complete ===")


def main():
    parser = argparse.ArgumentParser(description="Generate triplets from GitTables")
    parser.add_argument("--gittables-dir", type=str, default="data/GitTables",
                        help="Path to GitTables directory")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for triplet files")
    parser.add_argument("--split-type", type=str, default="both",
                        choices=["vertical", "horizontal", "both"],
                        help="Type of synthetic splits")
    parser.add_argument("--mode", type=str, default="full",
                        choices=["schema_only", "data_only", "full"],
                        help="Serialization mode")
    parser.add_argument("--sample-size", type=int, default=3,
                        help="Sample values per column")
    parser.add_argument("--num-negatives", type=int, default=2,
                        help="Number of negatives per triplet")
    parser.add_argument("--min-columns", type=int, default=6,
                        help="Minimum columns for vertical split")
    parser.add_argument("--min-rows", type=int, default=50,
                        help="Minimum rows for horizontal split")
    parser.add_argument("--max-tables-per-topic", type=int, default=None,
                        help="Maximum tables per topic (for testing)")
    parser.add_argument("--topics", type=str, nargs="+", default=None,
                        help="Specific topics to process")
    parser.add_argument("--train-ratio", type=float, default=0.8,
                        help="Ratio for training set")
    parser.add_argument("--val-ratio", type=float, default=0.1,
                        help="Ratio for validation set")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    args = parser.parse_args()
    
    generate_triplets_jsonl(
        gittables_dir=args.gittables_dir,
        output_dir=args.output_dir,
        split_type=args.split_type,
        mode=args.mode,
        sample_size=args.sample_size,
        num_negatives=args.num_negatives,
        min_columns=args.min_columns,
        min_rows=args.min_rows,
        max_tables_per_topic=args.max_tables_per_topic,
        topics=args.topics,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
