#!/usr/bin/env python
"""
IMPORTANT: GPU selection must happen before importing torch.
Use --gpu argument or CUDA_VISIBLE_DEVICES environment variable.

Model Trainer for BGE-M3 Embeddings

This script provides a command-line interface for training the BGE-M3 embedding model
with configurable hyperparameters. It replaces the hardcoded train_bge_softmax.py to
allow the shell script to pass parameters.

Usage:
    python -m preprocess.trainer \
        --train-triplets triplets/triplets_train.jsonl \
        --val-triplets triplets/triplets_val.jsonl \
        --schema-dir data/schema \
        --csv-dir data/unzip \
        --output-dir out/model \
        --lr 1e-05 \
        --epochs 10 \
        --batch-size 32
"""

import argparse
import os
import sys


def parse_gpu_arg():
    """Parse --gpu argument early, before importing torch."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--gpu", type=str, default=None,
                        help="GPU device ID(s) to use (e.g., '0', '1', '0,1')")
    args, _ = parser.parse_known_args()
    return args.gpu


# Set GPU before importing torch
gpu_id = parse_gpu_arg()
if gpu_id is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    print(f"Setting CUDA_VISIBLE_DEVICES={gpu_id}")

# Now import torch and other modules
import torch

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.BGEEmbedder import BGEEmbedder
from model.WKDataset import WKDataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train BGE-M3 embedding model for column matching"
    )
    
    # Data paths
    parser.add_argument(
        "--train-triplets",
        type=str,
        required=True,
        help="Path to training triplets JSONL file"
    )
    parser.add_argument(
        "--val-triplets",
        type=str,
        required=True,
        help="Path to validation triplets JSONL file"
    )
    parser.add_argument(
        "--schema-dir",
        type=str,
        required=True,
        help="Path to schema directory"
    )
    parser.add_argument(
        "--csv-dir",
        type=str,
        required=True,
        help="Path to CSV data directory"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for trained model weights"
    )
    
    # Training hyperparameters
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-05,
        help="Learning rate (default: 1e-05)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs (default: 10)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Training batch size (default: 32)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.5,
        help="Temperature for InfoNCE loss (default: 0.5)"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=1024,
        help="Maximum sequence length (default: 1024)"
    )
    
    # Model options
    parser.add_argument(
        "--model-type",
        type=str,
        default="bge-m3",
        help="Model type (default: bge-m3)"
    )
    parser.add_argument(
        "--base-model-path",
        type=str,
        default=None,
        help="Path to base model (default: BAAI/bge-m3)"
    )
    
    # GPU options
    parser.add_argument(
        "--gpu",
        type=str,
        default=None,
        help="GPU device ID(s) to use (e.g., '0', '1', '0,1'). Sets CUDA_VISIBLE_DEVICES."
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Validate paths
    if not os.path.exists(args.train_triplets):
        print(f"Error: Training triplets not found: {args.train_triplets}")
        sys.exit(1)
    if not os.path.exists(args.val_triplets):
        print(f"Error: Validation triplets not found: {args.val_triplets}")
        sys.exit(1)
    if not os.path.isdir(args.schema_dir):
        print(f"Error: Schema directory not found: {args.schema_dir}")
        sys.exit(1)
    if not os.path.isdir(args.csv_dir):
        print(f"Error: CSV directory not found: {args.csv_dir}")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 60)
    print("BGE-M3 Model Training")
    print("=" * 60)
    print(f"Training triplets: {args.train_triplets}")
    print(f"Validation triplets: {args.val_triplets}")
    print(f"Schema directory: {args.schema_dir}")
    print(f"CSV directory: {args.csv_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Learning rate: {args.lr}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Temperature: {args.temperature}")
    print(f"Max length: {args.max_length}")
    print("=" * 60)
    
    # Initialize data loader
    print("\nInitializing data loader...")
    loader = WKDataset(schema_dir=args.schema_dir, csv_base_dir=args.csv_dir)
    
    # Initialize embedder
    print("Initializing BGE-M3 embedder...")
    embedder = BGEEmbedder(
        model_type=args.model_type,
        model_path=args.base_model_path
    )
    
    # Train model
    print("\nStarting training...")
    embedder.fit(
        train_path=args.train_triplets,
        val_path=args.val_triplets,
        loader=loader,
        save_dir=args.output_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        temperature=args.temperature,
        max_length=args.max_length,
    )
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Model saved to: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
