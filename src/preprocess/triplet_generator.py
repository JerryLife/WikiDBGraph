"""
Triplet dataset generation for contrastive learning.

Generates (anchor, positive, negatives) triplets from qid_pairs for training
embedding models with InfoNCE loss.
"""

import os
import csv
import json
import random
from pathlib import Path
from collections import defaultdict
from typing import List, Tuple, Dict, Optional


class TripletGenerator:
    """
    Generate triplet datasets for contrastive learning.
    
    Creates (anchor, positive, negatives) triplets from QID pairs where:
    - anchor/positive: databases sharing the same Wikidata topic QID
    - negatives: randomly sampled unrelated databases
    
    Args:
        num_negatives: Number of negative samples per triplet (default: 6)
        seed: Random seed for reproducibility (default: 42)
        train_ratio: Ratio of data for training (default: 0.7)
        val_ratio: Ratio of data for validation (default: 0.1)
    
    Example:
        >>> generator = TripletGenerator(num_negatives=6, seed=42)
        >>> generator.generate(
        ...     qid_pairs_path="data/qid_pairs.csv",
        ...     negative_pool_path="data/negative_candidates.csv",
        ...     output_dir="data/split_triplets"
        ... )
    """
    
    SPLIT_SEED = 2024  # Fixed seed for train/val/test split consistency
    
    def __init__(
        self,
        num_negatives: int = 6,
        seed: int = 42,
        train_ratio: float = 0.7,
        val_ratio: float = 0.1
    ):
        """Initialize the triplet generator."""
        if num_negatives < 1:
            raise ValueError("num_negatives must be at least 1")
        if not (0 < train_ratio + val_ratio < 1):
            raise ValueError("train_ratio + val_ratio must be between 0 and 1")
        
        self.num_negatives = num_negatives
        self.seed = seed
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
    
    def load_qid_pairs(self, qid_pairs_path: str) -> Dict[str, List[Tuple[str, str]]]:
        """
        Load QID pairs from CSV file.
        
        Args:
            qid_pairs_path: Path to CSV with columns: qid, db_1, db_2
        
        Returns:
            Dict mapping QID to list of (db_1, db_2) pairs
        """
        qid_groups = defaultdict(list)
        with open(qid_pairs_path, newline='', encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                qid = row["qid"]
                db1 = row["db_1"].zfill(5)
                db2 = row["db_2"].zfill(5)
                qid_groups[qid].append((db1, db2))
        return qid_groups
    
    def load_negative_pool(self, negative_pool_path: str) -> List[str]:
        """
        Load pool of negative candidate database IDs.
        
        Args:
            negative_pool_path: Path to CSV with negative candidate IDs
        
        Returns:
            List of database IDs
        """
        with open(negative_pool_path, newline='', encoding='utf-8') as f:
            next(f)  # skip header
            return [line.strip() for line in f]
    
    def split_by_qid(
        self,
        qid_groups: Dict[str, List[Tuple[str, str]]]
    ) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]], List[Tuple[str, str]]]:
        """
        Split pairs into train/val/test by QID to prevent data leakage.
        
        Uses a fixed seed for reproducibility of splits.
        
        Args:
            qid_groups: Dict mapping QID to list of pairs
        
        Returns:
            Tuple of (train_pairs, val_pairs, test_pairs)
        """
        all_qids = list(qid_groups.keys())
        random.Random(self.SPLIT_SEED).shuffle(all_qids)
        
        total_pairs = sum(len(pairs) for pairs in qid_groups.values())
        target_train = int(total_pairs * self.train_ratio)
        target_val = int(total_pairs * self.val_ratio)
        
        train_pairs = []
        val_pairs = []
        test_pairs = []
        
        count_train = 0
        count_val = 0
        
        for qid in all_qids:
            group = qid_groups[qid]
            group_size = len(group)
            
            if count_train + group_size <= target_train:
                train_pairs.extend(group)
                count_train += group_size
            elif count_val + group_size <= target_val:
                val_pairs.extend(group)
                count_val += group_size
            else:
                test_pairs.extend(group)
        
        return train_pairs, val_pairs, test_pairs
    
    def generate_triplets(
        self,
        pairs: List[Tuple[str, str]],
        negative_pool: List[str],
        seed: int
    ) -> List[Dict]:
        """
        Generate triplets from pairs with random negatives.
        
        Args:
            pairs: List of (anchor, positive) pairs
            negative_pool: Pool of negative candidate IDs
            seed: Random seed for this generation
        
        Returns:
            List of triplet dictionaries
        """
        random.seed(seed)
        shuffled_pool = negative_pool.copy()
        random.shuffle(shuffled_pool)
        
        neg_index = 0
        triplets = []
        
        for anchor, positive in pairs:
            # Get next batch of negatives
            negatives = shuffled_pool[neg_index:neg_index + self.num_negatives]
            neg_index += self.num_negatives
            
            # Wrap around if we run out
            if len(negatives) < self.num_negatives:
                neg_index = 0
                random.shuffle(shuffled_pool)
                negatives = shuffled_pool[:self.num_negatives]
                neg_index = self.num_negatives
            
            triplet = {
                "anchor": anchor,
                "positive": positive,
                "negatives": negatives
            }
            triplets.append(triplet)
        
        return triplets
    
    def save_triplets(
        self,
        triplets: List[Dict],
        output_path: str
    ) -> None:
        """
        Save triplets to JSONL file.
        
        Args:
            triplets: List of triplet dictionaries
            output_path: Output file path
        """
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for triplet in triplets:
                f.write(json.dumps(triplet) + "\n")
        print(f"✅ Saved {len(triplets)} triplets to {output_path}")
    
    def generate(
        self,
        qid_pairs_path: str,
        negative_pool_path: str,
        output_dir: str,
        test_seeds: Optional[List[int]] = None
    ) -> Dict[str, str]:
        """
        Generate complete triplet dataset with train/val/test splits.
        
        Args:
            qid_pairs_path: Path to QID pairs CSV
            negative_pool_path: Path to negative candidates CSV
            output_dir: Output directory for JSONL files
            test_seeds: List of seeds for generating test set variants (default: [42-46])
        
        Returns:
            Dict mapping split names to output file paths
        """
        if test_seeds is None:
            test_seeds = [42, 43, 44, 45, 46]
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Load data
        qid_groups = self.load_qid_pairs(qid_pairs_path)
        negative_pool = self.load_negative_pool(negative_pool_path)
        
        print(f"Loaded {sum(len(v) for v in qid_groups.values())} pairs from {len(qid_groups)} QIDs")
        print(f"Loaded {len(negative_pool)} negative candidates")
        
        # Split by QID
        train_pairs, val_pairs, test_pairs = self.split_by_qid(qid_groups)
        print(f"Split: train={len(train_pairs)}, val={len(val_pairs)}, test={len(test_pairs)}")
        
        output_paths = {}
        
        # Generate train set
        train_triplets = self.generate_triplets(train_pairs, negative_pool, self.seed)
        train_path = os.path.join(output_dir, "triplets_train.jsonl")
        self.save_triplets(train_triplets, train_path)
        output_paths["train"] = train_path
        
        # Generate val set
        val_triplets = self.generate_triplets(val_pairs, negative_pool, self.seed + 1)
        val_path = os.path.join(output_dir, "triplets_val.jsonl")
        self.save_triplets(val_triplets, val_path)
        output_paths["val"] = val_path
        
        # Generate test sets with multiple seeds
        for seed in test_seeds:
            test_triplets = self.generate_triplets(test_pairs, negative_pool, seed)
            test_path = os.path.join(output_dir, f"triplets_test_seed{seed}.jsonl")
            self.save_triplets(test_triplets, test_path)
            output_paths[f"test_seed{seed}"] = test_path
        
        print(f"✅ Generated triplets with {self.num_negatives} negatives per triplet")
        return output_paths
    
    @classmethod
    def from_config(cls, config: "PreprocessConfig") -> "TripletGenerator":
        """Create a TripletGenerator from a PreprocessConfig."""
        from .config import PreprocessConfig
        return cls(
            num_negatives=config.num_negatives,
            seed=config.seed
        )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate triplet dataset for contrastive learning")
    parser.add_argument("--qid-pairs", type=str, required=True, help="Path to QID pairs CSV")
    parser.add_argument("--negative-pool", type=str, required=True, help="Path to negative candidates CSV")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
    parser.add_argument("--num-negatives", type=int, default=6, help="Negatives per triplet (default: 6)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    generator = TripletGenerator(num_negatives=args.num_negatives, seed=args.seed)
    generator.generate(
        qid_pairs_path=args.qid_pairs,
        negative_pool_path=args.negative_pool,
        output_dir=args.output_dir
    )
