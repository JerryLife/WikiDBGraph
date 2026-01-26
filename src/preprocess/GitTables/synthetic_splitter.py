"""
Synthetic Splitter for GitTables

Creates synthetic vertical and horizontal splits from tables for
self-supervised contrastive learning in Collaborative Learning scenarios.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, List
import random


def vertical_split(
    df: pd.DataFrame,
    overlap_ratio: float = 0.0,
    min_cols_per_split: int = 2,
    seed: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split a table vertically (by columns) into two parts.
    
    Simulates Vertical Collaborative Learning where different parties
    hold different features for the same instances.
    
    Args:
        df: Input DataFrame
        overlap_ratio: Ratio of columns to overlap (0-1)
        min_cols_per_split: Minimum columns per split
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (left_split, right_split) DataFrames
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    columns = list(df.columns)
    n_cols = len(columns)
    
    if n_cols < 2 * min_cols_per_split:
        raise ValueError(f"Table needs at least {2 * min_cols_per_split} columns for vertical split")
    
    # Shuffle columns
    shuffled_cols = columns.copy()
    random.shuffle(shuffled_cols)
    
    # Calculate split point
    n_overlap = int(n_cols * overlap_ratio)
    
    if overlap_ratio > 0:
        # With overlap
        left_size = (n_cols + n_overlap) // 2
        right_start = left_size - n_overlap
        
        left_cols = shuffled_cols[:left_size]
        right_cols = shuffled_cols[right_start:]
    else:
        # No overlap - disjoint split
        split_point = n_cols // 2
        left_cols = shuffled_cols[:split_point]
        right_cols = shuffled_cols[split_point:]
    
    left_df = df[left_cols].copy()
    right_df = df[right_cols].copy()
    
    return left_df, right_df


def horizontal_split(
    df: pd.DataFrame,
    split_ratio: float = 0.5,
    shuffle_rows: bool = True,
    seed: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split a table horizontally (by rows) into two parts.
    
    Simulates Horizontal Collaborative Learning where different parties
    hold different instances with the same features.
    
    Args:
        df: Input DataFrame
        split_ratio: Ratio of rows for the first split (0-1)
        shuffle_rows: Whether to shuffle rows before splitting
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (top_split, bottom_split) DataFrames
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    n_rows = len(df)
    
    if shuffle_rows:
        indices = list(range(n_rows))
        random.shuffle(indices)
        df = df.iloc[indices].reset_index(drop=True)
    
    split_point = int(n_rows * split_ratio)
    
    top_df = df.iloc[:split_point].copy()
    bottom_df = df.iloc[split_point:].copy()
    
    return top_df, bottom_df


def generate_split_pair(
    df: pd.DataFrame,
    split_type: str = "vertical",
    overlap_ratio: float = 0.0,
    split_ratio: float = 0.5,
    seed: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    """
    Generate a split pair from a table.
    
    Args:
        df: Input DataFrame
        split_type: "vertical" or "horizontal"
        overlap_ratio: For vertical split, ratio of overlapping columns
        split_ratio: For horizontal split, ratio for first partition
        seed: Random seed
        
    Returns:
        Tuple of (anchor_df, positive_df, split_type)
    """
    if split_type == "vertical":
        anchor, positive = vertical_split(df, overlap_ratio=overlap_ratio, seed=seed)
    elif split_type == "horizontal":
        anchor, positive = horizontal_split(df, split_ratio=split_ratio, seed=seed)
    else:
        raise ValueError(f"Unknown split_type: {split_type}")
    
    return anchor, positive, split_type


def sample_negative_split(
    tables: List[dict],
    exclude_table_id: str,
    split_type: str = "vertical",
    seed: Optional[int] = None,
) -> Tuple[pd.DataFrame, str]:
    """
    Sample a negative split from a random different table.
    
    Args:
        tables: List of table dictionaries with 'df' and 'table_id'
        exclude_table_id: Table ID to exclude (the anchor's source)
        split_type: Type of split to create
        seed: Random seed
        
    Returns:
        Tuple of (negative_df, source_table_id)
    """
    if seed is not None:
        random.seed(seed)
    
    # Filter valid tables
    if split_type == "vertical":
        valid_tables = [t for t in tables if t['valid_vertical'] and t['table_id'] != exclude_table_id]
    else:
        valid_tables = [t for t in tables if t['valid_horizontal'] and t['table_id'] != exclude_table_id]
    
    if not valid_tables:
        raise ValueError("No valid tables available for negative sampling")
    
    # Sample random table
    neg_table = random.choice(valid_tables)
    
    # Generate split
    if split_type == "vertical":
        neg_left, neg_right = vertical_split(neg_table['df'], seed=seed)
        # Return random side
        neg_split = random.choice([neg_left, neg_right])
    else:
        neg_top, neg_bottom = horizontal_split(neg_table['df'], seed=seed)
        neg_split = random.choice([neg_top, neg_bottom])
    
    return neg_split, neg_table['table_id']


class SplitGenerator:
    """
    Generator for creating synthetic splits for contrastive learning.
    """
    
    def __init__(
        self,
        tables: List[dict],
        split_type: str = "both",  # "vertical", "horizontal", or "both"
        vertical_overlap: float = 0.0,
        horizontal_ratio: float = 0.5,
        num_negatives: int = 2,
        seed: int = 42,
    ):
        """
        Initialize the split generator.
        
        Args:
            tables: List of table dictionaries
            split_type: Type of splits to generate
            vertical_overlap: Overlap ratio for vertical splits
            horizontal_ratio: Split ratio for horizontal splits
            num_negatives: Number of negatives per positive pair
            seed: Random seed
        """
        self.tables = tables
        self.split_type = split_type
        self.vertical_overlap = vertical_overlap
        self.horizontal_ratio = horizontal_ratio
        self.num_negatives = num_negatives
        self.seed = seed
        
        # Filter tables by valid split types
        if split_type == "vertical":
            self.valid_tables = [t for t in tables if t['valid_vertical']]
        elif split_type == "horizontal":
            self.valid_tables = [t for t in tables if t['valid_horizontal']]
        else:  # both
            self.valid_tables = [t for t in tables if t['valid_vertical'] or t['valid_horizontal']]
        
        print(f"SplitGenerator initialized with {len(self.valid_tables)} valid tables")
    
    def generate_triplets(self, max_triplets: Optional[int] = None):
        """
        Generate triplets (anchor, positive, negatives) for contrastive learning.
        
        Yields:
            Dict with 'anchor', 'positive', 'negatives', 'split_type', 'table_id'
        """
        random.seed(self.seed)
        np.random.seed(self.seed)
        
        triplet_count = 0
        
        for table in self.valid_tables:
            if max_triplets and triplet_count >= max_triplets:
                break
            
            table_id = table['table_id']
            df = table['df']
            
            # Generate vertical split triplet if valid
            if self.split_type in ["vertical", "both"] and table['valid_vertical']:
                try:
                    anchor, positive, _ = generate_split_pair(
                        df, split_type="vertical",
                        overlap_ratio=self.vertical_overlap,
                        seed=self.seed + triplet_count
                    )
                    
                    # Sample negatives
                    negatives = []
                    for i in range(self.num_negatives):
                        neg_df, neg_id = sample_negative_split(
                            self.tables, table_id, split_type="vertical",
                            seed=self.seed + triplet_count + i
                        )
                        negatives.append({'df': neg_df, 'source_id': neg_id})
                    
                    yield {
                        'anchor': anchor,
                        'positive': positive,
                        'negatives': negatives,
                        'split_type': 'vertical',
                        'table_id': table_id,
                        'topic': table.get('topic', 'unknown'),
                    }
                    triplet_count += 1
                except Exception as e:
                    print(f"Error generating vertical triplet for {table_id}: {e}")
            
            # Generate horizontal split triplet if valid
            if self.split_type in ["horizontal", "both"] and table['valid_horizontal']:
                if max_triplets and triplet_count >= max_triplets:
                    break
                    
                try:
                    anchor, positive, _ = generate_split_pair(
                        df, split_type="horizontal",
                        split_ratio=self.horizontal_ratio,
                        seed=self.seed + triplet_count
                    )
                    
                    # Sample negatives
                    negatives = []
                    for i in range(self.num_negatives):
                        neg_df, neg_id = sample_negative_split(
                            self.tables, table_id, split_type="horizontal",
                            seed=self.seed + triplet_count + i
                        )
                        negatives.append({'df': neg_df, 'source_id': neg_id})
                    
                    yield {
                        'anchor': anchor,
                        'positive': positive,
                        'negatives': negatives,
                        'split_type': 'horizontal',
                        'table_id': table_id,
                        'topic': table.get('topic', 'unknown'),
                    }
                    triplet_count += 1
                except Exception as e:
                    print(f"Error generating horizontal triplet for {table_id}: {e}")
        
        print(f"Generated {triplet_count} triplets")
