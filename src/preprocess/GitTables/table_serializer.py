"""
Table Serializer for GitTables

Serializes table splits into text format for embedding models.
Similar to schema_serializer.py but adapted for individual tables.
"""

import pandas as pd
from typing import List, Optional
import random


def serialize_table(
    df: pd.DataFrame,
    mode: str = "full",
    sample_size: int = 3,
    max_columns: int = 50,
    seed: Optional[int] = None,
) -> str:
    """
    Serialize a table (or table split) into text for embedding.
    
    Args:
        df: DataFrame to serialize
        mode: Serialization mode - "schema_only", "data_only", or "full"
        sample_size: Number of sample values per column
        max_columns: Maximum columns to include
        seed: Random seed for sampling
        
    Returns:
        Serialized text representation
    """
    if seed is not None:
        random.seed(seed)
    
    columns = list(df.columns)[:max_columns]
    parts = []
    
    for col in columns:
        col_text = serialize_column(
            df, col, mode=mode, sample_size=sample_size, seed=seed
        )
        if col_text:
            parts.append(col_text)
    
    return " | ".join(parts)


def serialize_column(
    df: pd.DataFrame,
    column: str,
    mode: str = "full",
    sample_size: int = 3,
    seed: Optional[int] = None,
) -> str:
    """
    Serialize a single column into text.
    
    Args:
        df: DataFrame containing the column
        column: Column name
        mode: "schema_only", "data_only", or "full"
        sample_size: Number of sample values
        seed: Random seed for sampling
        
    Returns:
        Serialized column text
    """
    col_name = str(column).strip()
    
    if mode == "schema_only":
        return f"[{col_name}]"
    
    # Get sample values
    values = df[column].dropna().astype(str).tolist()
    
    if not values:
        if mode == "data_only":
            return ""
        return f"[{col_name}]"
    
    # Sample values
    if seed is not None:
        random.seed(seed)
    
    if len(values) > sample_size:
        sample_values = random.sample(values, sample_size)
    else:
        sample_values = values
    
    # Clean values
    sample_values = [v.strip()[:100] for v in sample_values]  # Truncate long values
    values_text = ", ".join(sample_values)
    
    if mode == "data_only":
        return values_text
    
    # Full mode
    return f"[{col_name}]: {values_text}"


def serialize_triplet(
    triplet: dict,
    mode: str = "full",
    sample_size: int = 3,
) -> dict:
    """
    Serialize a complete triplet (anchor, positive, negatives).
    
    Args:
        triplet: Dict with 'anchor', 'positive', 'negatives' DataFrames
        mode: Serialization mode
        sample_size: Sample values per column
        
    Returns:
        Dict with serialized texts
    """
    anchor_text = serialize_table(triplet['anchor'], mode=mode, sample_size=sample_size)
    positive_text = serialize_table(triplet['positive'], mode=mode, sample_size=sample_size)
    
    negative_texts = []
    for neg in triplet['negatives']:
        neg_text = serialize_table(neg['df'], mode=mode, sample_size=sample_size)
        negative_texts.append(neg_text)
    
    return {
        'anchor': anchor_text,
        'positive': positive_text,
        'negatives': negative_texts,
        'split_type': triplet['split_type'],
        'table_id': triplet['table_id'],
        'topic': triplet.get('topic', 'unknown'),
    }


class TableSerializer:
    """
    Serializer for table splits with caching support.
    """
    
    def __init__(
        self,
        mode: str = "full",
        sample_size: int = 3,
        max_columns: int = 50,
    ):
        self.mode = mode
        self.sample_size = sample_size
        self.max_columns = max_columns
        self._cache = {}
    
    def serialize(self, df: pd.DataFrame, cache_key: Optional[str] = None) -> str:
        """Serialize a table with optional caching."""
        if cache_key and cache_key in self._cache:
            return self._cache[cache_key]
        
        text = serialize_table(
            df, mode=self.mode, sample_size=self.sample_size,
            max_columns=self.max_columns
        )
        
        if cache_key:
            self._cache[cache_key] = text
        
        return text
    
    def clear_cache(self):
        """Clear the serialization cache."""
        self._cache = {}
