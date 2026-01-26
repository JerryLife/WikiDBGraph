"""
Preprocessing configuration with centralized hyperparameter defaults.

All defaults match the existing implementation for backward compatibility.
"""

from dataclasses import dataclass, field
from typing import Literal
from enum import Enum


class SerializationMode(str, Enum):
    """Schema serialization modes for ablation studies."""
    SCHEMA_ONLY = "schema_only"  # Table and column names only
    DATA_ONLY = "data_only"      # Only representative sample values
    FULL = "full"                # Combined format (default)


@dataclass
class PreprocessConfig:
    """
    Centralized configuration for preprocessing pipeline.
    
    Attributes:
        serialization_mode: How to serialize database schemas.
            - "schema_only": Names of tables and columns only
            - "data_only": Only the representative sample values
            - "full": Current combined format (default, backward-compatible)
        sample_size: Number of representative values per column (default: 3)
        show_wikidata_property_id: Whether to include Wikidata property IDs
        num_negatives: Number of negative samples per triplet (default: 6)
        similarity_threshold: Threshold for edge filtering (default: 0.6713)
        batch_size: Batch size for embedding generation
        seed: Random seed for reproducibility
    """
    
    # Schema serialization
    serialization_mode: Literal["schema_only", "data_only", "full"] = "full"
    sample_size: int = 3  # matches format_schema_from_loader default
    show_wikidata_property_id: bool = False
    
    # Triplet generation
    num_negatives: int = 6  # matches split_dataset.py default
    
    # Similarity computation
    similarity_threshold: float = 0.6713  # matches test_all_possible_pairs default
    
    # Embedding generation
    embedding_batch_size: int = 8
    db_id_range: tuple = (0, 100000)
    
    # Graph building
    num_nodes: int = 100000
    
    # General
    seed: int = 42
    
    def __post_init__(self):
        """Validate configuration values."""
        if self.serialization_mode not in ["schema_only", "data_only", "full"]:
            raise ValueError(
                f"Invalid serialization_mode: {self.serialization_mode}. "
                "Must be one of: schema_only, data_only, full"
            )
        if self.sample_size < 1:
            raise ValueError("sample_size must be at least 1")
        if self.num_negatives < 1:
            raise ValueError("num_negatives must be at least 1")
        if not (0.0 < self.similarity_threshold < 1.0):
            raise ValueError("similarity_threshold must be between 0 and 1")
