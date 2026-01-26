"""
Edge filtering based on similarity threshold.

Filters prediction results to create edge lists for graph construction.
"""

import os
import torch
import pandas as pd
import numpy as np
from typing import Optional


class EdgeFilter:
    """
    Filter edges based on similarity threshold.
    
    Takes similarity predictions and filters to keep only edges above threshold.
    
    Args:
        threshold: Similarity threshold (default: None = auto-determine from data)
        quantile: If threshold is None, use this quantile of positive labels (default: 0.15)
    
    Example:
        >>> filter = EdgeFilter(threshold=0.94)
        >>> filter.filter(
        ...     predictions_file="data/graph/all_predictions.pt",
        ...     output_file="data/graph/filtered_edges.csv"
        ... )
    """
    
    def __init__(
        self,
        threshold: Optional[float] = None,
        quantile: float = 0.15
    ):
        """Initialize the edge filter."""
        self.threshold = threshold
        self.quantile = quantile
    
    def load_predictions(self, predictions_file: str) -> pd.DataFrame:
        """
        Load predictions from file.
        
        Supports both .pt (PyTorch) and .csv formats.
        
        Args:
            predictions_file: Path to predictions file
        
        Returns:
            DataFrame with columns: src, tgt, similarity, label, edge
        """
        if predictions_file.endswith('.pt'):
            arr = torch.load(predictions_file, weights_only=False)
            if isinstance(arr, list):
                arr = np.array(arr)
            elif isinstance(arr, torch.Tensor):
                arr = arr.cpu().numpy()
            return pd.DataFrame(arr, columns=['src', 'tgt', 'similarity', 'label', 'edge'])
        else:
            return pd.read_csv(predictions_file)
    
    def determine_threshold(self, df: pd.DataFrame) -> float:
        """
        Determine threshold from ground truth labels.
        
        Uses quantile of positive label similarities.
        
        Args:
            df: DataFrame with predictions
        
        Returns:
            Determined threshold value
        """
        ground_truth_df = df[df['label'] == 1]
        
        if ground_truth_df.empty:
            raise ValueError("No positive labels found in the data")
        
        threshold = ground_truth_df['similarity'].quantile(self.quantile)
        print(f"Auto-determined threshold: {threshold:.6f} (quantile={self.quantile})")
        return threshold
    
    def filter(
        self,
        predictions_file: str,
        output_file: str,
        threshold: Optional[float] = None
    ) -> str:
        """
        Filter edges based on similarity threshold.
        
        Args:
            predictions_file: Path to predictions file (.pt or .csv)
            output_file: Output path for filtered edges (.csv)
            threshold: Override threshold (default: use instance threshold or auto)
        
        Returns:
            Path to the output file
        """
        # Load predictions
        print(f"Loading predictions from {predictions_file}")
        df = self.load_predictions(predictions_file)
        print(f"Loaded {len(df):,} predictions")
        
        # Determine threshold
        if threshold is not None:
            use_threshold = threshold
        elif self.threshold is not None:
            use_threshold = self.threshold
        else:
            use_threshold = self.determine_threshold(df)
        
        # Filter edges
        filtered_df = df[df['similarity'] >= use_threshold].copy()
        filtered_df['edge'] = 1
        
        # Rename columns for graph building compatibility
        if 'src' not in filtered_df.columns:
            filtered_df = filtered_df.rename(columns={
                filtered_df.columns[0]: 'src',
                filtered_df.columns[1]: 'tgt'
            })
        
        print(f"Filtered {len(filtered_df):,} edges from {len(df):,} predictions (threshold={use_threshold:.4f})")
        
        # Generate output path with threshold if needed
        if '{threshold}' in output_file:
            output_file = output_file.replace('{threshold}', f'{use_threshold:.2f}')
        elif not output_file.endswith('.csv'):
            base = output_file
            output_file = f"{base}_threshold_{use_threshold:.2f}.csv"
        
        # Save filtered edges
        os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
        filtered_df.to_csv(output_file, index=False)
        print(f"✅ Saved {len(filtered_df):,} filtered edges to {output_file}")
        
        return output_file
    
    def filter_at_multiple_thresholds(
        self,
        predictions_file: str,
        output_dir: str,
        thresholds: list
    ) -> dict:
        """
        Filter edges at multiple thresholds.
        
        Useful for creating graphs at different similarity levels.
        
        Args:
            predictions_file: Path to predictions file
            output_dir: Output directory for filtered edge files
            thresholds: List of thresholds to filter at
        
        Returns:
            Dict mapping threshold to output file path
        """
        os.makedirs(output_dir, exist_ok=True)
        results = {}
        
        # Load once
        df = self.load_predictions(predictions_file)
        
        for threshold in thresholds:
            output_file = os.path.join(output_dir, f"filtered_edges_threshold_{threshold:.2f}.csv")
            filtered_df = df[df['similarity'] >= threshold].copy()
            filtered_df['edge'] = 1
            filtered_df.to_csv(output_file, index=False)
            results[threshold] = output_file
            print(f"Threshold {threshold:.2f}: {len(filtered_df):,} edges → {output_file}")
        
        return results
    
    @classmethod
    def from_config(cls, config: "PreprocessConfig") -> "EdgeFilter":
        """Create an EdgeFilter from a PreprocessConfig."""
        from .config import PreprocessConfig
        return cls(threshold=config.similarity_threshold)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Filter edges by similarity threshold")
    parser.add_argument("--predictions", type=str, required=True, help="Path to predictions file")
    parser.add_argument("--output", type=str, required=True, help="Output path for filtered edges")
    parser.add_argument("--threshold", type=float, default=None, help="Similarity threshold (auto if not set)")
    parser.add_argument("--quantile", type=float, default=0.15, help="Quantile for auto-threshold")
    
    args = parser.parse_args()
    
    filter = EdgeFilter(threshold=args.threshold, quantile=args.quantile)
    filter.filter(
        predictions_file=args.predictions,
        output_file=args.output
    )
