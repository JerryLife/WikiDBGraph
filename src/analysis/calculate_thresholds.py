"""
Calculate similarity thresholds for positive edges.

This script mirrors the logic in `filter_edges.py` but only reports the
similarity thresholds instead of writing filtered edge files.
"""

import argparse
import pandas as pd
import torch


def load_all_predictions(input_file: str) -> pd.DataFrame:
    """Load the prediction tensor into a DataFrame."""
    print(f"Loading predictions from {input_file}...")
    arr = torch.load(input_file, weights_only=False).cpu().numpy()
    df = pd.DataFrame(arr, columns=["src", "tgt", "similarity", "label", "edge"])
    print(f"Loaded {len(df):,} predictions")
    return df


def calculate_positive_thresholds(df: pd.DataFrame, quantiles=None):
    """
    Calculate similarity thresholds for the positive labels at the requested quantiles.
    """
    quantiles = quantiles or [0.05, 0.10, 0.15, 0.20]
    positives = df[df["label"] == 1]

    if positives.empty:
        raise ValueError("No positive labels found in the data.")

    thresholds = {}
    for q in quantiles:
        threshold = positives["similarity"].quantile(q)
        thresholds[q] = threshold
        print(f"Quantile {q:.2f}: threshold = {threshold:.10f}")

    return thresholds


def main(input_file: str):
    df = load_all_predictions(input_file)
    calculate_positive_thresholds(df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate similarity thresholds for positive edges."
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        default="data/data/out/graph/all_exhaustive_predictions.pt",
        help="Path to input predictions file.",
    )
    args = parser.parse_args()

    main(args.input)
