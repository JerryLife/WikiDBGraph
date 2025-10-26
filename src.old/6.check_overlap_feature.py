import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from tqdm import tqdm
import numpy as np

# Define paths
CSV_DIR = "out"
SCHEMA_DIR = "data/schema"
LOG_DIR = "logs"
PLOT_DIR = "plots"

os.makedirs(LOG_DIR, exist_ok=True)  # Ensure log directory exists
os.makedirs(PLOT_DIR, exist_ok=True)  # Ensure plot directory exists

# Get all CSV files
csv_files = [f for f in os.listdir(CSV_DIR) if f.startswith("pairwise_stats_seed") and f.endswith(".csv")]

# Process each CSV file individually
for csv_file in tqdm(csv_files, desc="Processing CSV files"):
    file_path = os.path.join(CSV_DIR, csv_file)
    df = pd.read_csv(file_path)

    # Extract unique database names from db1 and db2
    relevant_db_names = set(df["db1"]).union(set(df["db2"]))

    # Load only the necessary schema files
    schema_data = {}
    for db_name in tqdm(relevant_db_names, desc=f"Loading schemas for {csv_file}", leave=False):
        schema_files = [f for f in os.listdir(SCHEMA_DIR) if db_name in f and f.endswith(".json")]

        for schema_file in schema_files:
            schema_path = os.path.join(SCHEMA_DIR, schema_file)
            try:
                with open(schema_path, "r", encoding="utf-8") as f:
                    schema = json.load(f)

                columns = set()
                for table in schema.get("tables", []):
                    for column in table.get("columns", []):
                        columns.add(column["column_name"])
                        columns.update(column.get("alternative_column_names", []))

                schema_data[db_name] = columns
            except json.JSONDecodeError:
                print(f"⚠️ Failed to parse {schema_file}")

    # Count overlapping feature occurrences
    feature_overlap_counts = Counter()

    # Process each row in the CSV file
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {csv_file}", leave=False):
        db1, db2, overlap_features = row["db1"], row["db2"], row["overlap_features"]

        if overlap_features > 0 and db1 in schema_data and db2 in schema_data:
            common_features = schema_data[db1].intersection(schema_data[db2])
            feature_overlap_counts.update(common_features)

    # Save results to log file (top 10 features with names)
    log_file_path = os.path.join(LOG_DIR, f"{csv_file.replace('.csv', '.log')}")
    with open(log_file_path, "w", encoding="utf-8") as log_file:
        log_file.write("Top 10 Most Frequent Overlapping Features:\n")
        for feature, count in feature_overlap_counts.most_common(20):
            log_file.write(f"{feature}: {count}\n")

    print(f"✅ Results saved to {log_file_path}")

    # Generate and save broken axis plot (first half + last 1000 features)
    if feature_overlap_counts:
        sorted_features = sorted(feature_overlap_counts.items(), key=lambda x: x[1], reverse=True)
        feature_ids = np.arange(len(sorted_features))  # Assign index to each feature
        counts = np.array([count for _, count in sorted_features])

        total_features = len(feature_ids)
        split_point = 4000  # Middle index
        tail_size = 100  # Last part to keep

        # Define index ranges: First half (0–split_point) + last tail_size
        if total_features > tail_size:
            first_half_indices = feature_ids[:split_point]
            first_half_counts = counts[:split_point]

            last_part_indices = feature_ids[-tail_size:]
            last_part_counts = counts[-tail_size:]
        else:
            first_half_indices = feature_ids
            first_half_counts = counts
            last_part_indices = []
            last_part_counts = []

        # Determine dynamic Y-axis max (based on highest occurrence count)
        ymax = max(counts) * 1.1  # Add 10% padding

        # Create two subplots (shared Y-axis)
        fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(12, 6), dpi=100, gridspec_kw={'width_ratios': [50, 1]})

        # Plot first half
        ax1.fill_between(first_half_indices, first_half_counts, color="skyblue", alpha=0.6)
        ax1.set_xlim(min(first_half_indices), max(first_half_indices))
        ax1.set_xlabel("Feature ID (Sorted)")

        ax2.fill_between(last_part_indices, last_part_counts, color="skyblue", alpha=0.6)
        ax2.set_xlim(min(last_part_indices), max(last_part_indices))
        ax2.set_xlabel("Feature ID (Sorted)")
        
        # Shared settings
        ax1.set_ylabel("Occurrence Count")
        ax1.set_ylim(0, ymax)
        fig.suptitle(f"Feature Overlap Distribution - {csv_file}")
        ax1.grid(axis="y", linestyle="--", alpha=0.7)
        ax2.grid(axis="y", linestyle="--", alpha=0.7)

        # Hide spines to show broken axis
        ax1.spines['right'].set_visible(False)
        ax2.spines['left'].set_visible(False)

        # Add broken axis markers
        d = 0.85  # Slant angle
        kwargs = dict(marker=[(-1, -d), (1, d)], markersize=15,
                    linestyle='none', color='r', mec='r', mew=1, clip_on=False)

        ax1.plot([1, 1], [1, 0], transform=ax1.transAxes, **kwargs)
        ax2.plot([0, 0], [0, 1], transform=ax2.transAxes, **kwargs)

        # Save the plot
        plot_file_path = os.path.join(PLOT_DIR, f"{csv_file.replace('.csv', '.png')}")
        plt.savefig(plot_file_path, bbox_inches="tight")
        plt.close()

        print(f"📊 Plot saved to {plot_file_path}")
