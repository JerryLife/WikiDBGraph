#!/usr/bin/env python3
"""Find shared label columns with suitable distributions across all DBs."""
import os
import pandas as pd
from collections import Counter

DB_IDS = ["03001", "06604", "36035", "39064", "53148", "86018", "91688"]
CSV_DIR = "data/unzip"

# Columns to check for label potential
CANDIDATE_COLS = [
    "cellular_component", "cellularcomponent", "cellular_location",
    "biological_process", "biologicalprocess", "process_type",
    "molecular_function", "molecularfunction", "function_type",
    "protein_subclass", "proteinsubclass", "entity_type", "instance_type"
]

def find_csv_dir(db_id):
    for name in os.listdir(CSV_DIR):
        if name.startswith(db_id):
            return os.path.join(CSV_DIR, name)
    return None

results = {}
for db_id in DB_IDS:
    csv_dir = find_csv_dir(db_id)
    if not csv_dir:
        print(f"DB {db_id}: No CSV directory found")
        continue
    
    print(f"\n=== DB {db_id} ({os.path.basename(csv_dir)}) ===")
    results[db_id] = {}
    
    for csv_file in os.listdir(csv_dir):
        if not csv_file.endswith('.csv'):
            continue
        df = pd.read_csv(os.path.join(csv_dir, csv_file), low_memory=False)
        for col in df.columns:
            col_lower = col.lower().replace("_", "")
            for cand in CANDIDATE_COLS:
                if cand.replace("_", "") in col_lower:
                    vals = df[col].dropna()
                    if len(vals) < 10:
                        continue
                    counts = Counter(vals)
                    # Support multi-class: 2-10 unique values
                    if len(counts) >= 2 and len(counts) <= 10:
                        total = sum(counts.values())
                        print(f"  {csv_file}.{col}: {len(vals)} rows, {len(counts)} classes")
                        for val, cnt in counts.most_common(5):
                            print(f"    '{val}': {cnt} ({100*cnt/total:.1f}%)")
                        results[db_id][col] = {"file": csv_file, "counts": dict(counts), "num_classes": len(counts)}

