import os
import csv
import time
import random
import argparse
import itertools
import pandas as pd
from tqdm import tqdm
from typing import List, Tuple, Dict
from model.WKDataset import WKDataset
from model.col_embedding_model import EmbeddingMatcher

out_dir = "/hpctmp/e1351271/wkdbs/out/col_matcher_cross_encoder"
os.makedirs(out_dir, exist_ok=True)

def run_col_matcher(db_id_1, db_id_2, save_path):
    wk = WKDataset(schema_dir="../data/schema", csv_base_dir="../data/unzip")
    matcher_params = {
        "embedding_model": "sentence-transformers/all-mpnet-base-v2",
        "embedding_threshold": 0.1,
        "topk": 1,
        "encoding_mode": "table_header_values_repeat",
        "sampling_mode": "priority_sampling",
        "sampling_size": 5
    }
    matcher = EmbeddingMatcher(matcher_params)
    start_time = time.time()
    max_sim_val, matched_pair, total_column_count = match_databases_by_embedding(db_id_1, db_id_2, wk, matcher, flatten_columns=True)
    # max_sim_val, matched_pair, total_column_count = 0, 0, 0
    elapsed = time.time() - start_time

    with open(save_path, "a", encoding="utf-8") as f:
        f.write(f"Total columns: {total_column_count} \n")
        f.write(f"Matched pair: {matched_pair} \n")
        f.write(f"Max similarity: {max_sim_val:.4f} \n")
        f.write(f"Runtime: {elapsed:.2f} seconds\n")

    return total_column_count, max_sim_val, elapsed

def generate_random_pairs(seed: int, num_pairs: int, output_csv: str):
    all_ids = list(range(100000))  # 00000 - 99999
    random.seed(seed)

    seen = set()
    pairs = []

    while len(pairs) < num_pairs:
        db1, db2 = random.sample(all_ids, 2)
        pair = (min(db1, db2), max(db1, db2))
        if pair not in seen:
            seen.add(pair)
            pairs.append(pair)

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["db1", "db2"])
        for db1, db2 in pairs:
            writer.writerow([f"{db1:05d}", f"{db2:05d}"])  # 补0为5位字符串

    print(f"Saved {len(pairs)} random pairs to {output_csv}")

def load_pairs_from_csv(mode, csv_path):
    df = pd.read_csv(csv_path)
    pairs = []

    if mode in ["qid", "sample_10k"]:
        for _, row in df.iterrows():
            db1 = str(row["db1"]).zfill(5)
            db2 = str(row["db2"]).zfill(5)
            qid = row["qid"] if "qid" in row else "random"
            pairs.append((db1, db2, qid))
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    return pairs


def load_completed_pairs(csv_path):
    if not os.path.exists(csv_path):
        return set()
    df = pd.read_csv(csv_path)
    return set(
        (str(row["db1"]).zfill(5), str(row["db2"]).zfill(5))
        for _, row in df.iterrows()
    )

def run_match_loop(pairs, result_csv_path):
    edge_count = 0
    start_time = time.time()

    completed = load_completed_pairs(result_csv_path)
    write_header = not os.path.exists(result_csv_path)
    filtered_pairs = [p for p in pairs if (p[0], p[1]) not in completed]

    with open(result_csv_path, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        if write_header:
            writer.writerow(["db_1", "db_2", "column_1", "column_2", "similarity", "runtime_seconds"])

        for db1, db2, _ in tqdm(filtered_pairs, desc="Matching", ncols=100):
            if (db1, db2) in completed or (db2, db1) in completed:
                continue
            try:
                matched_pair, max_sim, elapsed = run_col_matcher(db1, db2, None)
                col1, col2 = matched_pair
                writer.writerow([db1, db2, col1, col2, round(max_sim, 4), round(elapsed, 2)])
                csvfile.flush()
                edge_count += 1
            except Exception as e:
                print(f"Error in {db1} <--> {db2}: {e}")

    elapsed_total = time.time() - start_time
    print(f"Total pairs attempted: {edge_count}")
    print(f"Elapsed time: {elapsed_total:.2f} seconds")
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["qid", "sample_10k"], required=True, help="Choose input format")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for generating sample pairs")
    args = parser.parse_args()

    mode_to_input = {
        "qid": "/hpctmp/e1351271/wkdbs/data/duplicate_qids.csv",
        "sample_10k": f"/hpctmp/e1351271/wkdbs/data/sample_pairs_10k_seed{args.seed}.csv"
    }
    input_path = mode_to_input[args.mode]

    if args.mode == "sample_10k" and not os.path.exists(input_path):
        generate_random_pairs(seed=args.seed, num_pairs=10000, output_csv=input_path)

    pairs = load_pairs_from_csv(args.mode, input_path)

    output_csv = os.path.join(out_dir, f"col_matcher_{args.mode}.csv")
    run_match_loop(pairs, output_csv)

if __name__ == "__main__":
    main()