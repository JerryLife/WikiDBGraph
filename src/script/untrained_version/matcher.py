import os
import csv
import time
import random
import argparse
import itertools
import traceback
import pandas as pd
from tqdm import tqdm
from typing import List, Tuple, Dict
from model.WKDataset import WKDataset
from model.col_embedding_model import EmbeddingMatcher
from model.BGEEmbedder import BGEEmbedder

def match_databases_by_embedding(
    db_id_1: str,
    db_id_2: str,
    wk_dataset: WKDataset,
    matcher: EmbeddingMatcher,
    flatten_columns: bool = False
) -> Dict[Tuple[str, str], float]:
    """
    Match columns between two databases using embedding similarity.

    Args:
        db_id_1 (str): ID of the first database.
        db_id_2 (str): ID of the second database.
        wk_dataset (WKDataset): The dataset loader.
        matcher (EmbeddingMatcher): The embedding-based matcher.

    Returns:
        Dict[Tuple[str, str], float]: A dictionary where keys are column pairs 
        (from db1 and db2), and values are similarity scores.
    """
    db_df1 = wk_dataset.load_csv_data(db_id_1, flatten_columns=flatten_columns, using_database_title=False)
    db_df2 = wk_dataset.load_csv_data(db_id_2, flatten_columns=flatten_columns, using_database_title=False)

    # Merge all tables into one wide DataFrame (by columns)
    df1_merged = pd.concat(
        [df.reset_index(drop=True) for df in db_df1.values() if not df.empty],
        axis=1
    )
    df2_merged = pd.concat(
        [df.reset_index(drop=True) for df in db_df2.values() if not df.empty],
        axis=1
    )
    df1_merged = df1_merged.loc[:, ~df1_merged.columns.duplicated()]
    df2_merged = df2_merged.loc[:, ~df2_merged.columns.duplicated()]

    # Perform embedding-based column matching
    results = matcher.get_embedding_similarity_candidates(df1_merged, df2_merged)

    return results

def run_col_matcher(db_id_1, db_id_2, method, level, model_path):
    wk = WKDataset(schema_dir="../data/schema", csv_base_dir="../data/unzip")
    # modal_name = method
    if method.startswith("bge"):
        if method.split("_")[-1] == "ft":
            embedder = BGEEmbedder(model_type=method.split("_")[0], model_path=model_path)
        else:
            embedder = BGEEmbedder(model_type=method)
        show_wikidata_property_id = False
        if level.endswith("wkpid"):
            show_wikidata_property_id = True
        
        max_sim_val, matched_pair, elapsed = embedder.database_similarity(wk, db_id_1, db_id_2, show_wikidata_property_id=show_wikidata_property_id)
        return max_sim_val, matched_pair, elapsed
        # elif level == "column":
        #     model_name = "BAAI/bge-m3"

    # is_idf_weighted = False
    # similarity_function = "cosine"
    # encoding_mode = "table_column_values"

    # if method.endswith("_idf"):
    #     is_idf_weighted = True

    # matcher_params = {
    #     "similarity_function": similarity_function,
    #     "encoding_mode": encoding_mode,
    #     "is_idf_weighted": is_idf_weighted,
    #     "embedding_model": model_name,
    #     "embedding_threshold": 0.1,
    #     "topk": 1,
    #     "sampling_mode": "priority_sampling",
    #     "sampling_size": 3,
    # }
    # matcher = EmbeddingMatcher(matcher_params)
    # start_time = time.time()
    # max_sim_val, matched_pair, total_column_count = match_databases_by_embedding(db_id_1, db_id_2, wk, matcher, flatten_columns=True)
    # elapsed = time.time() - start_time

    # with open(save_path, "a", encoding="utf-8") as f:
    #     f.write(f"Total columns: {total_column_count} \n")
    #     f.write(f"Matched pair: {matched_pair} \n")
    #     f.write(f"Max similarity: {max_sim_val:.4f} \n")
    #     f.write(f"Runtime: {elapsed:.2f} seconds\n")

    return max_sim_val, matched_pair, elapsed

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
        writer.writerow(["db_1", "db_2"])
        for db1, db2 in pairs:
            writer.writerow([f"{db1:05d}", f"{db2:05d}"])  # 补0为5位字符串

    print(f"Saved {len(pairs)} random pairs to {output_csv}")

# idf_cache = {}

# def get_idf(idf_map, col):
#     if col not in idf_cache:
#         col_name = col.split("::")[1].lower()
#         idf_cache[col] = idf_map.get(col_name, 1.0)
#     return idf_cache[col]

# def compute_max_weighted_similarity(results, idf_map):
#     data = [
#         {"col1": k[0], "col2": k[1], "score": v}
#         for k, v in results.items()
#     ]
#     df = pd.DataFrame(data)

#     df["col1_name"] = df["col1"].apply(lambda x: x.split("::")[1].lower())
#     df["col2_name"] = df["col2"].apply(lambda x: x.split("::")[1].lower())

#     df["idf1"] = df["col1_name"].map(idf_map).fillna(1.0)
#     df["idf2"] = df["col2_name"].map(idf_map).fillna(1.0)

#     df["weighted_score"] = df["score"] * df[["idf1", "idf2"]].min(axis=1)

#     max_score = df["weighted_score"].max()

#     return max_score, df

def load_pairs_from_csv(mode, csv_path):
    df = pd.read_csv(csv_path)
    pairs = []

    if mode in ["qid", "sample_10k", "neg_sample"]:
        for _, row in df.iterrows():
            db1 = str(row["db_1"]).zfill(5)
            db2 = str(row["db_2"]).zfill(5)
            qid = row["qid"] if "qid" in row else "random"
            pairs.append((db1, db2, qid))
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    return pairs


def load_completed_pairs(csv_path):
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        return set()
    df = pd.read_csv(csv_path)
    return set(
        (str(row["db_1"]).zfill(5), str(row["db_2"]).zfill(5))
        for _, row in df.iterrows()
    )

def run_match_loop(pairs, result_csv_path, method, level, model_path):
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
                max_sim, matched_pair, elapsed = run_col_matcher(db1, db2, method, level, model_path)
                col1, col2 = matched_pair
                writer.writerow([db1, db2, col1, col2, round(max_sim, 4), round(elapsed, 2)])
                csvfile.flush()
                edge_count += 1
            except Exception as e:
                print(f"Error in {db1} <--> {db2}: {e}")
                traceback.print_exc()

    elapsed_total = time.time() - start_time
    print(f"Total pairs attempted: {edge_count}")
    print(f"Elapsed time: {elapsed_total:.2f} seconds")
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, required=True, help="Choose matching method")
    parser.add_argument("--dataset", choices=["qid", "sample_10k", "neg_sample"], required=True, help="Choose input format")
    parser.add_argument("--level", choices=["database-wkpid", "database", "table", "column"], required=True, help="Choose matching level")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for generating sample pairs")
    parser.add_argument("--model-path", type=str, default=None, help="Path to the model file")
    args = parser.parse_args()

    out_dir = f"/hpctmp/e1351271/wkdbs/out/col_matcher_{args.method}_{args.level}"
    os.makedirs(out_dir, exist_ok=True)
    if args.dataset == "qid":
        output_csv = os.path.join(out_dir, f"col_matcher_{args.dataset}_{args.method}.csv")
    else:
        output_csv = os.path.join(out_dir, f"col_matcher_{args.dataset}_{args.method}_seed{args.seed}.csv")

    dataset_to_input = {
        "qid": "/hpctmp/e1351271/wkdbs/data/qid_pairs.csv",
        "sample_10k": f"/hpctmp/e1351271/wkdbs/data/sample_pairs_10k_seed{args.seed}.csv",
        "neg_sample": f"/hpctmp/e1351271/wkdbs/data/neg_samples/neg_seed{args.seed}.csv"
    }
    input_path = dataset_to_input[args.dataset]

    if args.dataset == "sample_10k" and not os.path.exists(input_path):
        generate_random_pairs(seed=args.seed, num_pairs=10000, output_csv=input_path)

    pairs = load_pairs_from_csv(args.dataset, input_path)

    run_match_loop(pairs, output_csv, args.method, args.level, args.model_path)

if __name__ == "__main__":
    main()