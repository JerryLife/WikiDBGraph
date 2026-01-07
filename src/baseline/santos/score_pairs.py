
import os
import json
import pickle
import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from collections import defaultdict
import sys

# Add src to path
sys.path.append(str(Path(__file__).resolve().parents[3]))
from src.baseline.santos import config
from src.baseline.santos.synthesize_kb import preprocess_string, check_if_null_string, get_column_type, preprocess_list_values

# --- Scoring Logic ---

def load_indices():
    print("Loading indices...")
    with open(config.SYNTH_TYPE_KB_PATH, 'rb') as f:
        synth_type_kb = pickle.load(f)
    print("Loaded Type KB")
    
    with open(config.SYNTH_RELATION_KB_PATH, 'rb') as f:
        synth_relation_kb = pickle.load(f)
    print("Loaded Relation KB")

    with open(config.INDEX_DIR / "synth_type_inverted_index.pkl", 'rb') as f:
        type_inverted = pickle.load(f)
    print("Loaded Type Inverted Index")
    
    with open(config.SYNTH_RELATION_INVERTED_INDEX_PATH, 'rb') as f:
        relation_inverted = pickle.load(f)
    print("Loaded Relation Inverted Index")
    
    return synth_type_kb, synth_relation_kb, type_inverted, relation_inverted

def compute_synth_column_semantics(input_table, synth_type_kb):
    all_column_semantics = {}
    col_id = 0
    for col_name in input_table.columns:
        col_data = input_table[col_name].tolist()
        if get_column_type(col_data) == 1:
            input_table[col_name] = input_table[col_name].astype(str)
            value_list = preprocess_list_values(input_table[col_name].unique())
            hit_found = 0
            sem = defaultdict(float)
            
            for value in value_list:
                if value in synth_type_kb:
                    hit_found += 1
                    # Item is list of tuples (semName, score)
                    for semName, semScore in synth_type_kb[value]:
                         sem[semName] += semScore
            
            if hit_found > 0:
                for s in sem:
                    sem[s] /= hit_found
                    
            all_column_semantics[str(col_id)] = dict(sem)
        col_id += 1
    return all_column_semantics

def compute_synth_relation(input_table, synth_relation_kb, subject_index=None):
    # Santos logic: finds relationships between columns
    # Returns: relationSemantics, synth_triple_dict, subject_semantics
    
    synth_triple_dict = {}
    subject_semantics = set()
    total_cols = input_table.shape[1]
    
    # Identify text cols
    text_cols = [i for i in range(total_cols) if get_column_type(input_table.iloc[:, i].tolist()) == 1]
    
    for idx_i in range(len(text_cols)):
        i = text_cols[idx_i]
        for idx_j in range(idx_i + 1, len(text_cols)):
            j = text_cols[idx_j]
            
            merge_rel_sem = defaultdict(float)
            pair_df = input_table.iloc[:, [i, j]].dropna().drop_duplicates()
            rows = pair_df.shape[0]
            if rows == 0: continue
            
            for row in pair_df.itertuples(index=False):
                sub = preprocess_string(str(row[0]))
                obj = preprocess_string(str(row[1]))
                
                if check_if_null_string(sub) and check_if_null_string(obj):
                    val = f"{sub}__{obj}"
                    item = []
                    if val in synth_relation_kb:
                        item = synth_relation_kb[val]
                    else:
                        # Try reverse? (SANTOS does check reverse)
                        val_rev = f"{obj}__{sub}"
                        if val_rev in synth_relation_kb:
                            item = synth_relation_kb[val_rev]
                            
                    if item:
                        for semName, semScore in item:
                            if semScore > 0:
                                merge_rel_sem[semName] += (semScore / rows)
                                
            # Collect triples
            # synth_triple_dict key is "i-j"
            triple_list = [(k, v) for k, v in merge_rel_sem.items()]
            synth_triple_dict[f"{i}-{j}"] = triple_list
            
            # Subject semantics logic (not strictly needed for pure pairwise score but good for intent)
            if subject_index is not None and (int(subject_index) == i or int(subject_index) == j):
                for sem in merge_rel_sem:
                    subject_semantics.add(sem)
                    
    return synth_triple_dict, subject_semantics

def score_pair(query_triples, query_cols, candidate_name, relation_inverted, type_inverted, synthetic_cols):
    # Calculate score intersection following SANTOS scoring logic
    # query_triples: { "i-j": [(relName, score), ...] }
    
    total_score = 0.0
    already_used_column = {}  # Deduplication per (candidate, col_pairs)
    
    # Helper to get max single-key product (per SANTOS original)
    def dict_max_key(d1, d2):
        common = d1.keys() & d2.keys()
        if not common:
            return 0.0
        return max(d1[k] * d2[k] for k in common)
    
    # Iterate over query triples
    for col_pair, triples in query_triples.items():
        q_col1, q_col2 = col_pair.split("-")
        
        for rel_name, rel_score in triples:
            # Look up this relation semantic in inverted index
            if rel_name in relation_inverted:
                # relation_inverted[rel_name] = { TableName: (Score, Col1, Col2) }
                if candidate_name in relation_inverted[rel_name]:
                    cand_info = relation_inverted[rel_name][candidate_name]
                    cand_score = cand_info[0]
                    cand_col1 = cand_info[1]
                    cand_col2 = cand_info[2]
                    
                    # Fetch semantics
                    q_sem1 = synthetic_cols.get(q_col1, {})
                    q_sem2 = synthetic_cols.get(q_col2, {})
                    
                    c_sem1 = type_inverted.get((candidate_name, cand_col1), {})
                    c_sem2 = type_inverted.get((candidate_name, cand_col2), {})
                    
                    # Calculate all 4 direction max scores
                    max_scores = [
                        dict_max_key(q_sem1, c_sem1),
                        dict_max_key(q_sem1, c_sem2),
                        dict_max_key(q_sem2, c_sem1),
                        dict_max_key(q_sem2, c_sem2)
                    ]
                    max_scores.sort(reverse=True)
                    # SANTOS uses top-2 product
                    match_score = max_scores[0] * max_scores[1]
                    
                    # Relation contribution
                    triple_score = rel_score * cand_score * match_score
                    
                    # Deduplication logic (from SANTOS)
                    dedup_key = (candidate_name, col_pair)
                    if dedup_key not in already_used_column:
                        total_score += triple_score
                        already_used_column[dedup_key] = triple_score
                    else:
                        if triple_score > already_used_column[dedup_key]:
                            total_score -= already_used_column[dedup_key]
                            total_score += triple_score
                            already_used_column[dedup_key] = triple_score
                    
    return total_score

def index_databases():
    # Map DB_ID -> Directory
    # Assumes format "ID Name" or just "ID"
    # Returns map: ID -> Path
    print("Indexing databases...")
    db_map = {}
    for p in config.UNZIP_DIR.glob("*"):
        if p.is_dir():
            # Extract ID usually first token?
            # Existing check showed "00000 Name"
            # Try to match numeric IDs or just basename
            parts = p.name.split(" ")
            if parts:
                db_id = parts[0]
                db_map[db_id] = p
                # Also handle just basename if needed or stripped
                db_map[p.name] = p
    print(f"Indexed {len(db_map)} databases.")
    return db_map

def get_tables(db_id, db_map):
    if db_id not in db_map:
        return []
    path = db_map[db_id]
    # Check tables dir
    tables_dir = path / "tables"
    if tables_dir.exists():
        return list(tables_dir.glob("*.csv"))
    # Fallback: maybe csvs in root?
    return list(path.glob("*.csv"))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--triplets", type=str, required=True, help="Input triplets JSONL")
    parser.add_argument("--output", type=str, required=True, help="Output scores CSV")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of pairs")
    args = parser.parse_args()
    
    # Load indices
    synth_type_kb, synth_relation_kb, type_inverted, relation_inverted = load_indices()
    db_map = index_databases()
    
    results = []
    
    processed_count = 0
    with open(args.triplets, "r") as f:
        for line in f:
            if args.limit and processed_count >= args.limit:
                break
                
            triplet = json.loads(line)
            anchor_id = triplet["anchor"]
            pos_id = triplet["positive"]
            neg_ids = triplet["negatives"]
            
            # --- Scoring Logic (Max Score) ---
            
            def score_db_pair(id1, id2):
                tables1 = get_tables(id1, db_map)
                tables2 = get_tables(id2, db_map)
                
                if not tables1 or not tables2:
                    return 0.0
                    
                # Cache query tables representations?
                # Optimization: For anchor, we reuse it across negatives.
                # So we return representation of tables1.
                return 0.0 # Placeholder for structure, logic below
            
            # Process Anchor
            anchor_tables = get_tables(anchor_id, db_map)
            if not anchor_tables:
                print(f"Warning: No tables for anchor {anchor_id}")
                continue
                
            # Precompute Anchor Representations
            anchor_reps = []
            for t_path in anchor_tables:
                try:
                    df = pd.read_csv(t_path, encoding='latin1', on_bad_lines='skip')
                    cols = compute_synth_column_semantics(df, synth_type_kb)
                    rels, sub_sem = compute_synth_relation(df, synth_relation_kb, subject_index=None)
                    anchor_reps.append({
                        "name": t_path.name,
                        "cols": cols,
                        "rels": rels,
                        "sub": sub_sem
                    })
                except Exception as e:
                    pass
            
            def get_db_score(candidate_id):
                cand_tables = get_tables(candidate_id, db_map)
                if not cand_tables:
                    return 0.0
                    
                max_score = 0.0
                for c_path in cand_tables:
                    # Construct key as DB_ID/TableName to match synthesized index
                    c_name = f"{candidate_id}/{c_path.name}"
                    
                    for anchor_rep in anchor_reps:
                        # Score pair (anchor_table, candidate_table)
                        score = score_pair(anchor_rep["rels"], anchor_rep["cols"], c_name, relation_inverted, type_inverted, anchor_rep["cols"])
                        if score > max_score:
                            max_score = score
                return max_score

            # Score Positive
            pos_score = get_db_score(pos_id)
            results.append({"db1": anchor_id, "db2": pos_id, "label": 1, "score": pos_score})
            
            # Score Negatives
            for neg_id in neg_ids:
                neg_score = get_db_score(neg_id)
                results.append({"db1": anchor_id, "db2": neg_id, "label": 0, "score": neg_score})
                    
            processed_count += 1
            if processed_count % 10 == 0:
                print(f"Processed {processed_count} triplets")
                
    # Save
    pd.DataFrame(results).to_csv(args.output, index=False)
    print(f"Saved scores to {args.output}")

if __name__ == "__main__":
    main()
