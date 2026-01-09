
import os
import glob
import time
import argparse
import polars as pl
import numpy as np
import pickle
import tempfile
import shutil
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from collections import defaultdict
from pathlib import Path
import re
import sys
from tqdm import tqdm

# Add src to path to import config
sys.path.append(str(Path(__file__).resolve().parents[3]))
from src.baseline.santos import config

# Global executor class (set by main)
ExecutorClass = ThreadPoolExecutor

# Temp directory for disk-based KB building (set by main)
TEMP_DIR = None

# --- Helper Functions (Re-implemented from generalFunctions.py) ---

def check_if_null_string(s):
    if s is None:
        return 0
    null_list = {'nan', '-', 'unknown', 'other (unknown)', 'null', 'na', 'none', '', ' '}
    return 0 if str(s).lower() in null_list else 1

def preprocess_string(s):
    if s is None:
        return ""
    s = str(s).lower()
    s = re.sub(r'[^\w]', ' ', s)
    s = s.replace("nbsp", "")
    return " ".join(s.split())

def get_column_type(attribute, column_threshold=0.5, entity_threshold=0.5):
    # Returns 1 for Text, 0 for Numeric/Other
    # Filter out None and "nan"
    attribute = [item for item in attribute if item is not None and str(item).lower() != "nan"]
    if not attribute:
        return 0
    
    str_attribute = [str(item) for item in attribute]
    # Filter out pure digits
    str_att_candidates = [item for item in str_attribute if not item.isdigit()]
    
    valid_text_count = 0
    for entity in str_att_candidates:
        digit_count = sum(c.isdigit() for c in entity)
        if len(entity) > 0 and (digit_count / len(entity)) <= entity_threshold:
            valid_text_count += 1
            
    # If ratio of "valid text strings" to "total non-null values" is high enough
    if len(attribute) > 0 and (valid_text_count / len(attribute)) > column_threshold:
        return 1
    return 0

def preprocess_list_values(value_list):
    # Filter nulls and preprocess
    processed = []
    for x in value_list:
        if check_if_null_string(x):
            processed.append(preprocess_string(x))
    return processed

def extract_db_id(file_path):
    """Extract database ID from path like 'data/unzip/00000 Name/tables/X.csv' -> '00000'"""
    parts = Path(file_path).parts
    # Find the database directory (format: "ID Name" or just "ID")
    for part in parts:
        # Check if starts with digits (the ID)
        if part and part.split(" ")[0].isdigit():
            return part.split(" ")[0]
    # Fallback: use parent directory name
    return Path(file_path).parent.parent.name.split(" ")[0]

def make_table_key(file_path):
    """Create a unique table key: 'DB_ID/TableName.csv'"""
    db_id = extract_db_id(file_path)
    table_name = Path(file_path).name
    return f"{db_id}/{table_name}"

def read_csv_polars(file_path):
    """Read CSV using polars for 2-5x speedup over pandas."""
    try:
        # We use try/except as some files might be corrupted or not valid CSVs
        df = pl.read_csv(file_path, encoding='latin1', ignore_errors=True, 
                         infer_schema_length=0, null_values=["", "NA", "null", "NaN", "nan"])
        return df
    except Exception:
        return None

# --- Worker Functions for Parallel Processing ---

def process_table_for_type_lookup(file_path, tab_id):
    df = read_csv_polars(file_path)
    if df is None:
        return None

    local_lookup = defaultdict(set)
    col_id = 0
    for col_name in df.columns:
        col_series = df[col_name]
        col_data_list = col_series.to_list()
        if get_column_type(col_data_list) == 1: # Text column
            unique_values = col_series.unique().to_list()
            # SANTOS logic: map(str) called in original
            unique_values = [str(x) for x in unique_values if x is not None]
            value_list = preprocess_list_values(unique_values)
            
            col_sem = f"c{tab_id}_{col_id}"
            for val in value_list:
                local_lookup[val].add(col_sem)
        col_id += 1
    return local_lookup


def process_chunk_for_type_lookup(chunk_data):
    """
    Process a chunk of files for Type Lookup Pass 1.
    Returns partial lookup table to be merged.
    """
    files_chunk, start_tab_id = chunk_data
    
    partial_lookup = defaultdict(set)
    
    for i, file_path in enumerate(files_chunk):
        tab_id = start_tab_id + i
        result = process_table_for_type_lookup(file_path, tab_id)
        if result:
            for val, col_ids in result.items():
                partial_lookup[val].update(col_ids)
    
    return partial_lookup


def process_chunk_for_type_kb(chunk_data):
    """
    Process a chunk of files for Type KB Pass 2.
    Returns partial results to be merged.
    """
    files_chunk, start_tab_id, filtered_lookup = chunk_data
    
    partial_kb = defaultdict(dict)
    partial_index = {}
    
    for i, file_path in enumerate(files_chunk):
        tab_id = start_tab_id + i
        
        df = read_csv_polars(file_path)
        if df is None:
            continue
            
        col_id = 0
        table_name = make_table_key(file_path)
        
        for col_name in df.columns:
            col_series = df[col_name]
            col_data_list = col_series.to_list()
            if get_column_type(col_data_list) == 1:
                # Text Column
                unique_values = [str(x) for x in col_series.unique().to_list() if x is not None]
                value_list = preprocess_list_values(unique_values)
                divide_by = len(value_list) if value_list else 1
                
                # Calculate semantic distribution for this column
                sem = defaultdict(float)
                for val in value_list:
                    if val in filtered_lookup:
                        for s in filtered_lookup[val]:
                            sem[s] += (1.0 / divide_by)
                
                # Assign to partial KB
                for val in value_list:
                    if val in filtered_lookup:
                        current_entry = partial_kb[val]
                        for s, score in sem.items():
                            current_entry[s] = max(current_entry.get(s, 0), score)
                            
                partial_index[(table_name, str(col_id))] = dict(sem)
            col_id += 1
    
    return partial_kb, partial_index


def process_chunk_for_type_kb_to_disk(chunk_data):
    """
    Process a chunk of files for Type KB Pass 2, writing to disk.
    Returns the path to the temp file instead of the dict.
    """
    files_chunk, start_tab_id, filtered_lookup, chunk_idx, temp_dir = chunk_data
    
    partial_kb = defaultdict(dict)
    partial_index = {}
    
    for i, file_path in enumerate(files_chunk):
        tab_id = start_tab_id + i
        
        df = read_csv_polars(file_path)
        if df is None:
            continue
            
        col_id = 0
        table_name = make_table_key(file_path)
        
        for col_name in df.columns:
            col_series = df[col_name]
            col_data_list = col_series.to_list()
            if get_column_type(col_data_list) == 1:
                # Text Column
                unique_values = [str(x) for x in col_series.unique().to_list() if x is not None]
                value_list = preprocess_list_values(unique_values)
                divide_by = len(value_list) if value_list else 1
                
                # Calculate semantic distribution for this column
                sem = defaultdict(float)
                for val in value_list:
                    if val in filtered_lookup:
                        for s in filtered_lookup[val]:
                            sem[s] += (1.0 / divide_by)
                
                # Assign to partial KB
                for val in value_list:
                    if val in filtered_lookup:
                        current_entry = partial_kb[val]
                        for s, score in sem.items():
                            current_entry[s] = max(current_entry.get(s, 0), score)
                            
                partial_index[(table_name, str(col_id))] = dict(sem)
            col_id += 1
    
    # Write to temp file
    temp_file = Path(temp_dir) / f"type_kb_chunk_{chunk_idx}.pkl"
    with open(temp_file, 'wb') as f:
        pickle.dump((dict(partial_kb), partial_index), f)
    
    # Return file path (lightweight)
    return str(temp_file)


def process_table_for_relation_lookup(file_path, tab_id):
    df = read_csv_polars(file_path)
    if df is None:
        return None
        
    local_lookup = defaultdict(set)
    total_cols = len(df.columns)
    
    # Identify text cols first
    text_cols = []
    for i in range(total_cols):
        col_data_list = df[:, i].to_list()
        if get_column_type(col_data_list) == 1:
            text_cols.append(i)
            
    # Pairs of text cols
    for idx_i in range(len(text_cols)):
        i = text_cols[idx_i]
        for idx_j in range(idx_i + 1, len(text_cols)):
            j = text_cols[idx_j]
            
            # Relation Semantic ID
            rel_sem = f"r{tab_id}_{i}_{j}"
            
            # Get pairs
            pair_df = df.select([df.columns[i], df.columns[j]]).drop_nulls().unique()
            
            for row in pair_df.iter_rows():
                sub = preprocess_string(row[0])
                obj = preprocess_string(row[1])
                
                if check_if_null_string(sub) and check_if_null_string(obj):
                    val = f"{sub}__{obj}"
                    local_lookup[val].add(rel_sem)
                    
    return local_lookup


def process_chunk_for_relation_lookup(chunk_data):
    """
    Process a chunk of files for Relation Lookup Pass 3.
    Returns partial lookup table to be merged.
    """
    files_chunk, start_tab_id = chunk_data
    
    partial_lookup = defaultdict(set)
    
    for i, file_path in enumerate(files_chunk):
        tab_id = start_tab_id + i
        result = process_table_for_relation_lookup(file_path, tab_id)
        if result:
            for val, rels in result.items():
                partial_lookup[val].update(rels)
    
    return partial_lookup


def process_chunk_for_relation_kb(chunk_data):
    """
    Process a chunk of files for Relation KB Pass 2.
    Returns partial results to be merged.
    """
    files_chunk, start_tab_id, lookup_table = chunk_data
    
    partial_kb = defaultdict(dict)
    partial_inverted = {}
    
    for i, file_path in enumerate(files_chunk):
        tab_id = start_tab_id + i
        
        df = read_csv_polars(file_path)
        if df is None:
            continue
            
        table_name = make_table_key(file_path)
        total_cols = len(df.columns)
        text_cols = []
        for c_idx in range(total_cols):
            if get_column_type(df[:, c_idx].to_list()) == 1:
                text_cols.append(c_idx)
        
        for idx_i in range(len(text_cols)):
            col_i_idx = text_cols[idx_i]
            for idx_j in range(idx_i + 1, len(text_cols)):
                col_j_idx = text_cols[idx_j]
                
                pair_df = df.select([df.columns[col_i_idx], df.columns[col_j_idx]]).drop_nulls().unique()
                rows = pair_df.height
                if rows == 0:
                    continue
                
                # Calculate Semantics for this pair of columns
                sem = defaultdict(float)
                
                pairs = []
                for row in pair_df.iter_rows():
                    sub = preprocess_string(row[0])
                    obj = preprocess_string(row[1])
                    if check_if_null_string(sub) and check_if_null_string(obj):
                        val = f"{sub}__{obj}"
                        if val in lookup_table:
                            for s in lookup_table[val]:
                                sem[s] += (1.0 / rows)
                        pairs.append(val)
                        
                # Assign to KB
                for val in pairs:
                    if val in lookup_table:
                        current_entry = partial_kb[val]
                        for s, score in sem.items():
                            current_entry[s] = max(current_entry.get(s, 0), score)

                # Inverted Index (Relation -> Table info)
                for s, score in sem.items():
                    if s not in partial_inverted:
                        partial_inverted[s] = {table_name: (score, str(col_i_idx), str(col_j_idx))}
                    else:
                        current_tables = partial_inverted[s]
                        if table_name in current_tables:
                            if score > current_tables[table_name][0]:
                                current_tables[table_name] = (score, str(col_i_idx), str(col_j_idx))
                        else:
                            current_tables[table_name] = (score, str(col_i_idx), str(col_j_idx))
    
    return partial_kb, partial_inverted


def process_chunk_for_relation_kb_to_disk(chunk_data):
    """
    Process a chunk of files for Relation KB Pass 4, writing to disk.
    Returns the path to the temp file instead of the dict.
    """
    files_chunk, start_tab_id, lookup_table, chunk_idx, temp_dir = chunk_data
    
    partial_kb = defaultdict(dict)
    partial_inverted = {}
    
    for i, file_path in enumerate(files_chunk):
        tab_id = start_tab_id + i
        
        df = read_csv_polars(file_path)
        if df is None:
            continue
            
        table_name = make_table_key(file_path)
        total_cols = len(df.columns)
        text_cols = []
        for c_idx in range(total_cols):
            if get_column_type(df[:, c_idx].to_list()) == 1:
                text_cols.append(c_idx)
        
        for idx_i in range(len(text_cols)):
            col_i_idx = text_cols[idx_i]
            for idx_j in range(idx_i + 1, len(text_cols)):
                col_j_idx = text_cols[idx_j]
                
                pair_df = df.select([df.columns[col_i_idx], df.columns[col_j_idx]]).drop_nulls().unique()
                rows = pair_df.height
                if rows == 0:
                    continue
                
                # Calculate Semantics for this pair of columns
                sem = defaultdict(float)
                
                pairs = []
                for row in pair_df.iter_rows():
                    sub = preprocess_string(row[0])
                    obj = preprocess_string(row[1])
                    if check_if_null_string(sub) and check_if_null_string(obj):
                        val = f"{sub}__{obj}"
                        if val in lookup_table:
                            for s in lookup_table[val]:
                                sem[s] += (1.0 / rows)
                        pairs.append(val)
                        
                # Assign to KB
                for val in pairs:
                    if val in lookup_table:
                        current_entry = partial_kb[val]
                        for s, score in sem.items():
                            current_entry[s] = max(current_entry.get(s, 0), score)

                # Inverted Index (Relation -> Table info)
                for s, score in sem.items():
                    if s not in partial_inverted:
                        partial_inverted[s] = {table_name: (score, str(col_i_idx), str(col_j_idx))}
                    else:
                        current_tables = partial_inverted[s]
                        if table_name in current_tables:
                            if score > current_tables[table_name][0]:
                                current_tables[table_name] = (score, str(col_i_idx), str(col_j_idx))
                        else:
                            current_tables[table_name] = (score, str(col_i_idx), str(col_j_idx))
    
    # Write to temp file
    temp_file = Path(temp_dir) / f"relation_kb_chunk_{chunk_idx}.pkl"
    with open(temp_file, 'wb') as f:
        pickle.dump((dict(partial_kb), partial_inverted), f)
    
    # Return file path (lightweight)
    return str(temp_file)


# --- Main Pipeline ---

def build_type_indices(files, chunk_size=10, force=False):
    """Build Type Lookup and Type KB with per-pass caching."""
    
    # Pass 1: Type Lookup
    if not force and config.SYNTH_TYPE_LOOKUP_PATH.exists():
        print("[1/4] [SKIP] Type Lookup cached (synth_type_lookup.pkl exists)")
        with open(config.SYNTH_TYPE_LOOKUP_PATH, 'rb') as f:
            filtered_lookup = pickle.load(f)
    else:
        print(f"[1/4] Building Type Lookup Table (chunked, chunk_size={chunk_size})...")
        lookup_table = defaultdict(set)
        
        # Build chunks
        chunks = []
        for i in range(0, len(files), chunk_size):
            chunk_files = files[i:i+chunk_size]
            chunks.append((chunk_files, i))
        
        with ExecutorClass(max_workers=config.NUM_WORKERS) as executor:
            futures = {executor.submit(process_chunk_for_type_lookup, c): i for i, c in enumerate(chunks)}
            
            for future in tqdm(as_completed(futures), total=len(chunks), desc="Type Lookup Pass 1"):
                result = future.result()
                if result:
                    for val, col_ids in result.items():
                        lookup_table[val].update(col_ids)

        print(f"Lookup Table built. Size: {len(lookup_table)} values.")
        filtered_lookup = {k: v for k, v in lookup_table.items() if len(v) < 300}
        print(f"Filtered Lookup Table size: {len(filtered_lookup)}")
        
        with open(config.SYNTH_TYPE_LOOKUP_PATH, 'wb') as f:
            pickle.dump(filtered_lookup, f)

    # Pass 2: Type KB (disk-based to avoid OOM)
    index_path = config.INDEX_DIR / "synth_type_inverted_index.pkl"
    if not force and config.SYNTH_TYPE_KB_PATH.exists() and index_path.exists():
        print("[2/4] [SKIP] Type KB cached (synth_type_kb.pkl exists)")
    else:
        print(f"[2/4] Building Type KB (disk-based, chunk_size={chunk_size})...")
        
        # Create temp directory for partial files
        temp_dir = config.INDEX_DIR / "tmp_type_kb"
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Phase A: Generate partial KB files in parallel
        print("  Phase A: Generating partial KB files...")
        chunks = []
        for idx, i in enumerate(range(0, len(files), chunk_size)):
            chunk_files = files[i:i+chunk_size]
            chunks.append((chunk_files, i, filtered_lookup, idx, str(temp_dir)))
        
        temp_files = []
        with ExecutorClass(max_workers=config.NUM_WORKERS) as executor:
            futures = {executor.submit(process_chunk_for_type_kb_to_disk, c): i for i, c in enumerate(chunks)}
            
            for future in tqdm(as_completed(futures), total=len(chunks), desc="Type KB Pass 2A"):
                temp_file = future.result()
                temp_files.append(temp_file)
        
        # Phase B: Sequential merge (memory-efficient)
        print(f"  Phase B: Merging {len(temp_files)} partial files...")
        synth_kb = defaultdict(dict)
        main_table_col_index = {}
        
        for temp_file in tqdm(sorted(temp_files), desc="Type KB Pass 2B"):
            with open(temp_file, 'rb') as f:
                partial_kb, partial_index = pickle.load(f)
            
            # Merge partial KB
            for val, sem_dict in partial_kb.items():
                target = synth_kb[val]
                for s, score in sem_dict.items():
                    target[s] = max(target.get(s, 0), score)
            
            # Merge index
            main_table_col_index.update(partial_index)
            
            # Delete temp file immediately to free disk space
            os.remove(temp_file)
        
        # Cleanup temp directory
        shutil.rmtree(temp_dir, ignore_errors=True)
        
        print("Converting Type KB to final format and saving...")
        synth_kb_final = {k: list(v.items()) for k, v in synth_kb.items()}
        with open(config.SYNTH_TYPE_KB_PATH, 'wb') as f:
            pickle.dump(synth_kb_final, f)
            
        with open(index_path, 'wb') as f:
            pickle.dump(main_table_col_index, f)
        
    return filtered_lookup


def build_relation_indices(files, chunk_size=10, force=False):
    """Build Relation Lookup and Relation KB with per-pass caching."""
    
    # Pass 3: Relation Lookup
    if not force and config.SYNTH_RELATION_LOOKUP_PATH.exists():
        print("[3/4] [SKIP] Relation Lookup cached (synth_relation_lookup.pkl exists)")
        with open(config.SYNTH_RELATION_LOOKUP_PATH, 'rb') as f:
            lookup_table = pickle.load(f)
    else:
        print(f"[3/4] Building Relation Lookup Table (chunked, chunk_size={chunk_size})...")
        lookup_table = defaultdict(set)
        
        # Build chunks
        chunks = []
        for i in range(0, len(files), chunk_size):
            chunk_files = files[i:i+chunk_size]
            chunks.append((chunk_files, i))
        
        with ExecutorClass(max_workers=config.NUM_WORKERS) as executor:
            futures = {executor.submit(process_chunk_for_relation_lookup, c): i for i, c in enumerate(chunks)}
            for future in tqdm(as_completed(futures), total=len(chunks), desc="Relation Lookup Pass 3"):
                 result = future.result()
                 if result:
                     for val, rels in result.items():
                         lookup_table[val].update(rels)
                         
        with open(config.SYNTH_RELATION_LOOKUP_PATH, 'wb') as f:
            pickle.dump(lookup_table, f)
    
    # Pass 4: Relation KB (disk-based)
    if not force and config.SYNTH_RELATION_KB_PATH.exists() and config.SYNTH_RELATION_INVERTED_INDEX_PATH.exists():
        print("[4/4] [SKIP] Relation KB cached (synth_relation_kb.pkl exists)")
    else:
        print(f"[4/4] Building Relation KB (disk-based, chunk_size={chunk_size})...")
        
        # Create temp directory for partial files
        temp_dir = config.INDEX_DIR / "tmp_relation_kb"
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Phase A: Generate partial KB files in parallel
        print("  Phase A: Generating partial KB files...")
        lookup_dict = dict(lookup_table)  # Regular dict for passing to workers
        chunks = []
        for idx, i in enumerate(range(0, len(files), chunk_size)):
            chunk_files = files[i:i+chunk_size]
            chunks.append((chunk_files, i, lookup_dict, idx, str(temp_dir)))
        
        temp_files = []
        with ExecutorClass(max_workers=config.NUM_WORKERS) as executor:
            futures = {executor.submit(process_chunk_for_relation_kb_to_disk, c): i for i, c in enumerate(chunks)}
            
            for future in tqdm(as_completed(futures), total=len(chunks), desc="Relation KB Pass 4A"):
                temp_file = future.result()
                temp_files.append(temp_file)
        
        # Phase B: Sequential merge (memory-efficient)
        print(f"  Phase B: Merging {len(temp_files)} partial files...")
        synth_kb = defaultdict(dict)
        synth_inverted_index = {}
        
        for temp_file in tqdm(sorted(temp_files), desc="Relation KB Pass 4B"):
            with open(temp_file, 'rb') as f:
                partial_kb, partial_inverted = pickle.load(f)
            
            # Merge partial KB
            for val, sem_dict in partial_kb.items():
                target = synth_kb[val]
                for s, score in sem_dict.items():
                    target[s] = max(target.get(s, 0), score)
            
            # Merge inverted index
            for s, tables in partial_inverted.items():
                if s not in synth_inverted_index:
                    synth_inverted_index[s] = dict(tables)
                else:
                    target_map = synth_inverted_index[s]
                    for table_name, info in tables.items():
                        if table_name not in target_map:
                            target_map[table_name] = info
                        elif info[0] > target_map[table_name][0]:
                            target_map[table_name] = info
            
            # Delete temp file immediately to free disk space
            os.remove(temp_file)
        
        # Cleanup temp directory
        shutil.rmtree(temp_dir, ignore_errors=True)
                        
        print("Converting Relation KB to final format and saving...")
        synth_kb_final = {k: list(v.items()) for k, v in synth_kb.items()}
        with open(config.SYNTH_RELATION_KB_PATH, 'wb') as f:
            pickle.dump(synth_kb_final, f)
            
        with open(config.SYNTH_RELATION_INVERTED_INDEX_PATH, 'wb') as f:
            pickle.dump(synth_inverted_index, f)


def extract_db_ids_from_triplets(triplet_files):
    """Extract all unique DB IDs from triplet JSONL files."""
    import json
    db_ids = set()
    for triplet_file in triplet_files:
        with open(triplet_file, 'r') as f:
            for line in f:
                triplet = json.loads(line)
                db_ids.add(triplet['anchor'])
                db_ids.add(triplet['positive'])
                db_ids.update(triplet['negatives'])
    return db_ids


def filter_files_by_db_ids(files, db_ids):
    """Filter CSV files to only those belonging to specified DB IDs."""
    # DB ID is typically the first token of the directory name, e.g. "00123 SomeName"
    filtered = []
    for f in files:
        # Get the database directory (parent of 'tables' if exists, else parent)
        db_dir = f.parent.parent if f.parent.name == 'tables' else f.parent
        db_name = db_dir.name
        # Extract ID (first token)
        db_id = db_name.split()[0] if ' ' in db_name else db_name
        if db_id in db_ids or db_name in db_ids:
            filtered.append(f)
    return filtered


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Synthesize SANTOS Knowledge Base")
    parser.add_argument("--use-process", action="store_true",
                        help="Use ProcessPoolExecutor instead of ThreadPoolExecutor")
    parser.add_argument("--chunk-size", type=int, default=100,
                        help="Number of files per chunk for Pass 2 (default: 100)")
    parser.add_argument("--force", action="store_true",
                        help="Force rebuild, ignore cached files")
    parser.add_argument("--filter-triplets", nargs='+', type=str, default=None,
                        help="Filter to only DBs in these triplet JSONL files")
    args = parser.parse_args()
    
    # Set executor class based on flag
    if args.use_process:
        ExecutorClass = ProcessPoolExecutor
        print("Using ProcessPoolExecutor (multiprocessing)")
    else:
        ExecutorClass = ThreadPoolExecutor
        print("Using ThreadPoolExecutor (multithreading)")
    
    # Override chunk size from config if provided in env
    final_chunk_size = int(os.environ.get("SANTOS_CHUNK_SIZE", args.chunk_size))
    print(f"Chunk size: {final_chunk_size}")
    if args.force:
        print("Force mode: ignoring cached files")
    
    files = list(config.UNZIP_DIR.rglob("*.csv"))
    print(f"Found {len(files)} total CSV files.")
    
    # Apply filtering if triplet files provided
    force = args.force
    if args.filter_triplets:
        print(f"Filtering to DBs from {len(args.filter_triplets)} triplet file(s)...")
        db_ids = extract_db_ids_from_triplets(args.filter_triplets)
        print(f"  Unique DB IDs in triplets: {len(db_ids)}")
        files = filter_files_by_db_ids(files, db_ids)
        print(f"  Files after filtering: {len(files)}")
    
    # Sort files to ensure deterministic IDs
    files.sort()
    
    build_type_indices(files, chunk_size=final_chunk_size, force=force)
    build_relation_indices(files, chunk_size=final_chunk_size, force=force)
