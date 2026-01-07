
import os
import glob
import time
import pandas as pd
import numpy as np
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict
from pathlib import Path
import re
import sys

# Add src to path to import config
sys.path.append(str(Path(__file__).resolve().parents[3]))
from src.baseline.santos import config

# --- Helper Functions (Re-implemented from generalFunctions.py) ---

def check_if_null_string(s):
    null_list = {'nan', '-', 'unknown', 'other (unknown)', 'null', 'na', '', ' '}
    return 0 if str(s).lower() in null_list else 1

def preprocess_string(s):
    s = str(s).lower()
    s = re.sub(r'[^\w]', ' ', s)
    s = s.replace("nbsp", "")
    return " ".join(s.split())

def get_column_type(attribute, column_threshold=0.5, entity_threshold=0.5):
    # Returns 1 for Text, 0 for Numeric/Other
    attribute = [item for item in attribute if str(item).lower() != "nan"]
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

# --- Worker Functions for Parallel Processing ---

def process_table_for_type_lookup(file_path, tab_id):
    try:
        df = pd.read_csv(file_path, encoding='latin1', on_bad_lines='skip')
    except Exception:
        return None

    local_lookup = defaultdict(set)
    col_id = 0
    for col_name in df.columns:
        col_data = df[col_name].tolist()
        if get_column_type(col_data) == 1: # Text column
            unique_values = df[col_name].unique()
            # SANTOS logic: map(str) called in original
            unique_values = [str(x) for x in unique_values]
            value_list = preprocess_list_values(unique_values)
            
            col_sem = f"c{tab_id}_{col_id}"
            for val in value_list:
                local_lookup[val].add(col_sem)
        col_id += 1
    return local_lookup

def process_table_for_kb_creation(file_path, tab_id, lookup_table):
    # This step requires the global lookup_table, so it might be tricky to parallelize 
    # if lookup_table is huge. 
    # Strategy: Pass purely local data? No, we need lookup checks.
    # If lookup_table is read-only here, we can use shared memory or just pass it 
    # (expensive if pickling).
    # Since we are implementing "synthesized KB", let's follow the 2-pass approach.
    pass 

# To avoid passing huge dicts to workers, we will use a different strategy for the second pass:
# The second pass in SANTOS iterates tables again.
# `createColumnSemanticsSynthKB` logic:
# For each column, if text:
#   Calculate distribution of semantics (types) based on values' hits in lookupTable.
#   Assign this distribution to each value in the column.
#   Update global synthKB.

# Optimization: 
# The lookup table maps Value -> Set[ColumnIDs].
# The KB maps Value -> {ColID -> Score}.
# We can run the second pass in parallel if we allow workers to return partial KB updates.

def worker_column_semantics_pass2(file_path, tab_id, lookup_subset_keys):
    # We can't easily pass the whole lookup table. 
    # BUT, we can just process the table, extract values, and return (Value, ColumnDistribution) tuples.
    # The main process, which holds the Lookup Table, can then resolve the ColumnIDs.
    # Wait, the logic is: "for value in col: get semantics from lookup(value)".
    # So the worker NEEDS the lookup table.
    # Given 100k files, the lookup table might be large (millions of values).
    return None

# --- Main Pipeline ---

def build_type_indices(files):
    print("Building Type Lookup Table...")
    # PASS 1: Build Lookup Table
    # Parallelize file processing
    lookup_table = defaultdict(set)
    
    with ProcessPoolExecutor(max_workers=config.NUM_WORKERS) as executor:
        futures = {executor.submit(process_table_for_type_lookup, f, i): i for i, f in enumerate(files)}
        
        for i, future in enumerate(as_completed(futures)):
            if i % 1000 == 0:
                print(f"Processed {i}/{len(files)} files for Type Lookup")
            result = future.result()
            if result:
                for val, col_ids in result.items():
                    lookup_table[val].update(col_ids)

    print(f"Lookup Table built. Size: {len(lookup_table)} values.")
    
    # Filter noise (SANTOS logic: remove values appearing in < 300 columns? No, logic was len < 300 checks??)
    # Original: if len(lookupTable[every]) < 300: noise_free...
    # This removes "stop words" essentially.
    
    filtered_lookup = {k: v for k, v in lookup_table.items() if len(v) < 300}
    print(f"Filtered Lookup Table size: {len(filtered_lookup)}")
    
    # Save Lookup
    with open(config.SYNTH_TYPE_LOOKUP_PATH, 'wb') as f:
        pickle.dump(filtered_lookup, f)

    # PASS 2: Build Synth KB
    # Since passing filtered_lookup to workers is expensive, we might do this serially or 
    # use a shared memory manager if needed. 
    # For 100k files, serial might take a while but let's try to optimize the single thread 
    # or just use multi-threading (threads share memory) instead of multi-processing for this read-heavy part?
    # Python GIL might block CPU bound work. Preprocessing strings is CPU bound.
    # Let's try to just run it serially first or use a chunked approach.
    
    print("Building Type KB (Serial for now to avoid IPC overhead)...")
    synth_kb = defaultdict(dict)
    main_table_col_index = {}
    
    # We can parallelize this ONLY if we can efficiently share filtered_lookup.
    # For now, let's implement a robust serial loop or batch it.
    
    for tab_id, file_path in enumerate(files):
        if tab_id % 100 == 0:
            print(f"Processing table {tab_id}/{len(files)} for Type KB")
            
        try:
            df = pd.read_csv(file_path, encoding='latin1', on_bad_lines='skip')
        except:
            continue
            
        col_id = 0
        table_name = make_table_key(file_path)
        
        for col_name in df.columns:
            col_data = df[col_name].tolist()
            if get_column_type(col_data) == 1:
                # Text Column
                unique_values = [str(x) for x in df[col_name].unique()]
                value_list = preprocess_list_values(unique_values)
                divide_by = len(value_list) if value_list else 1
                
                # Calculate semantic distribution for this column
                sem = defaultdict(float)
                for val in value_list:
                    if val in filtered_lookup:
                        for s in filtered_lookup[val]:
                            sem[s] += (1.0 / divide_by)
                
                # Assign to values
                for val in value_list:
                    if val in filtered_lookup:
                        # Update global KB
                        # synthKB maps Value -> {SemanticID -> Score}
                        current_entry = synth_kb[val]
                        for s, score in sem.items():
                            current_entry[s] = max(current_entry.get(s, 0), score)
                            
                main_table_col_index[(table_name, str(col_id))] = dict(sem)
            col_id += 1

    # Convert defaultdicts to dicts for pickling
    synth_kb = {k: list(v.items()) for k, v in synth_kb.items()} # Convert inner dict to list of tuples as per SANTOS
    
    with open(config.SYNTH_TYPE_KB_PATH, 'wb') as f:
        pickle.dump(synth_kb, f)
        
    # We don't save main_table_col_index in config paths but it's used in query. 
    # Actually, config has SYNTH_TYPE_INVERTED_INDEX_PATH?
    # Let's add it to config if missing or just save it.
    # SANTOS: synthTypeInvertedIndex = main_table_col_index
    # We need SYNTH_TYPE_INVERTED_INDEX_PATH in config.
    index_path = config.INDEX_DIR / "synth_type_inverted_index.pkl"
    with open(index_path, 'wb') as f:
        pickle.dump(main_table_col_index, f)
        
    return filtered_lookup

# --- Relation Logic ---

def process_table_for_relation_lookup(file_path, tab_id):
    try:
        df = pd.read_csv(file_path, encoding='latin1', on_bad_lines='skip')
    except:
        return None
        
    local_lookup = defaultdict(set)
    total_cols = df.shape[1]
    
    # Identify text cols first
    text_cols = []
    for i in range(total_cols):
        if get_column_type(df.iloc[:, i].tolist()) == 1:
            text_cols.append(i)
            
    # Pairs of text cols
    for idx_i in range(len(text_cols)):
        i = text_cols[idx_i]
        for idx_j in range(idx_i + 1, len(text_cols)):
            j = text_cols[idx_j]
            
            # Relation Semantic ID
            rel_sem = f"r{tab_id}_{i}_{j}"
            
            # Get pairs
            # Drop duplicates and na
            pair_df = df.iloc[:, [i, j]].dropna().drop_duplicates()
            
            for row in pair_df.itertuples(index=False):
                sub = preprocess_string(str(row[0]))
                obj = preprocess_string(str(row[1]))
                
                if check_if_null_string(sub) and check_if_null_string(obj):
                    val = f"{sub}__{obj}"
                    local_lookup[val].add(rel_sem)
                    
    return local_lookup

def build_relation_indices(files):
    print("Building Relation Lookup Table...")
    lookup_table = defaultdict(set)
    
    with ProcessPoolExecutor(max_workers=config.NUM_WORKERS) as executor:
        futures = {executor.submit(process_table_for_relation_lookup, f, i): i for i, f in enumerate(files)}
        for i, future in enumerate(as_completed(futures)):
             if i % 1000 == 0:
                print(f"Processed {i}/{len(files)} files for Relation Lookup")
             result = future.result()
             if result:
                 for val, rels in result.items():
                     lookup_table[val].update(rels)
                     
    with open(config.SYNTH_RELATION_LOOKUP_PATH, 'wb') as f:
        pickle.dump(lookup_table, f)
        
    print("Building Relation KB...")
    synth_kb = defaultdict(dict)
    synth_inverted_index = {}
    
    # Serial Pass 2
    for tab_id, file_path in enumerate(files):
        if tab_id % 100 == 0:
            print(f"Processing table {tab_id}/{len(files)} for Relation KB")
            
        try:
            df = pd.read_csv(file_path, encoding='latin1', on_bad_lines='skip')
        except:
            continue
            
        table_name = make_table_key(file_path)
        total_cols = df.shape[1]
        text_cols = [i for i in range(total_cols) if get_column_type(df.iloc[:, i].tolist()) == 1]
        
        for idx_i in range(len(text_cols)):
            i = text_cols[idx_i]
            for idx_j in range(idx_i + 1, len(text_cols)):
                j = text_cols[idx_j]
                
                pair_df = df.iloc[:, [i, j]].dropna().drop_duplicates()
                rows = pair_df.shape[0]
                if rows == 0:
                    continue
                
                # Calculate Semantics for this pair of columns
                sem = defaultdict(float)
                
                pairs = []
                for row in pair_df.itertuples(index=False):
                    sub = preprocess_string(str(row[0]))
                    obj = preprocess_string(str(row[1]))
                    if check_if_null_string(sub) and check_if_null_string(obj):
                        val = f"{sub}__{obj}"
                        if val in lookup_table:
                            for s in lookup_table[val]:
                                sem[s] += (1.0 / rows)
                        pairs.append(val)
                        
                # Assign to KB
                for val in pairs:
                    if val in lookup_table:
                        current_entry = synth_kb[val]
                        for s, score in sem.items():
                            current_entry[s] = max(current_entry.get(s, 0), score)

                # Inverted Index (Relation -> Table info)
                # Key = SemanticID
                for s, score in sem.items():
                    key = s 
                    # Store (Score, Col1, Col2) for this table
                    if key not in synth_inverted_index:
                         synth_inverted_index[key] = {table_name: (score, str(i), str(j))}
                    else:
                        current_tables = synth_inverted_index[key]
                        # Keep max score if table already there (unlikely for r_ID which is unique per pair usually? 
                        # Wait, r_ID is unique per table-pair: r{tab_id}_{i}_{j}. 
                        # So sem keys are unique to this table? 
                        # No, lookup table aggregates r_IDs? 
                        # Wait. lookup_table[val] = {r_IDs}. 
                        # r_IDs are like "r0_1_2". They are unique to the table/columns.
                        # So 'sem' contains r_IDs from OTHER tables that share values.
                        # Correct.
                        
                        if table_name in current_tables:
                             if score > current_tables[table_name][0]:
                                 current_tables[table_name] = (score, str(i), str(j))
                        else:
                            current_tables[table_name] = (score, str(i), str(j))
                            
    # Format KB as list of tuples
    synth_kb = {k: list(v.items()) for k, v in synth_kb.items()}
    
    with open(config.SYNTH_RELATION_KB_PATH, 'wb') as f:
        pickle.dump(synth_kb, f)
        
    with open(config.SYNTH_RELATION_INVERTED_INDEX_PATH, 'wb') as f:
        pickle.dump(synth_inverted_index, f)

if __name__ == "__main__":
    files = list(config.UNZIP_DIR.rglob("*.csv"))
    print(f"Found {len(files)} files.")
    
    # Sort files to ensure deterministic IDs
    files.sort()
    
    build_type_indices(files)
    build_relation_indices(files)
