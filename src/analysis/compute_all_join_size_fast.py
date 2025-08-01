import sqlite3
import os
import pathlib
import argparse
import glob
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

def _get_ndv(cursor: sqlite3.Cursor, table_name: str, column_name: str) -> int:
    """Helper to get Number of Distinct Values (NDV) for a column."""
    try:
        # Ensure table and column names are quoted for safety
        query = f'SELECT COUNT(DISTINCT "{column_name}") FROM "{table_name}" WHERE "{column_name}" IS NOT NULL'
        cursor.execute(query)
        result = cursor.fetchone()
        return result[0] if result else 0
    except sqlite3.Error as e:
        # This might happen if the column name is strange or table doesn't exist as expected
        print(f"[WARNING] NDV calculation error for {table_name}.{column_name}: {e}")
        return 0 # Treat as 0 if error, implying no distinct values contribute

def estimate_join_cardinality(db_path_dir: str) -> float | None:
    """
    Estimates the cardinality of joining all tables in a SQLite database
    using a heuristic based on table sizes and foreign key NDVs.
    Uses an in-memory copy of the database for querying to improve stability.

    Args:
        db_path_dir: Directory path containing 'database.db'.

    Returns:
        An estimated cardinality (float), or None on critical error.
    """
    source_db_file_rel = os.path.join(db_path_dir, "database.db")
    # Use absolute path for the initial connection to the source DB
    absolute_source_db_file = os.path.abspath(source_db_file_rel)

    # print(f"Processing database file: {absolute_source_db_file}") # Changed to print absolute path

    if not os.path.exists(absolute_source_db_file):
        print(f"[ERROR] Database file not found: {absolute_source_db_file}")
        return None

    try:
        # Connect to the source database file once to back it up to memory
        with sqlite3.connect(absolute_source_db_file) as source_conn:
            # Create an in-memory database for all subsequent operations
            with sqlite3.connect(":memory:") as mem_conn:
                source_conn.backup(mem_conn) # Copy source to in-memory DB
                cursor = mem_conn.cursor()   # All operations now on the in-memory DB

                # 1. Get all user-defined table names
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
                tables_result = cursor.fetchall()
                if not tables_result:
                    return 0.0 
                
                table_names = [table[0] for table in tables_result]

                # 2. Get row counts (sizes) for all tables
                table_sizes = {}
                for tn_str in table_names:
                    try:
                        cursor.execute(f'SELECT COUNT(*) FROM "{tn_str}"')
                        table_sizes[tn_str] = cursor.fetchone()[0]
                    except sqlite3.Error as e:
                        print(f"[Count Error] {db_path_dir}, table {tn_str}: {e}")
                        table_sizes[tn_str] = 0 

                if not table_sizes: 
                    return 0.0

                # 3. Handle single table case
                if len(table_names) == 1:
                    return float(table_sizes.get(table_names[0], 0))

                # 4. Get all foreign key relationships
                fk_query = """
                SELECT m.name as child_table, f."table" as parent_table,
                       f."from" as child_column, f."to" as parent_column
                FROM sqlite_master m, pragma_foreign_key_list(m.name) f
                WHERE m.type = 'table' AND m.name NOT LIKE 'sqlite_%';
                """
                cursor.execute(fk_query)
                all_fks = cursor.fetchall()

                if not all_fks:
                    print(f"[WARNING] {db_path_dir}: Multiple tables exist but no foreign keys found. "
                          f"Cannot estimate join cardinality via FKs. "
                          f"Returning size of the largest table as a rough estimate.")
                    return float(max(table_sizes.values()) if table_sizes else 0)

                # 5. Select base_table (largest by row count)
                sorted_tables_by_name = sorted(table_names, key=lambda t: table_sizes.get(t, 0), reverse=True)
                base_table_name = sorted_tables_by_name[0]
                
                # 6. Initialize estimated_cardinality
                estimated_cardinality = float(table_sizes.get(base_table_name, 0))
                if estimated_cardinality == 0: 
                    return 0.0

                joined_tables_set = {base_table_name}
                
                # 7. Iteratively "join" tables
                max_iterations = len(table_names) 
                current_iteration = 0
                while len(joined_tables_set) < len(table_names) and current_iteration < max_iterations:
                    current_iteration += 1
                    added_table_in_pass = False
                    
                    for child_T, parent_T, child_col, parent_col in all_fks:
                        T_in, T_out, fk_col_in_T_out = (None, None, None) # Removed pk_col_in_T_in as it's not used
                        is_case_B = False 

                        if child_T in joined_tables_set and parent_T not in joined_tables_set:
                            T_in, T_out = child_T, parent_T
                        elif parent_T in joined_tables_set and child_T not in joined_tables_set:
                            T_in, T_out = parent_T, child_T
                            fk_col_in_T_out = child_col 
                            is_case_B = True
                        
                        if T_out: 
                            if is_case_B:
                                factor = 1.0
                                size_T_out = float(table_sizes.get(T_out, 0))
                                if size_T_out > 0:
                                    # _get_ndv now uses the cursor from mem_conn
                                    ndv_fk_in_T_out = float(_get_ndv(cursor, T_out, fk_col_in_T_out))
                                    if ndv_fk_in_T_out > 0:
                                        factor = size_T_out / ndv_fk_in_T_out
                                estimated_cardinality *= factor
                            
                            joined_tables_set.add(T_out)
                            added_table_in_pass = True
                    
                    if not added_table_in_pass:
                        if len(joined_tables_set) < len(table_names):
                             print(f"[WARNING] {db_path_dir}: Heuristic join could not connect all tables. "
                                   f"{len(table_names) - len(joined_tables_set)} tables remain unjoined. "
                                   f"Estimate is based on the joined component: {', '.join(joined_tables_set)}.")
                        break 

                return estimated_cardinality

    except sqlite3.Error as e:
        # absolute_source_db_file is defined at the start of the function
        print(f"[ERROR] SQLite error for {db_path_dir} (Path: {absolute_source_db_file}): {e}")
        return None
    except Exception as e:
        print(f"[ERROR] An unexpected error occurred for {db_path_dir}: {e}")
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--batch-size", type=int, default=100)
    parser.add_argument("-t", "--num-threads", type=int, default=64)
    args = parser.parse_args()

    db_dirs = sorted(glob.glob("data/unziplink/*"))
    results = []
    batch_size = args.batch_size
    output_file = "data/graph/all_join_size_results_est.csv"
    temp_output_dir = "data/graph/tmp/"

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Create temp directory if it doesn't exist
    if not os.path.exists(temp_output_dir):
        os.makedirs(temp_output_dir)
        
    # Process databases in batches with multiple threads
    total_dbs = len(db_dirs)
    total_batches = (total_dbs + batch_size - 1) // batch_size
    
    def process_batch(batch_id):
        thread_results = []
        start_idx = batch_id * batch_size
        end_idx = min(start_idx + batch_size, total_dbs)
        batch_dbs = db_dirs[start_idx:end_idx]
        
        for db_dir in batch_dbs:
            count = estimate_join_cardinality(db_dir) # Calls the modified function
            if count is not None:
                # Ensure db_id extraction is robust, e.g. if names have multiple parts
                db_name_part = os.path.basename(db_dir)
                db_id = db_name_part.split('-')[0] if '-' in db_name_part else db_name_part
                thread_results.append({"db_id": db_id, "all_join_size": count})
        
        return thread_results
    
    with ThreadPoolExecutor(max_workers=args.num_threads) as executor:
        futures = [executor.submit(process_batch, i) for i in range(total_batches)]
        
        # Use tqdm to show progress as futures complete
        for future in tqdm(as_completed(futures), total=total_batches, desc="Processing databases"):
            try:
                batch_result = future.result()
                if batch_result: # Ensure batch_result is not None or empty
                    results.extend(batch_result)
            except Exception as e:
                print(f"[ERROR] Error processing batch: {e}") # Catch errors from process_batch
    
    # Final save
    if results: # Only save if there are results
        df = pd.DataFrame(results)
        df.to_csv(output_file, index=False)
        print(f"\n======== AllJoinSize Stats (Heuristic Estimation) ========")
        print(f"Results saved to: {output_file}")
        print(f"Total databases processed with results: {len(df)}")
        if not df.empty and 'all_join_size' in df.columns:
            print(f"Min: {df['all_join_size'].min()}")
            print(f"Max: {df['all_join_size'].max()}")
            print(f"Mean: {df['all_join_size'].mean()}")
            print(f"Median: {df['all_join_size'].median()}")
        else:
            print("No 'all_join_size' data to report stats on.")
    else:
        print("No results were successfully processed to save or report stats.")

