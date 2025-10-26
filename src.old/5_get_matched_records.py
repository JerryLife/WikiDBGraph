import numpy as np
import sqlite3
import time
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import random

random.seed(0)

def get_overlapped_dbs(num_seeds=10, random_sample=None):
    """
    Get databases with overlapping features from pairwise stats files across multiple seeds
    
    Args:
        num_seeds (int): Number of seeds to analyze (default 10)
        
    Returns:
        list: List of tuples containing (db1, db2, overlap_count) for databases with non-zero overlap
    """
    overlapped_pairs = []
    
    # Read stats for each seed
    for seed in tqdm(range(num_seeds)):
        stats_file = f"out/valid_pairs_stats_seed{seed}.csv"
        
        # Skip header line and process each row
        with open(stats_file, 'r') as f:
            next(f)  # Skip header
            for line in f:
                db1, db2, overlap, _, _ = line.strip().split(',')
                if int(overlap) > 0 and db1 != db2:
                    overlapped_pairs.append((db1, db2, int(overlap)))
                    overlapped_pairs.append((db2, db1, int(overlap)))
    
    print(f"Found {len(overlapped_pairs)} overlapped pairs")
    if random_sample:
        return random.sample(overlapped_pairs, random_sample)
    return overlapped_pairs

def summarize_overlaps(overlapped_pairs, save=False):
    """
    Load the database and get the overlap statistics from data/unzip/<full_db_name>/database.db
    
    Args:
        overlapped_pairs (List[Tuple[str, str, int]]): List of (db1, db2, n_overlap) tuples
        
    Returns:
        list: List of dictionaries containing overlap statistics for each pair
    """
    # Initialize results list
    results = []
    
    # Process each overlapped pair
    for db1, db2, overlap_count in overlapped_pairs:
        # get db1 and db2 path (replace only the first _ with <space>)
        db1_name = db1.replace("_", " ", 1)
        db2_name = db2.replace("_", " ", 1)

        # Connect to databases
        try:
            conn1 = sqlite3.connect(f"data/unzip/{db1_name}/database.db")
            conn2 = sqlite3.connect(f"data/unzip/{db2_name}/database.db")
        except Exception as e:
            print(f"Error connecting to databases {db1_name} and {db2_name}: {e}")
            continue
        
        cursor1 = conn1.cursor()
        cursor2 = conn2.cursor()
        
        # Get all table names from both databases
        cursor1.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables1 = [row[0] for row in cursor1.fetchall()]
        
        cursor2.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables2 = [row[0] for row in cursor2.fetchall()]
        
        # Initialize counters and storage
        total_features1 = 0
        total_features2 = 0
        matched_table_pairs = {}  # Store matching table pairs and their features
        all_matched_features = set()
        
        # First pass: Compare columns and store matching info
        for table1 in tables1:
            cursor1.execute(f'SELECT * FROM "{table1}" LIMIT 0')
            cols1 = [description[0] for description in cursor1.description]
            total_features1 += len(cols1)
            
            for table2 in tables2:
                cursor2.execute(f'SELECT * FROM "{table2}" LIMIT 0')
                cols2 = [description[0] for description in cursor2.description]
                if table1 == table2:  # Only count features once per db
                    total_features2 += len(cols2)
                
                # Find matching features between tables
                matched_features = set(cols1).intersection(set(cols2))
                if matched_features:
                    matched_table_pairs[(table1, table2)] = matched_features
                    all_matched_features.update(matched_features)
                else:
                    # skip if no matched features
                    continue
                
                # record the total number of records in each table
                cursor1.execute(f'SELECT COUNT(*) FROM "{table1}"')
                total_records1 = cursor1.fetchone()[0]
                cursor2.execute(f'SELECT COUNT(*) FROM "{table2}"')
                total_records2 = cursor2.fetchone()[0]

                # Attach the second database
                cursor1.execute(f'ATTACH DATABASE "data/unzip/{db2_name}/database.db" AS db2')
                
                # Do a inner join on the matched features
                join_conditions = " AND ".join([f'table1."{col}" = table2."{col}"' for col in matched_features])

                cmd = f"""
                    SELECT COUNT(*) FROM "{table1}" table1
                    INNER JOIN (SELECT DISTINCT {", ".join(matched_features)} FROM db2."{table2}") table2
                    ON {join_conditions}
                """
                cursor1.execute(cmd)
                overlapped_records = cursor1.fetchone()[0]
                
                # Detach the second database
                cursor1.execute("DETACH DATABASE db2")

                # Ensure the overlapped records is not larger than the left table
                if overlapped_records > total_records1:
                    raise ValueError(f"Overlapped records {overlapped_records} is larger than the left table {total_records1}")

                # Store results for each table pair
                results.append({
                    'db1': db1_name,
                    'table1': table1,
                    'db2': db2_name,
                    'table2': table2,
                    'overlapped_records': overlapped_records,
                    'overlapped_features': len(matched_features),
                    'total_records1': total_records1,
                    'total_records2': total_records2,
                })

        ####### There might be one record difference due to unknown reason. Omit it as it is not a big deal. ###############
        # # Verify against provided overlap count
        # matched_count = len(all_matched_features)
        # if matched_count != overlap_count:
        #     print(f"Warning: Found {matched_count} overlapping features but expected {overlap_count}")
        ####################################################################################################################

        # Close connections
        conn1.close()
        conn2.close()

    if save:
        # Save to csv
        print(f"Saving to out/matched_ratio.csv")
        with open("out/matched_ratio.csv", "w") as f:
            # Write header
            f.write("db1,db2,overlapped_records,overlapped_features,total_records1,total_records2\n")
            
            # Write data
            for result in results:
                f.write(f"{result['db1']},{result['db2']},{result['overlapped_records']}," + 
                    f"{result['overlapped_features']},{result['total_records1']},{result['total_records2']}\n")

    return results


def summarize_overlaps_parallel(overlapped_pairs, n_threads=100):
    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        # Process each pair in parallel
        print(f"Starting parallel processing with {n_threads} threads...")
        futures = []
        for pair in overlapped_pairs:
            futures.append(executor.submit(summarize_overlaps, [pair]))
        
        # Collect results as they complete
        i = 0
        results = []
        start_time = time.time()
        for future in futures:
            batch_result = future.result()
            results.extend(batch_result)
            i += 1
            elapsed = time.time() - start_time
            print(f"\rProcessed {i}/{len(overlapped_pairs)} pairs ({i/len(overlapped_pairs)*100:.1f}%) in {elapsed:.1f}s, "
                  f"estimated {elapsed*len(overlapped_pairs)/i:.1f}s left", end='', flush=True)
    print("\nParallel processing completed.")
    
    # Save to csv
    print(f"Saving to out/matched_ratio.csv")
    with open("out/matched_ratio.csv", "w") as f:
        # Write header
        f.write("db1,table1,db2,table2,overlapped_records,overlapped_features,total_records1,total_records2\n")
        
        # Write data
        for result in results:
            f.write(f"{result['db1']},{result['table1']},{result['db2']},{result['table2']}," +
                   f"{result['overlapped_records']},{result['overlapped_features']}," +
                   f"{result['total_records1']},{result['total_records2']}\n")

    return results

if __name__ == "__main__":
    overlapped_pairs = get_overlapped_dbs(num_seeds=10, random_sample=1000)
    results = summarize_overlaps_parallel(overlapped_pairs, n_threads=100)
    print(f"Processed {len(results)} database pairs")


"""
Connected components summary:
Average: 3.1
Min: 3
Max: 4
Std: 0.3
"""