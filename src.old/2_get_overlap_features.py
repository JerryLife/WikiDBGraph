from concurrent.futures import ThreadPoolExecutor
from itertools import combinations
from typing import Dict, Tuple, List
import random
import os


# ensure GIL is off by PYTHON_CONFIGURE_OPTS
import sysconfig
print(sysconfig.get_config_var("Py_GIL_DISABLED"))


def load_column_stats(filepath):
    column_stats = {}
    with open(filepath, 'r') as f:
        for line in f:
            stats = line.split(',')
            database_name = stats[0]
            num_tables = int(stats[1])
            num_unique_columns = int(stats[2])
            column_names = stats[3:]

            column_stats[database_name] = column_names

    return column_stats


def calculate_pairwise_stats(column_stats: Dict[str, List[str]]) -> Dict[Tuple[str, str], Dict[str, int]]:
    def process_pair(pair: Tuple[str, str]) -> Tuple[str, str, int, int, int]:
        if pair[0] > pair[1]:
            return (pair[0], pair[1], -1, -1, -1)   # invalid pair, already processed
        
        db1, db2 = pair
        cols1 = set(column_stats[db1])
        cols2 = set(column_stats[db2])
        
        overlap = len(cols1.intersection(cols2))

        return (db1, db2, overlap, len(cols1), len(cols2))
    
    pairwise_stats = {}
    
    # Run 10 times with 1000 sampled databases each time
    for run in range(10):
        print(f"Running Run {run}...")
        generator = random.Random(run)
        sampled_dbs = generator.sample(list(column_stats.keys()), min(1000, len(column_stats)))
        
        # Process all pairs in parallel
        run_stats = {}
        pairs = [(db1, db2) for db1, db2 in combinations(sampled_dbs, 2)]
        
        with ThreadPoolExecutor(max_workers=100) as executor:
            # Process each pair in parallel
            for db1, db2, overlap, total1, total2 in executor.map(process_pair, pairs):
                run_stats[(db1, db2)] = {
                    'overlap_features': overlap,
                    'table1_features': total1,
                    'table2_features': total2
                }
                print(f"\rProcessed {len(run_stats)}/{len(pairs)} pairs", end='')
        
        os.makedirs('out', exist_ok=True)
        file_path = f'out/pairwise_stats_seed{run}.csv'
        with open(file_path, 'w') as f:
            print(f"Starting to write to file {file_path}...")
            f.write("db1,db2,overlap_features,table1_features,table2_features\n")
            for (db1, db2), stats in run_stats.items():
                if stats['table1_features'] != -1 and stats['table2_features'] != -1:
                    f.write(f"{db1},{db2},{stats['overlap_features']},{stats['table1_features']},{stats['table2_features']}\n")
            
        # Update overall stats
        pairwise_stats.update(run_stats)
    
    return pairwise_stats

if __name__ == "__main__":
    column_stats = load_column_stats("data/column_stats.txt")
    print(f"Loaded {len(column_stats)} databases")
    pairwise_stats = calculate_pairwise_stats(column_stats)
