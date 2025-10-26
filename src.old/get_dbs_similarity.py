import time
import sys
import os
import json
import random
import time
import csv
from fuzzywuzzy import fuzz
from itertools import combinations
from typing import Dict, Tuple, List
from concurrent.futures import ThreadPoolExecutor, as_completed

def process_schema(db_name, schema_folder):
    """
    Process a single database schema by loading its JSON file.
    
    :param db_name: Name of the database
    :param schema_folder: Folder containing schema JSON files
    :return: Tuple (db_name, schema_info) or None if file not found
    """
    schema_path = os.path.join(schema_folder, f"{db_name}.json")
    
    if not os.path.exists(schema_path):
        print(f"\nWarning: Schema file {schema_path} not found. Skipping...")
        return None

    try:
        with open(schema_path, 'r', encoding="utf-8") as schema_file:
            schema = json.load(schema_file)

        # Extract Q-ID → Label and P-ID → Label mappings
        qid_label_map, pid_label_map = extract_qids_pids(schema)

        # Extract table names (converted to lowercase)
        table_names = set(table["table_name"].lower() for table in schema.get("tables", []))

        return db_name, {
            "q_ids": set(qid_label_map.keys()),  # Set of Q-IDs
            "q_labels": qid_label_map,  # Q-ID to Label mapping
            "p_ids": set(pid_label_map.keys()),  # Set of P-IDs
            "p_labels": pid_label_map,  # P-ID to Label mapping
            "tables": table_names  # Set of table names
        }
    except Exception as e:
        print(f"\n❌ Error processing {schema_path}: {e}")
        return None

def load_database_stats(column_stats_path, schema_folder, max_dbs=10000, max_workers=12):
    """
    Load database statistics from column_stats.txt and fetch schemas from JSON files using multithreading.
    Stops after processing max_dbs databases.

    :param column_stats_path: Path to column_stats.txt
    :param schema_folder: Folder containing JSON schema files.
    :param max_workers: Number of threads to use.
    :param max_dbs: Maximum number of databases to process.
    :return: Dictionary of database statistics
    """
    db_stats = {}

    # Read only the first `max_dbs` lines
    with open(column_stats_path, 'r', encoding="utf-8") as f:
        db_names = [line.split(',')[0].strip() for _, line in zip(range(max_dbs), f)]  # Limit to max_dbs entries

    total_dbs = len(db_names)  # This ensures we process up to `max_dbs`
    print(f"Processing {total_dbs} databases from column_stats.txt using {max_workers} threads...")

    start_time = time.time()

    # Multithreading for loading schemas
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_db = {executor.submit(process_schema, db_name, schema_folder): db_name for db_name in db_names}

        for i, future in enumerate(as_completed(future_to_db), 1):
            result = future.result()
            if result:
                db_name, schema_info = result
                db_stats[db_name] = schema_info

            # Update progress bar
            elapsed_time = time.time() - start_time
            avg_time_per_db = elapsed_time / i
            remaining_time = (total_dbs - i) * avg_time_per_db
            progress = (i / total_dbs) * 100
            sys.stdout.write(f"\rProcessed: {i}/{total_dbs} databases [{progress:.2f}%] | ⏳ ETA: {remaining_time:.2f} sec")
            sys.stdout.flush()

    total_time = time.time() - start_time
    print(f"\nDatabase stats loading complete in {total_time:.2f} sec.")
    return db_stats


def extract_qids_pids(schema):
    """
    Recursively extract all Q-IDs and P-IDs from the database schema, mapping each to its label.
    :param schema: The schema dictionary of a database.
    :return: qid_label_map, pid_label_map
    """
    qid_label_map = {}  # Q-ID → Label mapping
    pid_label_map = {}  # P-ID → Label mapping

    # Extract Q-ID from the database-level metadata
    if "wikidata_topic_item_id" in schema and schema["wikidata_topic_item_id"]:
        qid_label_map[schema["wikidata_topic_item_id"]] = schema.get("wikidata_topic_item_label", "Unknown Label")

    # Extract P-ID from the database-level metadata
    if "wikidata_property_id" in schema and schema["wikidata_property_id"]:
        pid_label_map[schema["wikidata_property_id"]] = schema.get("wikidata_property_label", "Unknown Label")

    # Iterate through tables
    for table in schema.get("tables", []):
        
        # Extract P-IDs from the table level
        if "wikidata_property_id" in table and table["wikidata_property_id"]:
            pid_label_map[table["wikidata_property_id"]] = table.get("wikidata_property_label", "Unknown Label")

    return qid_label_map, pid_label_map


def find_duplicate_qids(db_stats, output_file="duplicate_qids.csv"):
    """
    Identify duplicate Q-IDs across databases with progress tracking and save the results.

    :param db_stats: Dictionary containing database statistics.
    :param output_file: Path to the output CSV file for duplicate Q-IDs.
    :return: Dictionary mapping Q-IDs to a list of databases that share them.
    """
    qid_to_dbs = {}  # Mapping Q-ID → List of database names
    qid_to_labels = {}  # Mapping Q-ID → Label (from db_stats)
    total_dbs = len(db_stats)
    start_time = time.time()

    print(f"🔍 Scanning {total_dbs} databases for duplicate Q-IDs...")

    # Count Q-IDs and track which databases share them
    for i, (db_name, stats) in enumerate(db_stats.items(), 1):
        for qid in stats["q_ids"]:  # q_ids is a set
            if qid not in qid_to_dbs:
                qid_to_dbs[qid] = []
                # Use stored labels from `db_stats`
                qid_to_labels[qid] = stats["q_labels"].get(qid, "Unknown Label")

            qid_to_dbs[qid].append(db_name)

        # Progress tracking
        elapsed_time = time.time() - start_time
        avg_time_per_db = elapsed_time / i
        remaining_time = (total_dbs - i) * avg_time_per_db
        progress = (i / total_dbs) * 100

        if i % 100 == 0 or i == total_dbs:  # Update progress every 100 databases
            sys.stdout.write(f"\rProcessed: {i}/{total_dbs} databases [{progress:.2f}%] | ⏳ ETA: {remaining_time:.2f} sec")
            sys.stdout.flush()

    print("\n✅ Q-ID scanning complete. Checking for duplicates...")

    # Find duplicates (Q-IDs shared by multiple databases)
    duplicate_qids = {qid: dbs for qid, dbs in qid_to_dbs.items() if len(dbs) > 1}

    # Print results
    if duplicate_qids:
        print("🔍 Duplicate Q-IDs Found:")
        for qid, dbs in duplicate_qids.items():
            label = qid_to_labels.get(qid, "Unknown Label")
            print(f"Q-ID {qid} ({label}) is shared by {len(dbs)} databases: {', '.join(dbs[:5])} ...")  # Limit to 5 names for readability

        # Save results to a CSV file
        with open(output_file, 'w', encoding="utf-8", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Q-ID", "Label", "Number of Databases", "Databases"])
            for qid, dbs in duplicate_qids.items():
                label = qid_to_labels.get(qid, "Unknown Label")
                writer.writerow([qid, label, len(dbs), "; ".join(dbs)])  # Use semicolon as separator

        print(f"📂 Duplicate Q-IDs saved to {output_file}")

    else:
        print("✅ No duplicate Q-IDs found.")

    return duplicate_qids


def compute_similarity(db1_info, db2_info):
    """
    Compute similarity metrics between two databases.
    """
    qid_overlap = len(db1_info["q_ids"] & db2_info["q_ids"])
    pid_overlap = len(db1_info["p_ids"] & db2_info["p_ids"])
    table_similarities = [fuzz.ratio(t1, t2) for t1 in db1_info["tables"] for t2 in db2_info["tables"]]
    table_name_similarity = (sum(table_similarities) / len(table_similarities) if table_similarities else 0) / 100.0
    similarity_score = (0.4 * qid_overlap + 0.3 * pid_overlap + 0.3 * table_name_similarity)

    return similarity_score, qid_overlap, pid_overlap, table_name_similarity


def calculate_pairwise_similarity(db_stats: Dict[str, Dict]) -> Dict[Tuple[str, str], Dict[str, float]]:
    """
    Compute pairwise similarity between databases using a random sampling strategy with a progress display.
    """
    def process_pair(pair: Tuple[str, str]) -> Tuple[str, str, float, int, int, float]:
        db1, db2 = pair
        similarity, qid_overlap, pid_overlap, table_similarity = compute_similarity(db_stats[db1], db_stats[db2])
        return (db1, db2, similarity, qid_overlap, pid_overlap, table_similarity)

    pairwise_similarities = {}

    # Run 10 times with 500 sampled databases each time
    for run in range(10):
        print(f"\n🔄 Running Run {run}...")

        generator = random.Random(run)
        sampled_dbs = generator.sample(list(db_stats.keys()), min(50, len(db_stats)))

        # Generate all pairs
        pairs = list(combinations(sampled_dbs, 2))
        total_pairs = len(pairs)
        print(f"📊 Sampled {len(sampled_dbs)} databases, generating {total_pairs} pairs to process...")

        start_time = time.time()
        run_stats = {}

        # Multithreading: Reduce max_workers if CPU usage is too high
        with ThreadPoolExecutor(max_workers=4) as executor:
            future_to_pair = {executor.submit(process_pair, pair): pair for pair in pairs}
            print(f"🚀 Submitted {total_pairs} tasks for processing...")

            for i, future in enumerate(as_completed(future_to_pair), 1):
                try:
                    db1, db2, similarity, qid_overlap, pid_overlap, table_similarity = future.result()
                    run_stats[(db1, db2)] = {
                        'similarity_score': similarity,
                        'pid_overlap': pid_overlap,
                        'qid_overlap': qid_overlap,
                        'table_name_similarity': table_similarity
                    }
                except Exception as e:
                    print(f"\n❌ Error processing pair {future_to_pair[future]}: {e}")
                    continue  # Skip faulty pair and continue

                # Progress tracking
                elapsed_time = time.time() - start_time
                avg_time_per_pair = elapsed_time / i
                remaining_time = (total_pairs - i) * avg_time_per_pair
                progress = (i / total_pairs) * 100

                if i % 100 == 0 or i == total_pairs:  # Print every 100 pairs
                    print(f"\n✅ Processed {i}/{total_pairs} pairs [{progress:.2f}%] | ⏳ ETA: {remaining_time:.2f} sec")

        file_path = f'../out/pairwise_similarity_seed{run}.csv'
        print(f"\n📂 Writing results to {file_path}...")

        with open(file_path, 'w', encoding="utf-8") as f:
            f.write("db1,db2,similarity_score,qid_overlap,pid_overlap,table_name_similarity\n")
            for (db1, db2), stats in run_stats.items():
                f.write(f"{db1},{db2},{stats['similarity_score']},{stats['qid_overlap']},{stats['pid_overlap']},{stats['table_name_similarity']}\n")

        pairwise_similarities.update(run_stats)
        print(f"✅ Completed Run {run} in {time.time() - start_time:.2f} sec.")

    print("\n🎉 All runs completed!")
    return pairwise_similarities



if __name__ == "__main__":
    column_stats_path = "../data/column_stats.txt"
    schema_folder = "../data/schema"
    
    db_stats = load_database_stats(column_stats_path, schema_folder, 1000)
    print(f"✅ Loaded {len(db_stats)} databases")

    # print(find_duplicate_qids(db_stats))
    
    pairwise_similarity = calculate_pairwise_similarity(db_stats)

