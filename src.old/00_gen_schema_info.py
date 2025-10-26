import os
import json
import time
import pandas as pd
from multiprocessing import Pool, cpu_count
from model.WKDataset import WKDataset
from tqdm import tqdm

PROCESSED_DB_RECORD = "processed_db_ids.txt"

def load_processed_ids():
    if not os.path.exists(PROCESSED_DB_RECORD):
        return set()
    with open(PROCESSED_DB_RECORD, "r") as f:
        return set(line.strip() for line in f)

def mark_processed_batch(db_ids):
    with open(PROCESSED_DB_RECORD, "a") as f:
        for db_id in db_ids:
            f.write(f"{db_id}\n")

def process_db_wrapper(args):
    """Wrapper for multiprocessing (with new WKDataset instance inside each process)."""
    db_id, sample, sample_size = args
    try:
        loader = WKDataset()
        schema = loader.load_database(db_id)
        db_name = schema.get("database_name", db_id)
        tables_data = loader.load_csv_data(db_id=db_id, sample=sample, sample_size=sample_size)

        db_entry = {
            "db_id": db_id,
            "database_name": db_name,
            "alternative_database_names": schema.get("alternative_database_names", []),
            "wikidata_property_id": schema.get("wikidata_property_id", ""),
            "wikidata_property_label": schema.get("wikidata_property_label", ""),
            "wikidata_topic_item_id": schema.get("wikidata_topic_item_id", ""),
            "wikidata_topic_item_label": schema.get("wikidata_topic_item_label", ""),
            "tables": []
        }

        for table in schema.get("tables", []):
            table_name = table["table_name"]
            table_entry = {
                "table_name": table_name,
                "columns": []
            }

            df = tables_data.get(table_name, pd.DataFrame())

            for col in table.get("columns", []):
                col_name = col["column_name"]
                data_type = col.get("data_type", "unknown")
                wikidata_id = col.get("wikidata_property_id", "N/A")

                if col_name in df.columns:
                    values = df[col_name].dropna().astype(str).unique().tolist()[:sample_size]
                    sample_str = " | ".join(values) if values else "N/A"
                else:
                    sample_str = "N/A"

                col_entry = {
                    "column_name": col_name,
                    "data_type": data_type,
                    "sample": sample_str,
                    "wikidata_property_id": wikidata_id
                }

                table_entry["columns"].append(col_entry)

            db_entry["tables"].append(table_entry)

        return db_id, db_entry

    except Exception as e:
        print(f"Failed to process DB {db_id}: {e}")
        return None

def extract_all_schemas(output_file, sample=True, sample_size=5, max_workers=None):
    if max_workers is None:
        max_workers = max(cpu_count() - 1, 1)

    schemas = {}

    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            old_data = json.load(f)
        schemas.update(old_data)
        print(f"Loaded {len(old_data)} previously saved schemas.")

    processed_ids = load_processed_ids()
    db_ids = [f"{i:05d}" for i in range(100000) if f"{i:05d}" not in processed_ids]
    print(f"Processing {len(db_ids)} databases using {max_workers} processes...")

    start_time = time.time()
    batch_args = [(db_id, sample, sample_size) for db_id in db_ids]
    newly_processed = []

    with Pool(processes=max_workers) as pool:
        for result in tqdm(pool.imap_unordered(process_db_wrapper, batch_args), total=len(batch_args)):
            if result:
                db_id, db_entry = result
                schemas[db_id] = db_entry
                newly_processed.append(db_id)

    mark_processed_batch(newly_processed)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(schemas, f, indent=2, ensure_ascii=False)

    elapsed = time.time() - start_time
    print(f"\nDone. Total {len(schemas)} schemas saved.")
    print(f"Time elapsed: {elapsed:.2f} seconds.")


output_file = "../data/database_schema.json"
extract_all_schemas(output_file, sample=True, sample_size=5, max_workers=12)
