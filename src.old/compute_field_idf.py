import os
import json
import math
from collections import defaultdict
import pandas as pd
from tqdm import tqdm

SCHEMA_DIR = "data/schema"

def load_schema_files(schema_dir):
    schema_files = [
        os.path.join(schema_dir, f)
        for f in os.listdir(schema_dir)
        if f.endswith(".json")
    ]
    return schema_files

def extract_field_names(schema_json):
    field_names = set()
    for table in schema_json.get("tables", []):
        for column in table.get("columns", []):
            name = column.get("column_name")
            if name:
                field_names.add(name.lower().strip())
    return field_names

def compute_idf():
    schema_files = load_schema_files(SCHEMA_DIR)
    df_counter = defaultdict(int)
    total_docs = len(schema_files)

    progress = tqdm(schema_files, total=total_docs, desc="📦 Processing schemas", ncols=100)

    for idx, file_path in enumerate(progress, start=1):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                schema = json.load(f)
                field_names = extract_field_names(schema)
                for field in field_names:
                    df_counter[field] += 1
        except Exception as e:
            progress.write(f"[Error] {file_path}: {e}")

        if idx % 10000 == 0:
            progress.write(f"✅ Processed {idx}/{total_docs} files...")

    print(f"✅ Total schemas processed: {total_docs}")
    
    idf_scores = {
        field: round(math.log(total_docs / (1 + df)), 5)
        for field, df in df_counter.items()
    }

    # Save to CSV
    df_out = pd.DataFrame([
        {"field": field, "df": df_counter[field], "idf": idf_scores[field]}
        for field in df_counter
    ])
    df_out.sort_values(by="idf", ascending=False).to_csv("field_idf_scores.csv", index=False)
    print("📄 Saved to field_idf_scores.csv")

if __name__ == "__main__":
    compute_idf()