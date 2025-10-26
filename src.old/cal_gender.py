import os
import json
from tqdm import tqdm

SCHEMA_DIR = "/home/wangzy/wikidbs/wikidbs/data/schema"
GENDER_KEYWORDS = {"gender", "sex"}

def contains_gender(columns):
    """
    Check if the columns contain gender-related fields.
    """
    for column in columns:
        column_name = column.get("column_name", "").lower()
        alt_names = column.get("alternative_column_names", [])
        
        if column_name in GENDER_KEYWORDS or any(name.lower() in GENDER_KEYWORDS for name in alt_names):
            return True
    return False

def count_databases_with_gender(schema_dir):
    """
    Count the number of databases that contain gender-related fields.
    Print progress every 1000 files.
    """
    total_files = [f for f in os.listdir(schema_dir) if f.endswith(".json")]
    gender_db_count = 0

    for i, json_file in enumerate(tqdm(total_files, desc="Processing JSON files")):
        json_path = os.path.join(schema_dir, json_file)
        
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            if any(contains_gender(table.get("columns", [])) for table in data.get("tables", [])):
                gender_db_count += 1  
        
        except (json.JSONDecodeError, KeyError) as e:
            print(f"⚠️ Failed to parse {json_file}: {e}")

        # Print progress every 1000 files
        if (i + 1) % 1000 == 0:
            print(f"\n🔹 Processed {i + 1} files, current count: {gender_db_count}")

    return gender_db_count

gender_db_count = count_databases_with_gender(SCHEMA_DIR)

print(f"\n✅ Total databases containing 'Gender' related fields: {gender_db_count}")
