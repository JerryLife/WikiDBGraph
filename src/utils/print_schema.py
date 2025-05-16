import json
import os
import glob
def extract_schema_info(schema_file):
    """
    Parse the schema.json file and extract all table names and column names.
    :param schema_file: Path to the JSON file
    :return: A dictionary containing table names as keys and column names as values
    """
    with open(schema_file, "r", encoding="utf-8") as file:
        schema = json.load(file)

    schema_info = {}
    print(schema.get("wikidata_topic_item_id"))
    print(schema.get("wikidata_topic_item_label"))
    for table in schema.get("tables", []):
        table_name = table.get("table_name", "UNKNOWN_TABLE")  # Get table name
        column_names = [col["column_name"] for col in table.get("columns", [])]  # Extract column names
        schema_info[table_name] = column_names  # Store in dictionary

    return schema_info

def display_schema_info(schema_info):
    """
    Print the extracted schema information in a structured format.
    """
    for table, columns in schema_info.items():
        print(f"    ├── 📂 Table: {table.lower()}")
        for col in columns:
            print(f"    |    ├── {col.lower()}")
        print("")  # Add a blank line for readability

def print_schema(dbs_name):
    # Example usage
    print(f"{dbs_name.lower()}")

    schema_dir = "/hpctmp/e1351271/wkdbs/data/schema"
    pattern = os.path.join(schema_dir, f"{dbs_name}*.json")
    matched_files = glob.glob(pattern)

    if not matched_files:
        print(f"⚠️ No schema files found with prefix '{dbs_name}'")
        return

    schema_path = matched_files[0]
    print(f"📂 Matched schema file: {os.path.basename(schema_path)}")

    schema_info = extract_schema_info(schema_path)
    display_schema_info(schema_info)

if __name__ == "__main__":
    print_schema("56747")
    # print_schema("62771")