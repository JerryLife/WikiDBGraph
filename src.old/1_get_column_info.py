"""
Path: data/schema/<filename>.json
"""
import os

from tqdm import tqdm
import ujson as json


def extract_schema_info(schema_data, database_file):
    database_name = schema_data["database_name"]
    tables = schema_data["tables"]
    
    # Extract all unique column names
    unique_columns = set()
    for table in tables:
        for column in table["columns"]:
            unique_columns.add(column["column_name"])
    
    # Convert to sorted list for consistent output
    column_list = sorted(list(unique_columns))
    
    # Create output string
    output_parts = [
        database_file.replace(".json", ""),
        str(len(tables)),
        str(len(unique_columns))
    ]
    output_parts.extend(column_list)
    
    return ",".join(output_parts)



def process_schema_file(schema_dir, output_filepath="data/column_stats.txt"):
    if os.path.exists(output_filepath):
        respond = input(f"File {output_filepath} already exists. Do you want to overwrite it? (y/N) ")
        if respond != "y":
            return
    
    with open(output_filepath, 'w') as outf:
        for database_file in tqdm(os.listdir(schema_dir)):
            schema_file = os.path.join(schema_dir, database_file)
            with open(schema_file, 'r') as f:
                try:
                    schema_data = json.load(f)
                    output = extract_schema_info(schema_data, os.path.basename(database_file))
                    
                    # Write to output file
                    outf.write(output + '\n')
                except json.JSONDecodeError:
                    print(f"Warning: {schema_file} is not a valid JSON file or is empty.")
                    return

if __name__ == "__main__":
    process_schema_file("data/schema", "data/column_stats.txt")
