# src/utils/schema_formatter.py
import sys
import os
sys.path.append("/hpctmp/e1351271/wkdbs/src")
import pandas as pd
from model.WKDataset import WKDataset

def format_column_from_dict(column: dict) -> str:
    table_name, col_name = column["column_name"].split("::")
    data_type = column["data_type"]
    values = column["values"]
    sample_str = "| ".join(values) if values else "N/A"
    return f"Table: {table_name} ; Column: {col_name} ; Data type: {data_type} ; Samples: {sample_str}"

def format_schema_from_dict(schema: dict, db_id: str, sample: bool = True, sample_size: int = 3, show_wikidata_property_id: bool = True, only_show_column_name: bool = False) -> str:
    db_name = schema.get("database_name", db_id)
    tables = schema.get("tables", [])

    lines = [f"DB: {db_id} ({db_name})"]
    for table in tables:
        table_name = table["table_name"]
        lines.append(f"Table: {table_name}")
        for col in table["columns"]:
            col_name = col["column_name"]
            if only_show_column_name:
                lines.append(f" - Column: {col_name}")
                continue
            data_type = col["data_type"]
            values = col["values"]
            sample_str = "| ".join(values) if values else "N/A"
            if show_wikidata_property_id:
                wikidata_property_id = col["wikidata_property_id"]
                if wikidata_property_id is None:
                    wikidata_property_id = "Null"
                lines.append(f" - Column: {col_name} ; Data type: {data_type} ; Wikidata property ID: {wikidata_property_id} ; Samples: {sample_str}")
            else:
                lines.append(f" - Column: {col_name} ; Data type: {data_type} ; Samples: {sample_str}")

    content = "\n".join(lines)
    # with open(f"/hpctmp/e1351271/wkdbs/src/model/schema_{db_id}.txt", "w") as f:
    #     f.write(content)

    return content

def format_schema_from_loader(loader, db_id: str, sample: bool = True, sample_size: int = 3, show_wikidata_property_id: bool = False, only_show_column_name: bool = False, only_show_table_name: bool = False) -> str:
    schema = loader.load_database(db_id)
    db_name = schema.get("database_name", db_id)
    tables = loader.load_csv_data(db_id=db_id, sample=sample, sample_size=sample_size)

    lines = [f"Database Name: {db_name}"]
    for table in schema.get("tables", []):
        table_name = table["table_name"]
        lines.append(f"Table: {table_name}")
        if only_show_table_name:
            continue
        df = tables.get(table_name, pd.DataFrame())
        for col in table["columns"]:
            col_name = col["column_name"]
            if only_show_column_name:
                lines.append(f" - Column: {col_name}")
                continue
            # data_type = col["data_type"]
            values = df[col_name].dropna().astype(str).unique().tolist()[:sample_size] if col_name in df.columns else []
            sample_str = "|".join(values) if values else "N/A"
            if show_wikidata_property_id:
                wikidata_property_id = col["wikidata_property_id"]
                if wikidata_property_id is None:
                    wikidata_property_id = "Null"
                lines.append(f" - Column: {col_name} ; Wikidata property ID: {wikidata_property_id} ; Samples: {sample_str}")
            else:
                lines.append(f" - {col_name}, Samples: [{sample_str}]")

    content = "\n".join(lines)  
    # with open(f"/hpctmp/e1351271/wkdbs/src/model/schema_{db_id}.txt", "w") as f:
    #     f.write(content)

    return content

if __name__ == "__main__":

    loader = WKDataset(schema_dir="../data/schema", csv_base_dir="../data/unzip")
    print(format_schema_from_loader(loader, "78145", show_wikidata_property_id=False))