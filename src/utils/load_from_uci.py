import os
import pandas as pd
def parse_columns_from_names(names_path):
    columns = []

    with open(names_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                columns.append(line)

    print(f"✅ Parsed {len(columns)} columns from {names_path}")
    return columns

def load_wdbc_dataset(data_path="wdbc.data", names_path="wdbc.names"):
    columns = parse_columns_from_names(names_path)
    df = pd.read_csv(data_path, header=None, names=columns)

    return df

def format_schema_from_dataframe(df, db_name: str = "breast_cancer", sample: bool = True, sample_size: int = 3) -> str:
    lines = [f"DB: {db_name}"]

    table_name = f"{db_name}_table"  # 默认表名
    lines.append(f"Table: {table_name}")

    for col in df.columns:
        # Skip ID column if you want (optional)
        if col == "id":
            continue

        data_type = str(df[col].dtype)
        values = df[col].dropna().astype(str).tolist()

        if sample and values:
            sampled_values = values[:sample_size]
        else:
            sampled_values = values

        sample_str = "| ".join(sampled_values) if sampled_values else "N/A"

        lines.append(f" - Column: {col} ; Data type: {data_type} ; Samples: {sample_str}")

    content = "\n".join(lines)

    return content

if __name__ == "__main__":
    dataset_dir = "/hpctmp/e1351271/wkdbs/data/uci_datasets/breast+cancer+wisconsin+diagnostic"
    data_path = os.path.join(dataset_dir, "wdbc.data")
    names_path = os.path.join(dataset_dir, "wdbc.names")
    df = load_wdbc_dataset(data_path, names_path)
    schema = format_schema_from_dataframe(df)
    print(schema)
