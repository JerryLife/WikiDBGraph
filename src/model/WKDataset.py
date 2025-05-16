import os
import json
import random
import pandas as pd
from typing import List, Dict, Any, Optional


class WKDataset:
    def __init__(self, schema_dir: str = "../data/schema", csv_base_dir: str = "../data/unzip", schema_column_count_file: str = "../data/schema_column_counts.csv"):
        self.schema_dir = schema_dir
        self.csv_base_dir = csv_base_dir
        self.schemas: Dict[str, Dict[str, Any]] = {}  # key: db_id, value: schema dict
        self.dataframes: Dict[str, Dict[str, pd.DataFrame]] = {}  # db_id -> {table_name: dataframe}
        self.schema_column_count_file = schema_column_count_file
        
    def _get_schema_path(self, db_id: str) -> str:
        """Return the full path to the schema JSON file for the given db_id."""
        matches = [f for f in os.listdir(self.schema_dir) if f.startswith(db_id + "_")]
        if not matches:
            raise FileNotFoundError(f"No schema file found for db_id: {db_id}")
        return os.path.join(self.schema_dir, matches[0])

    def _get_db_folder_name(self, db_id: str) -> str:
        """Extract the full folder name (id + name) from the schema filename."""
        schema_path = self._get_schema_path(db_id)
        return os.path.splitext(os.path.basename(schema_path))[0]

    def load_database(self, db_id: str) -> Dict[str, Any]:
        """Load the schema JSON file for a single database."""
        if db_id in self.schemas:
            return self.schemas[db_id]
        schema_path = self._get_schema_path(db_id)
        with open(schema_path, "r", encoding="utf-8") as f:
            schema = json.load(f)
        self.schemas[db_id] = schema
        return schema

    def load_databases_from_txt(self, txt_path: str) -> Dict[str, Dict[str, Any]]:
        """Load schemas for all database IDs listed in a text file (one ID per line)."""
        with open(txt_path, "r", encoding="utf-8") as f:
            db_ids = [line.strip() for line in f if line.strip()]
        for db_id in db_ids:
            self.load_database(db_id)
        return {db_id: self.schemas[db_id] for db_id in db_ids}

    def sample_databases(self, k: int, seed: Optional[int] = None) -> List[str]:
        """Randomly sample k database IDs from the schema directory."""
        if seed is not None:
            random.seed(seed)
        all_ids = [f.split("_")[0] for f in os.listdir(self.schema_dir) if f.endswith(".json")]
        return random.sample(all_ids, k)

    def save_sampled_ids(self, output_txt_path: str, k: int, seed: Optional[int] = None) -> List[str]:
        """Sample k database IDs and save them to a text file."""
        sampled_ids = self.sample_databases(k, seed)
        with open(output_txt_path, "w", encoding="utf-8") as f:
            for db_id in sampled_ids:
                f.write(db_id + "\n")
        return sampled_ids

    def get_schema(self, db_id: str) -> Dict[str, Any]:
        """Return the schema dictionary for a given db_id (must be loaded first)."""
        if db_id not in self.schemas:
            raise ValueError(f"Database {db_id} not loaded.")
        return self.schemas[db_id]

    def get_all_loaded(self) -> Dict[str, Dict[str, Any]]:
        """Return all loaded schemas."""
        return self.schemas

    def get_table_names(self, db_id: str) -> List[str]:
        """Return a list of table names for a given database."""
        schema = self.get_schema(db_id)
        return [table["table_name"] for table in schema.get("tables", [])]

    def get_column_names(self, db_id: str, table_name: str) -> List[str]:
        """Return a list of column names for a given table in a database."""
        schema = self.get_schema(db_id)
        for table in schema.get("tables", []):
            if table["table_name"] == table_name:
                return [col["column_name"] for col in table.get("columns", [])]
        raise ValueError(f"Table {table_name} not found in database {db_id}.")
    
    def get_column_count_by_db_id(self, db_id: str) -> int:
        """Return the number of columns for a given database."""
        df = pd.read_csv(self.schema_column_count_file)
        return df[df["db_id"].str.split("_").str[0] == db_id]["num_columns"].values[0]

    def load_csv_data(
        self,
        db_id: str,
        sample: bool = False,
        sample_size: int = 100,
        flatten_columns: bool = False,
        using_database_title: bool = False
    ) -> Dict[str, pd.DataFrame]:
        """
        Load all table CSV files for a database.

        Args:
            db_id: ID of the database to load.
            sample: Whether to sample rows from each table.
            sample_size: Number of rows to sample per table.
            flatten_columns: Whether to prefix each column name with its table name.
        Returns:
            A dict mapping table names to pandas DataFrames.
        """
        schema = self.load_database(db_id)
        db_folder_name = self._get_db_folder_name(db_id)
        db_name = db_folder_name[6:]
        db_folder_name = db_folder_name[:5] + ' ' + db_name
        db_path = os.path.join(self.csv_base_dir, db_folder_name, "tables")

        if db_id not in self.dataframes:
            self.dataframes[db_id] = {}

        for table in schema.get("tables", []):
            file_name = table["file_name"]
            table_name = table["table_name"]
            file_path = os.path.join(db_path, file_name)
            if not os.path.exists(file_path):
                print(f"[Warning] CSV file not found: {file_path}")
                continue
            df = pd.read_csv(file_path)
            if sample and len(df) > sample_size:
                df = df.sample(n=sample_size, random_state=42)
            if flatten_columns:
                df.columns = [f"{table_name}::{col}" for col in df.columns]
                if using_database_title:
                    df.columns = [f"{db_name}::{col}" for col in df.columns]
            self.dataframes[db_id][table_name] = df

        return self.dataframes[db_id]

    def get_table_data(self, db_id: str, table_name: str) -> pd.DataFrame:
        """Return the DataFrame for a specific table (must call load_csv_data first)."""
        if db_id not in self.dataframes or table_name not in self.dataframes[db_id]:
            raise ValueError(f"Data for table {table_name} in database {db_id} not loaded.")
        return self.dataframes[db_id][table_name]


# Example usage
if __name__ == "__main__":
    wk = WKDataset(schema_dir="data/schema", csv_base_dir="data/unzip")

    # Load schema
    schema = wk.load_database("00000")

    # Load CSV data (optionally with sampling)
    table_dfs = wk.load_csv_data("00000", sample=True, sample_size=5)
    for tname, df in table_dfs.items():
        print(f"\nTable: {tname}")
        print(df.head())
