"""
Schema serialization with ablation modes for embedding model training.

Supports three modes for ablation studies:
- schema_only: Table and column names only
- data_only: Only representative sample values
- full: Combined format (default, backward-compatible)
"""

import os
import sys
import pandas as pd
from typing import Dict, Any, Optional, List, Literal

# Allow imports from parent directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from model.WKDataset import WKDataset


class SchemaSerializer:
    """
    Serialize database schemas to text for embedding generation.
    
    Supports multiple serialization modes for ablation studies comparing
    schema-only, data-only, and full (combined) representations.
    
    Args:
        mode: Serialization mode
            - "schema_only": Names of tables and columns only
            - "data_only": Only the representative sample values
            - "full": Current combined format (default)
        sample_size: Number of representative values per column (default: 3)
        show_wikidata_property_id: Whether to include Wikidata property IDs
    
    Example:
        >>> serializer = SchemaSerializer(mode="full", sample_size=3)
        >>> loader = WKDataset(schema_dir="data/schema", csv_base_dir="data/unzip")
        >>> text = serializer.serialize(loader, "00001")
    """
    
    def __init__(
        self,
        mode: Literal["schema_only", "data_only", "full"] = "full",
        sample_size: int = 3,
        show_wikidata_property_id: bool = False
    ):
        """Initialize the schema serializer."""
        if mode not in ["schema_only", "data_only", "full"]:
            raise ValueError(
                f"Invalid mode: {mode}. Must be one of: schema_only, data_only, full"
            )
        self.mode = mode
        self.sample_size = sample_size
        self.show_wikidata_property_id = show_wikidata_property_id
    
    def serialize(
        self,
        loader: WKDataset,
        db_id: str,
        sample: bool = True
    ) -> str:
        """
        Serialize a database schema to text.
        
        Args:
            loader: WKDataset instance for loading schema and data
            db_id: Database ID to serialize
            sample: Whether to include sample values (ignored for schema_only mode)
        
        Returns:
            Text representation of the database schema
        """
        if self.mode == "schema_only":
            return self._serialize_schema_only(loader, db_id)
        elif self.mode == "data_only":
            return self._serialize_data_only(loader, db_id)
        else:  # full mode
            return self._serialize_full(loader, db_id, sample)
    
    def _serialize_schema_only(self, loader: WKDataset, db_id: str) -> str:
        """
        Serialize schema with table and column names only.
        
        Format:
            Database Name: <db_name>
            Table: <table_name>
             - Column: <col_name>
        """
        schema = loader.load_database(db_id)
        db_name = schema.get("database_name", db_id)
        
        lines = [f"Database Name: {db_name}"]
        for table in schema.get("tables", []):
            table_name = table["table_name"]
            lines.append(f"Table: {table_name}")
            for col in table["columns"]:
                col_name = col["column_name"]
                if self.show_wikidata_property_id:
                    wikidata_property_id = col.get("wikidata_property_id") or "Null"
                    lines.append(f" - Column: {col_name} ; Wikidata property ID: {wikidata_property_id}")
                else:
                    lines.append(f" - Column: {col_name}")
        
        return "\n".join(lines)
    
    def _serialize_data_only(self, loader: WKDataset, db_id: str) -> str:
        """
        Serialize with only representative sample values.
        
        Format:
            Database Name: <db_name>
            Table: <table_name>
             - Samples: [val1|val2|val3]
        """
        schema = loader.load_database(db_id)
        db_name = schema.get("database_name", db_id)
        tables = loader.load_csv_data(db_id=db_id, sample=True, sample_size=self.sample_size)
        
        lines = [f"Database Name: {db_name}"]
        for table in schema.get("tables", []):
            table_name = table["table_name"]
            lines.append(f"Table: {table_name}")
            df = tables.get(table_name, pd.DataFrame())
            for col in table["columns"]:
                col_name = col["column_name"]
                values = df[col_name].dropna().astype(str).unique().tolist()[:self.sample_size] if col_name in df.columns else []
                sample_str = "|".join(values) if values else "N/A"
                lines.append(f" - Samples: [{sample_str}]")
        
        return "\n".join(lines)
    
    def _serialize_full(
        self,
        loader: WKDataset,
        db_id: str,
        sample: bool = True
    ) -> str:
        """
        Serialize with full schema and data (default mode).
        
        This matches the existing format_schema_from_loader() function
        for backward compatibility with existing finetuned models.
        
        Format:
            Database Name: <db_name>
            Table: <table_name>
             - <col_name>, Samples: [val1|val2|val3]
        """
        schema = loader.load_database(db_id)
        db_name = schema.get("database_name", db_id)
        tables = loader.load_csv_data(db_id=db_id, sample=sample, sample_size=self.sample_size)
        
        lines = [f"Database Name: {db_name}"]
        for table in schema.get("tables", []):
            table_name = table["table_name"]
            lines.append(f"Table: {table_name}")
            df = tables.get(table_name, pd.DataFrame())
            for col in table["columns"]:
                col_name = col["column_name"]
                values = df[col_name].dropna().astype(str).unique().tolist()[:self.sample_size] if col_name in df.columns else []
                sample_str = "|".join(values) if values else "N/A"
                if self.show_wikidata_property_id:
                    wikidata_property_id = col.get("wikidata_property_id") or "Null"
                    lines.append(f" - Column: {col_name} ; Wikidata property ID: {wikidata_property_id} ; Samples: {sample_str}")
                else:
                    lines.append(f" - {col_name}, Samples: [{sample_str}]")
        
        return "\n".join(lines)
    
    @classmethod
    def from_config(cls, config: "PreprocessConfig") -> "SchemaSerializer":
        """Create a SchemaSerializer from a PreprocessConfig."""
        from .config import PreprocessConfig
        return cls(
            mode=config.serialization_mode,
            sample_size=config.sample_size,
            show_wikidata_property_id=config.show_wikidata_property_id
        )


# Backward-compatible function to match existing API
def format_schema_from_loader(
    loader: WKDataset,
    db_id: str,
    sample: bool = True,
    sample_size: int = 3,
    show_wikidata_property_id: bool = False,
    only_show_column_name: bool = False,
    only_show_table_name: bool = False,
    mode: Literal["schema_only", "data_only", "full"] = "full"
) -> str:
    """
    Format a database schema from a WKDataset loader.
    
    Backward-compatible wrapper around SchemaSerializer.
    
    Args:
        loader: WKDataset instance
        db_id: Database ID
        sample: Whether to include sample values
        sample_size: Number of sample values per column
        show_wikidata_property_id: Whether to show Wikidata property IDs
        only_show_column_name: If True, only show column names (maps to schema_only)
        only_show_table_name: If True, only show table names
        mode: Serialization mode (overrides only_show_* flags if set)
    
    Returns:
        Formatted schema text
    """
    # Handle legacy flags
    if only_show_table_name:
        # Special case: only table names
        schema = loader.load_database(db_id)
        db_name = schema.get("database_name", db_id)
        lines = [f"Database Name: {db_name}"]
        for table in schema.get("tables", []):
            lines.append(f"Table: {table['table_name']}")
        return "\n".join(lines)
    
    if only_show_column_name:
        mode = "schema_only"
    
    serializer = SchemaSerializer(
        mode=mode,
        sample_size=sample_size,
        show_wikidata_property_id=show_wikidata_property_id
    )
    return serializer.serialize(loader, db_id, sample=sample)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Serialize database schema to text")
    parser.add_argument("--db-id", type=str, default="00000", help="Database ID")
    parser.add_argument("--mode", type=str, default="full",
                        choices=["schema_only", "data_only", "full"],
                        help="Serialization mode")
    parser.add_argument("--sample-size", type=int, default=3, help="Sample size per column")
    parser.add_argument("--schema-dir", type=str, default="data/schema", help="Schema directory")
    parser.add_argument("--csv-dir", type=str, default="data/unzip", help="CSV base directory")
    
    args = parser.parse_args()
    
    loader = WKDataset(schema_dir=args.schema_dir, csv_base_dir=args.csv_dir)
    serializer = SchemaSerializer(mode=args.mode, sample_size=args.sample_size)
    
    print(f"=== Mode: {args.mode} ===")
    print(serializer.serialize(loader, args.db_id))
