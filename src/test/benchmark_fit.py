import os
import sys
import torch
import time
import json
import tempfile
import pandas as pd
from typing import List, Dict, Any

# Add src to path
sys.path.insert(0, os.path.join(os.getcwd(), "src"))

from model.BGEEmbedder import BGEEmbedder
from model.WKDataset import WKDataset

def create_mock_data(temp_dir):
    schema_dir = os.path.join(temp_dir, "schema")
    csv_dir = os.path.join(temp_dir, "unzip")
    os.makedirs(schema_dir, exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)
    
    # Create mock schemas and CSVs
    db_ids = [str(i).zfill(5) for i in range(10)]
    for db_id in db_ids:
        db_name = f"db_{db_id}"
        table_name = "table1"
        schema = {
            "database_name": db_name,
            "tables": [
                {
                    "table_name": table_name,
                    "file_name": f"{table_name}.csv",
                    "columns": [
                        {"column_name": "col1", "data_type": "text"},
                        {"column_name": "col2", "data_type": "int"}
                    ]
                }
            ]
        }
        with open(os.path.join(schema_dir, f"{db_id}_{db_name}.json"), "w") as f:
            json.dump(schema, f)
            
        full_db_dir = os.path.join(csv_dir, f"{db_id} {db_name}", "tables")
        os.makedirs(full_db_dir, exist_ok=True)
        df = pd.DataFrame({"col1": ["val1", "val2", "val3"], "col2": [1, 2, 3]})
        df.to_csv(os.path.join(full_db_dir, f"{table_name}.csv"), index=False)
        
    # Create mock triplets
    triplets = []
    for i in range(100):
        triplets.append({
            "anchor": "00000",
            "positive": "00001",
            "negatives": ["00002", "00003", "00004"]
        })
        
    train_path = os.path.join(temp_dir, "train.jsonl")
    val_path = os.path.join(temp_dir, "val.jsonl")
    for p in [train_path, val_path]:
        with open(p, "w") as f:
            for t in triplets:
                f.write(json.dumps(t) + "\n")
                
    return schema_dir, csv_dir, train_path, val_path

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        schema_dir, csv_dir, train_path, val_path = create_mock_data(temp_dir)
        
        loader = WKDataset(schema_dir=schema_dir, csv_base_dir=csv_dir)
        embedder = BGEEmbedder(model_type="bge-m3") # Uses default model name if model_path is None
        
        print("\nStarting benchmark...")
        start_time = time.time()
        
        # Run for 1 epoch for benchmarking
        embedder.fit(
            train_path=train_path,
            val_path=val_path,
            loader=loader,
            save_dir=os.path.join(temp_dir, "out"),
            batch_size=8,
            epochs=1,
            lr=1e-5
        )
        
        end_time = time.time()
        duration = end_time - start_time
        print(f"\nBenchmark finished in {duration:.2f} seconds")
        
        if torch.cuda.is_available():
            peak_memory = torch.cuda.max_memory_allocated(device) / 1024**2
            print(f"Peak GPU memory: {peak_memory:.2f} MB")

if __name__ == "__main__":
    main()
