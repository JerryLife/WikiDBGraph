# src/model/bge_m3_similarity.py
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from utils.schema_formatter import format_schema_from_loader, format_schema_from_dict, format_column_from_dict
from model.WKDataset import WKDataset
import pandas as pd
import time
import os

model_name = "BAAI/bge-m3"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def get_embedding(texts, batch_size=64) -> torch.Tensor:
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]

        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            embedding = outputs.last_hidden_state[:, 0]  # CLS token
            embedding = F.normalize(embedding, p=2, dim=-1)
            all_embeddings.append(embedding)

    return torch.cat(all_embeddings, dim=0)

def bge_m3_database_similarity(loader, db_id_1: str, db_id_2: str, show_wikidata_property_id: bool = False, sample: bool = True) -> float:

    start_time = time.time()
    schema_1 = [format_schema_from_loader(loader, db_id_1, sample=sample, show_wikidata_property_id=show_wikidata_property_id)]
    schema_2 = [format_schema_from_loader(loader, db_id_2, sample=sample, show_wikidata_property_id=show_wikidata_property_id)]

    emb1 = get_embedding(schema_1)
    emb2 = get_embedding(schema_2)

    similarity = F.cosine_similarity(emb1, emb2).item()
    print(f"Similarity between {db_id_1} and {db_id_2}: {similarity:.4f}")
    elapsed = time.time() - start_time
    matched_pair=("_", "_")
    return similarity, matched_pair, elapsed

if __name__ == "__main__":
    import time
    db_id_1 = "78145"
    db_id_2 = "95960"

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    start_time = time.time()
    score = bge_m3_database_similarity(db_id_1, db_id_2)
    end_time = time.time()
    print(f"Time taken: {end_time - start_time:.2f} seconds")
    peak_memory = torch.cuda.max_memory_allocated(device="cuda")
    print(f"peak memory: {peak_memory/1e9} GB")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    start_time = time.time()
    random_score = bge_m3_database_similarity("00024", "62098")
    end_time = time.time()
    print(f"Time taken: {end_time - start_time:.2f} seconds")
    peak_memory = torch.cuda.max_memory_allocated(device="cuda")
    print(f"peak memory: {peak_memory/1e9} GB")
