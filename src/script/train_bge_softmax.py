# train_bge_softmax.py

import json
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from tqdm import tqdm
import os

from model.BGEEmbedder import BGEEmbedder
from model.WKDataset import WKDataset

if __name__ == "__main__":
    loader = WKDataset(schema_dir="../data/schema", csv_base_dir="../data/unzip")
    lr = 1e-5
    embedder = BGEEmbedder(model_type="bge-m3")
    embedder.fit(
        train_path="../data/split_triplets/triplets_train.jsonl",
        val_path="../data/split_triplets/triplets_val.jsonl",
        loader=loader,
        save_dir=f"../out/col_matcher_bge-m3_lr{lr}_ft_database/weights/finetuned_bge_m3_softmax_lr{lr}",
        lr=lr,
    )