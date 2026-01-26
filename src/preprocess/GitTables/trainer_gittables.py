#!/usr/bin/env python
"""
GitTables Trainer

Trainer for GitTables pre-serialized triplets.
Unlike the main trainer, this doesn't require schema/CSV directories
because triplets are already serialized text.

Usage:
    python -m preprocess.GitTables.trainer_gittables \
        --train-triplets triplets/triplets_train.jsonl \
        --val-triplets triplets/triplets_val.jsonl \
        --output-dir out/model \
        --lr 1e-05 \
        --epochs 10 \
        --batch-size 32
"""

import argparse
import os
import sys
import json


def parse_gpu_arg():
    """Parse --gpu argument early, before importing torch."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--gpu", type=str, default=None,
                        help="GPU device ID(s) to use (e.g., '0', '1', '0,1')")
    args, _ = parser.parse_known_args()
    return args.gpu


# Set GPU before importing torch
gpu_id = parse_gpu_arg()
if gpu_id is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    print(f"Setting CUDA_VISIBLE_DEVICES={gpu_id}")

# Now import torch and other modules
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from model.BGEEmbedder import BGEEmbedder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PreSerializedTripletDataset(Dataset):
    """
    Dataset for pre-serialized triplets (GitTables format).
    
    Each triplet has:
    - anchor: str (already serialized text)
    - positive: str (already serialized text)
    - negatives: List[str] (already serialized texts)
    """
    
    def __init__(self, path: str):
        self.data = []
        print(f"Loading pre-serialized triplets from {path}...")
        with open(path, "r") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    self.data.append(item)
        print(f"Loaded {len(self.data)} triplets")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        # Triplets are already serialized text
        anchor = item["anchor"]
        positive = item["positive"]
        negatives = item["negatives"]
        return anchor, positive, negatives


def info_nce_loss(anchor_embs, pos_embs, neg_embs_flat, num_negs, temperature=0.5):
    """
    Optimized InfoNCE loss for a batch.
    anchor_embs: [B, D]
    pos_embs: [B, D]
    neg_embs_flat: [B * num_negs, D]
    """
    batch_size = anchor_embs.shape[0]
    # Reshape negatives: [B, num_negs, D]
    neg_embs = neg_embs_flat.view(batch_size, num_negs, -1)
    
    # Combine positives and negatives: [B, 1 + num_negs, D]
    pos_neg_embs = torch.cat([pos_embs.unsqueeze(1), neg_embs], dim=1)
    
    # Calculate cosine similarities: [B, 1 + num_negs]
    sims = torch.bmm(anchor_embs.unsqueeze(1), pos_neg_embs.transpose(1, 2)).squeeze(1) / temperature
    
    # Labels are always 0 (positive at index 0)
    labels = torch.zeros(batch_size, dtype=torch.long, device=anchor_embs.device)
    loss = F.cross_entropy(sims, labels)
    return loss


def train_gittables(
    train_path: str,
    val_path: str,
    output_dir: str,
    model_type: str = "bge-m3",
    base_model_path: str = None,
    batch_size: int = 32,
    epochs: int = 10,
    lr: float = 1e-5,
    temperature: float = 0.5,
    max_length: int = 1024,
):
    """
    Train embedding model on pre-serialized GitTables triplets.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("GitTables Model Training (Pre-serialized Triplets)")
    print("=" * 60)
    print(f"Training triplets: {train_path}")
    print(f"Validation triplets: {val_path}")
    print(f"Output directory: {output_dir}")
    print(f"Model type: {model_type}")
    print(f"Learning rate: {lr}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Temperature: {temperature}")
    print("=" * 60)
    
    # Initialize embedder
    print("\nInitializing embedder...")
    embedder = BGEEmbedder(
        model_type=model_type,
        model_path=base_model_path
    )
    
    model = embedder.model
    tokenizer = embedder.tokenizer
    model.train()
    
    # Load datasets
    train_set = PreSerializedTripletDataset(train_path)
    val_set = PreSerializedTripletDataset(val_path)
    
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        collate_fn=lambda x: x, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False,
        collate_fn=lambda x: x, num_workers=4, pin_memory=True
    )
    
    optimizer = AdamW(model.parameters(), lr=lr)
    best_val_loss = float("inf")
    
    def evaluate():
        model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                all_anchors = [item[0] for item in batch]
                all_positives = [item[1] for item in batch]
                all_negatives = [neg for item in batch for neg in item[2]]
                num_negs = len(batch[0][2])
                
                all_texts = all_anchors + all_positives + all_negatives
                
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    embs = embedder.get_embedding(all_texts, batch_size=len(all_texts))
                    
                    b_size = len(batch)
                    anchor_embs = embs[:b_size]
                    pos_embs = embs[b_size:2*b_size]
                    neg_embs_flat = embs[2*b_size:]
                    
                    loss = info_nce_loss(anchor_embs, pos_embs, neg_embs_flat, num_negs, temperature)
                total_loss += loss.item()
        model.train()
        return total_loss / len(val_loader)
    
    # Training loop
    for epoch in range(epochs):
        total_loss = 0.0
        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch in progress:
            optimizer.zero_grad()
            
            all_anchors = [item[0] for item in batch]
            all_positives = [item[1] for item in batch]
            all_negatives = [neg for item in batch for neg in item[2]]
            num_negs = len(batch[0][2])
            
            all_texts = all_anchors + all_positives + all_negatives
            
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                embs = embedder.get_embedding(all_texts, is_train=True, batch_size=len(all_texts))
                
                b_size = len(batch)
                anchor_embs = embs[:b_size]
                pos_embs = embs[b_size:2*b_size]
                neg_embs_flat = embs[2*b_size:]
                
                loss = info_nce_loss(anchor_embs, pos_embs, neg_embs_flat, num_negs, temperature)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress.set_postfix(loss=loss.item())
        
        avg_train_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} train loss: {avg_train_loss:.4f}")
        
        val_loss = evaluate()
        print(f"Epoch {epoch+1} val loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs(os.path.join(output_dir, "best"), exist_ok=True)
            if tokenizer is not None:
                model.save_pretrained(os.path.join(output_dir, "best"))
                tokenizer.save_pretrained(os.path.join(output_dir, "best"))
            else:
                model.save(os.path.join(output_dir, "best"))
            print(f"💾 Saved best model to {output_dir}/best")
        
        os.makedirs(os.path.join(output_dir, "last"), exist_ok=True)
        if tokenizer is not None:
            model.save_pretrained(os.path.join(output_dir, "last"))
            tokenizer.save_pretrained(os.path.join(output_dir, "last"))
        else:
            model.save(os.path.join(output_dir, "last"))
        print(f"📝 Saved last model to {output_dir}/last")
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved to: {output_dir}")
    print("=" * 60)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train embedding model on pre-serialized GitTables triplets"
    )
    
    parser.add_argument("--train-triplets", type=str, required=True,
                        help="Path to training triplets JSONL file")
    parser.add_argument("--val-triplets", type=str, required=True,
                        help="Path to validation triplets JSONL file")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for trained model")
    parser.add_argument("--model-type", type=str, default="bge-m3",
                        help="Model type (default: bge-m3)")
    parser.add_argument("--base-model-path", type=str, default=None,
                        help="Path to base model")
    parser.add_argument("--lr", type=float, default=1e-05,
                        help="Learning rate (default: 1e-05)")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of epochs (default: 10)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size (default: 32)")
    parser.add_argument("--temperature", type=float, default=0.5,
                        help="Temperature for InfoNCE loss (default: 0.5)")
    parser.add_argument("--max-length", type=int, default=1024,
                        help="Maximum sequence length (default: 1024)")
    parser.add_argument("--gpu", type=str, default=None,
                        help="GPU device ID")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    if not os.path.exists(args.train_triplets):
        print(f"Error: Training triplets not found: {args.train_triplets}")
        sys.exit(1)
    if not os.path.exists(args.val_triplets):
        print(f"Error: Validation triplets not found: {args.val_triplets}")
        sys.exit(1)
    
    train_gittables(
        train_path=args.train_triplets,
        val_path=args.val_triplets,
        output_dir=args.output_dir,
        model_type=args.model_type,
        base_model_path=args.base_model_path,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        temperature=args.temperature,
        max_length=args.max_length,
    )


if __name__ == "__main__":
    main()
