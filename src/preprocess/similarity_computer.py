"""
All-pairs similarity computation using precomputed embeddings.

Computes cosine similarity between all database pairs and saves results.
Uses vectorized GPU operations for efficiency on large datasets.
"""

import os
import sys
import csv
import torch
import torch.nn.functional as F
from tqdm import tqdm
from typing import Optional, Set, Tuple, List, Dict


def set_gpu_device(gpu_id: str):
    """Set GPU device via CUDA_VISIBLE_DEVICES environment variable."""
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    print(f"Setting CUDA_VISIBLE_DEVICES={gpu_id}")


class SimilarityComputer:
    """
    Compute cosine similarity between all database pairs.
    
    Uses precomputed embeddings to efficiently compute all-pairs similarity
    with vectorized GPU operations. Filters by threshold on GPU to avoid
    storing billions of pairs in memory.
    
    Args:
        threshold: Similarity threshold for edge prediction (default: 0.6713)
        chunk_size: Number of rows to process per GPU batch (default: 1024)
        use_fp16: Use float16 for 2x speedup and 50% less VRAM (default: True)
    
    Example:
        >>> computer = SimilarityComputer(threshold=0.6713)
        >>> computer.compute_all_pairs(
        ...     embedding_path="data/graph/all_embeddings.pt",
        ...     output_path="data/graph/all_predictions.pt"
        ... )
    """
    
    def __init__(
        self,
        threshold: float = 0.6713,
        chunk_size: int = 1024,
        use_fp16: bool = True
    ):
        """Initialize the similarity computer."""
        self.threshold = threshold
        self.chunk_size = chunk_size
        self.use_fp16 = use_fp16
    
    def load_qid_pairs(self, qid_pairs_path: str) -> Set[Tuple[str, str]]:
        """
        Load ground truth QID pairs.
        
        Args:
            qid_pairs_path: Path to TSV file with ground truth pairs
        
        Returns:
            Set of (db_1, db_2) tuples (sorted order)
        """
        qid_pairs = set()
        with open(qid_pairs_path, "r") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                a, b = row["db_1"], row["db_2"]
                qid_pairs.add(tuple(sorted((a, b))))
        print(f"Loaded {len(qid_pairs):,} labeled pairs from {qid_pairs_path}")
        return qid_pairs
    
    def compute_all_pairs(
        self,
        embedding_path: str,
        output_path: str,
        qid_pairs_path: Optional[str] = None,
        chunk_size: Optional[int] = None
    ) -> str:
        """
        Compute similarity for all database pairs using vectorized GPU operations.
        
        Uses fast matrix multiplication and threshold filtering on GPU to avoid
        storing billions of pairs. Only pairs exceeding threshold are saved.
        
        Args:
            embedding_path: Path to precomputed embeddings (.pt file)
            output_path: Output path for predictions (.pt file)
            qid_pairs_path: Optional path to ground truth pairs (for post-hoc evaluation)
            chunk_size: Override default chunk size
        
        Returns:
            Path to the saved predictions file
        """
        if chunk_size is None:
            chunk_size = self.chunk_size
            
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        # Load embeddings
        print(f"Loading embeddings from {embedding_path}")
        saved_data = torch.load(embedding_path, map_location=device, weights_only=False)
        db_id_to_index = saved_data["db_id_to_index"]
        raw_embeddings = saved_data["embeddings"].to(device)
        
        # Sort db_ids and reorder embeddings for consistent indexing
        sorted_db_ids = sorted(db_id_to_index.keys())
        num_ids = len(sorted_db_ids)
        total_possible = num_ids * (num_ids - 1) // 2
        
        print(f"Total unique IDs: {num_ids:,}")
        print(f"Total unordered pairs C({num_ids},2): {total_possible:,}")
        print(f"Similarity threshold: {self.threshold}")
        print(f"Chunk size: {chunk_size}")
        print(f"Using FP16: {self.use_fp16}")
        
        # Reorder embeddings to match sorted db_id order
        print("Reordering and normalizing embeddings...")
        perm_idx = torch.tensor([db_id_to_index[db_id] for db_id in sorted_db_ids], device=device)
        embeddings = raw_embeddings[perm_idx]
        del raw_embeddings  # Free memory
        
        # Normalize and optionally cast to half precision
        embeddings = F.normalize(embeddings, p=2, dim=1)
        if self.use_fp16:
            embeddings = embeddings.half()
        
        # Precompute transpose for matmul
        embeddings_T = embeddings.T  # [dim, num_ids]
        
        # Accumulate results as tensors (much more efficient than Python lists)
        all_sources: List[torch.Tensor] = []
        all_targets: List[torch.Tensor] = []
        all_scores: List[torch.Tensor] = []
        
        print("Computing all-pairs similarity (vectorized)...")
        with torch.no_grad():
            for chunk_start in tqdm(range(0, num_ids, chunk_size), desc="Computing similarities"):
                chunk_end = min(chunk_start + chunk_size, num_ids)
                
                # Get chunk embeddings: [chunk_size, dim]
                chunk_embeddings = embeddings[chunk_start:chunk_end]
                
                # Compute similarity matrix: [chunk_size, num_ids]
                sim_matrix = torch.matmul(chunk_embeddings, embeddings_T)
                
                # Vectorized filtering: upper triangle (j > i) AND above threshold
                # Create row indices for this chunk: [chunk_size, 1]
                row_indices = torch.arange(chunk_start, chunk_end, device=device).unsqueeze(1)
                # Create col indices: [1, num_ids]
                col_indices = torch.arange(num_ids, device=device).unsqueeze(0)
                
                # Mask: keep only upper triangle AND values > threshold
                mask = (col_indices > row_indices) & (sim_matrix > self.threshold)
                
                # Extract indices where mask is True
                valid_indices = torch.nonzero(mask, as_tuple=True)
                local_rows, valid_cols = valid_indices
                
                if len(local_rows) == 0:
                    continue
                
                # Get similarity values
                valid_scores = sim_matrix[local_rows, valid_cols]
                
                # Convert local rows to global rows
                valid_rows = local_rows + chunk_start
                
                # Move to CPU to free GPU memory (store as int32 for efficiency)
                all_sources.append(valid_rows.int().cpu())
                all_targets.append(valid_cols.int().cpu())
                all_scores.append(valid_scores.float().cpu())
        
        # Concatenate all results
        print("Concatenating results...")
        if all_sources:
            final_sources = torch.cat(all_sources)
            final_targets = torch.cat(all_targets)
            final_scores = torch.cat(all_scores)
        else:
            final_sources = torch.tensor([], dtype=torch.int32)
            final_targets = torch.tensor([], dtype=torch.int32)
            final_scores = torch.tensor([], dtype=torch.float32)
        
        num_edges = len(final_scores)
        print(f"Found {num_edges:,} pairs above threshold {self.threshold}")
        
        # Save efficient tensor format
        output_data = {
            "sources": final_sources,      # int32 tensor of source indices
            "targets": final_targets,      # int32 tensor of target indices  
            "scores": final_scores,        # float32 tensor of similarity scores
            "sorted_db_ids": sorted_db_ids,  # list of db_id strings for index->id mapping
            "threshold": self.threshold,
            "num_nodes": num_ids,
        }
        
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        torch.save(output_data, output_path)
        
        # Save summary
        summary_path = output_path.replace(".pt", "_summary.txt")
        with open(summary_path, "w") as f:
            f.write(f"Total nodes: {num_ids:,}\n")
            f.write(f"Total possible pairs C(n,2): {total_possible:,}\n")
            f.write(f"Threshold: {self.threshold}\n")
            f.write(f"Edges above threshold: {num_edges:,}\n")
            f.write(f"Edge density: {num_edges / total_possible:.6f}\n")
            if num_ids > 0:
                avg_degree = 2 * num_edges / num_ids
                f.write(f"Average degree: {avg_degree:.2f}\n")
        
        print(f"✅ Saved {num_edges:,} edges to {output_path}")
        print(f"   Edge density: {num_edges / total_possible:.6f}")
        if num_ids > 0:
            print(f"   Average degree: {2 * num_edges / num_ids:.2f}")
        
        return output_path
    
    def evaluate_predictions(
        self,
        pred_path: str,
        qid_pairs_path: str
    ) -> Dict[str, float]:
        """
        Evaluate predictions against ground truth pairs.
        
        Uses integer indices instead of string tuples for efficient comparison.
        This is much faster and uses less RAM for millions of pairs.
        
        Args:
            pred_path: Path to saved predictions (.pt file)
            qid_pairs_path: Path to ground truth pairs file
            
        Returns:
            Dictionary with precision, recall, and F1 metrics
        """
        # Load predictions
        print(f"Loading predictions from {pred_path}")
        data = torch.load(pred_path, weights_only=False)
        sources = data["sources"]
        targets = data["targets"]
        sorted_db_ids = data["sorted_db_ids"]
        
        # Build string ID -> integer index mapping
        id_to_idx = {uid: i for i, uid in enumerate(sorted_db_ids)}
        
        # Load ground truth and convert to integer index pairs
        print("Loading and indexing ground truth...")
        gt_pairs_indices: Set[Tuple[int, int]] = set()
        with open(qid_pairs_path, "r") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                a, b = row["db_1"], row["db_2"]
                if a in id_to_idx and b in id_to_idx:
                    idx_a, idx_b = id_to_idx[a], id_to_idx[b]
                    # Ensure sorted order (min, max) to match prediction logic
                    gt_pairs_indices.add((min(idx_a, idx_b), max(idx_a, idx_b)))
        
        print(f"Loaded {len(gt_pairs_indices):,} ground truth pairs")
        
        # Convert predictions to set of integer tuples (much faster than strings)
        print("Indexing predictions...")
        pred_src = sources.tolist()
        pred_tgt = targets.tolist()
        
        # Predictions are already in (min, max) order from compute_all_pairs
        pred_pairs_indices: Set[Tuple[int, int]] = set(zip(pred_src, pred_tgt))
        
        print(f"Total predicted pairs: {len(pred_pairs_indices):,}")
        
        # Compute metrics using integer sets
        true_positives = len(pred_pairs_indices & gt_pairs_indices)
        precision = true_positives / len(pred_pairs_indices) if pred_pairs_indices else 0.0
        recall = true_positives / len(gt_pairs_indices) if gt_pairs_indices else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        print(f"Precision: {precision:.4f} ({true_positives}/{len(pred_pairs_indices)})")
        print(f"Recall: {recall:.4f} ({true_positives}/{len(gt_pairs_indices)})")
        print(f"F1: {f1:.4f}")
        
        return {"precision": precision, "recall": recall, "f1": f1, "true_positives": true_positives}
    
    @classmethod
    def from_config(cls, config: "PreprocessConfig") -> "SimilarityComputer":
        """Create a SimilarityComputer from a PreprocessConfig."""
        from .config import PreprocessConfig
        return cls(threshold=config.similarity_threshold)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Compute all-pairs similarity")
    parser.add_argument("--embeddings", type=str, required=True, help="Path to embeddings .pt file")
    parser.add_argument("--output", type=str, required=True, help="Output path for predictions")
    parser.add_argument("--qid-pairs", type=str, default=None, help="Path to ground truth pairs (for evaluation)")
    parser.add_argument("--threshold", type=float, default=0.6, help="Similarity threshold")
    parser.add_argument("--chunk-size", type=int, default=1024, help="Chunk size for GPU batching")
    parser.add_argument("--gpu", type=str, default="0", help="GPU device ID to use")
    parser.add_argument("--no-fp16", action="store_true", help="Disable FP16 (use FP32)")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate saved predictions against qid-pairs")
    
    args = parser.parse_args()
    
    # Set GPU before any CUDA operations
    set_gpu_device(args.gpu)
    
    computer = SimilarityComputer(
        threshold=args.threshold,
        chunk_size=args.chunk_size,
        use_fp16=not args.no_fp16
    )
    
    if args.evaluate and args.qid_pairs:
        # Evaluate mode: load existing predictions and evaluate
        computer.evaluate_predictions(args.output, args.qid_pairs)
    else:
        # Compute mode: generate predictions
        computer.compute_all_pairs(
            embedding_path=args.embeddings,
            output_path=args.output,
            qid_pairs_path=args.qid_pairs
        )
        
        # Optionally evaluate after computation
        if args.qid_pairs:
            print("\n--- Evaluation ---")
            computer.evaluate_predictions(args.output, args.qid_pairs)
