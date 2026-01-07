"""
All-pairs similarity computation using precomputed embeddings.

Computes cosine similarity between all database pairs and saves results.
"""

import os
import sys
import csv
import torch
import torch.nn.functional as F
from tqdm import tqdm
from typing import Optional, Set, Tuple


def set_gpu_device(gpu_id: str):
    """Set GPU device via CUDA_VISIBLE_DEVICES environment variable."""
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    print(f"Setting CUDA_VISIBLE_DEVICES={gpu_id}")


class SimilarityComputer:
    """
    Compute cosine similarity between all database pairs.
    
    Uses precomputed embeddings to efficiently compute all-pairs similarity
    and optionally compares against ground truth labels.
    
    Args:
        threshold: Similarity threshold for edge prediction (default: 0.6713)
        batch_size: Batch size for similarity computation (default: 256)
    
    Example:
        >>> computer = SimilarityComputer(threshold=0.6713)
        >>> computer.compute_all_pairs(
        ...     embedding_path="data/graph/all_embeddings.pt",
        ...     output_path="data/graph/all_predictions.pt",
        ...     qid_pairs_path="data/qid_pairs_fixed.csv"
        ... )
    """
    
    def __init__(
        self,
        threshold: float = 0.6713,
        batch_size: int = 256
    ):
        """Initialize the similarity computer."""
        self.threshold = threshold
        self.batch_size = batch_size
    
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
        chunk_size: int = 1024
    ) -> str:
        """
        Compute similarity for all database pairs using fast matrix multiplication.
        
        Args:
            embedding_path: Path to precomputed embeddings (.pt file)
            output_path: Output path for predictions (.pt file)
            qid_pairs_path: Optional path to ground truth pairs for labeling
            chunk_size: Number of rows to process at once (controls memory usage)
        
        Returns:
            Path to the saved predictions file
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load embeddings
        print(f"Loading embeddings from {embedding_path}")
        saved_data = torch.load(embedding_path, map_location=device)
        db_id_to_index = saved_data["db_id_to_index"]
        raw_embeddings = saved_data["embeddings"].to(device)
        
        # Load ground truth if provided
        qid_pairs = set()
        if qid_pairs_path and os.path.exists(qid_pairs_path):
            qid_pairs = self.load_qid_pairs(qid_pairs_path)
        
        # Sort db_ids and reorder embeddings once upfront (no indexing in loop)
        all_db_ids = sorted(db_id_to_index.keys())
        num_ids = len(all_db_ids)
        total_possible = num_ids * (num_ids - 1) // 2
        
        print(f"Total unique IDs: {num_ids}")
        print(f"Total unordered pairs (C(n,2)): {total_possible:,}")
        print(f"Similarity threshold: {self.threshold}")
        print(f"Using fast matrix multiplication with chunk_size={chunk_size}")
        
        # Reorder embeddings to match sorted db_id order, then normalize
        print("Reordering and normalizing embeddings...")
        reorder_indices = torch.tensor([db_id_to_index[db_id] for db_id in all_db_ids], device=device)
        embeddings = F.normalize(raw_embeddings[reorder_indices], p=2, dim=1)  # [num_ids, dim]
        del raw_embeddings  # Free memory
        
        # Precompute transpose for matmul
        embeddings_T = embeddings.T  # [dim, num_ids]
        
        all_records = []
        
        print("Computing all-pairs similarity...")
        with torch.no_grad():
            # Process in chunks of rows
            for chunk_start in tqdm(range(0, num_ids, chunk_size), desc="Computing similarities"):
                chunk_end = min(chunk_start + chunk_size, num_ids)
                
                # Direct slice, no indexing
                chunk_embeddings = embeddings[chunk_start:chunk_end]  # [chunk_size, dim]
                
                # Compute similarity: [chunk_size, dim] @ [dim, num_ids] -> [chunk_size, num_ids]
                sim_matrix = chunk_embeddings @ embeddings_T
                
                # Extract upper triangular pairs (j > i) to avoid duplicates
                for local_i in range(chunk_end - chunk_start):
                    global_i = chunk_start + local_i
                    anchor_id = all_db_ids[global_i]
                    
                    # Only consider j > global_i (upper triangle)
                    start_j = global_i + 1
                    if start_j >= num_ids:
                        continue
                    
                    sims = sim_matrix[local_i, start_j:]  # Similarities for j > global_i
                    
                    # Process all pairs for this anchor
                    for offset, sim_val in enumerate(sims):
                        global_j = start_j + offset
                        target_id = all_db_ids[global_j]
                        sim_val = sim_val.item()
                        edge = 1 if sim_val > self.threshold else 0
                        pair = tuple(sorted((anchor_id, target_id)))
                        label = 1 if pair in qid_pairs else 0
                        all_records.append([anchor_id, target_id, sim_val, label, edge])
        
        # Save results
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        torch.save(all_records, output_path)
        
        # Save summary
        summary_path = output_path.replace(".pt", "_summary.txt")
        with open(summary_path, "w") as f:
            f.write(f"Total IDs: {num_ids}\n")
            f.write(f"Total pairs: {len(all_records):,}\n")
            f.write(f"Threshold: {self.threshold}\n")
            f.write(f"Predicted edges: {sum(r[4] for r in all_records):,}\n")
            if qid_pairs:
                f.write(f"Ground truth pairs: {len(qid_pairs):,}\n")
        
        print(f"✅ Saved {len(all_records):,} predictions to {output_path}")
        return output_path
    
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
    parser.add_argument("--qid-pairs", type=str, default=None, help="Path to ground truth pairs")
    parser.add_argument("--threshold", type=float, default=0.6713, help="Similarity threshold")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--gpu", type=str, default="0", help="GPU device ID to use")
    
    args = parser.parse_args()
    
    # Set GPU before any CUDA operations
    set_gpu_device(args.gpu)
    
    computer = SimilarityComputer(threshold=args.threshold, batch_size=args.batch_size)
    computer.compute_all_pairs(
        embedding_path=args.embeddings,
        output_path=args.output,
        qid_pairs_path=args.qid_pairs
    )
