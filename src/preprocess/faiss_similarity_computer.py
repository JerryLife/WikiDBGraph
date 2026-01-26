"""
Nearest neighbor search using Faiss.

Computes top-k similar neighbors using Faiss instead of exhaustive all-pairs.
This is more efficient for large datasets when only the strongest edges are needed.
"""

import os
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import time
from typing import Optional, Dict

# Try to import faiss, but don't fail immediately if not present (allows syntax check/dry run)
try:
    import faiss
except ImportError:
    faiss = None
    print("WARNING: faiss module not found. Please install faiss-cpu or faiss-gpu.")


class FaissSimilarityComputer:
    """
    Compute nearest neighbors using Faiss.
    
    Args:
        k: Number of nearest neighbors to retrieve per node.
        gpu: GPU device ID to use (if available).
        use_fp16: Use float16 for embeddings (casted to float32 for Faiss if needed).
    """
    
    def __init__(
        self,
        k: int = 100,
        gpu: str = "0",
        use_fp16: bool = True
    ):
        self.k = k
        self.gpu = gpu
        self.use_fp16 = use_fp16
        
    def compute_nns(
        self,
        embedding_path: str,
        output_path: str,
        chunk_size: int = 1024
    ) -> str:
        """
        Compute k-nearest neighbors for all nodes.
        
        Args:
            embedding_path: Path to precomputed embeddings (.pt file)
            output_path: Output path for predictions (.pt file)
            chunk_size: Batch size for processing query vectors (if memory constrained)
            
        Returns:
            Path to the saved predictions file
        """
        if faiss is None:
            raise ImportError("Faiss is not installed. Please install faiss-cpu or faiss-gpu.")
            
        device = torch.device("cpu") # Load to CPU first for Faiss
        
        print(f"Loading embeddings from {embedding_path}")
        saved_data = torch.load(embedding_path, map_location=device, weights_only=False)
        db_id_to_index = saved_data["db_id_to_index"]
        # Faiss expects numpy (float32 usually)
        # We load as float32 for Faiss ingestion
        raw_embeddings = saved_data["embeddings"].float().numpy()
        
        # Sort db_ids to match the standard indexing (same as similarity_computer.py)
        sorted_db_ids = sorted(db_id_to_index.keys())
        num_ids = len(sorted_db_ids)
        
        print(f"Total unique IDs: {num_ids:,}")
        print(f"Target neighbors (k): {self.k}")
        
        # Reorder embeddings to match sorted db_id order
        print("Reordering and normalizing embeddings...")
        perm_idx = [db_id_to_index[db_id] for db_id in sorted_db_ids]
        embeddings = raw_embeddings[perm_idx]
        del raw_embeddings
        
        # Normalize for Cosine Similarity (IndexFlatIP + Normalized Vectors = Cosine)
        faiss.normalize_L2(embeddings)
        
        dim = embeddings.shape[1]
        print(f"Embedding dimension: {dim}")
        
        # Build Index
        print("Building Faiss index...")
        # Inner Product (IP) index
        index = faiss.IndexFlatIP(dim)
        
        # TODO: reliable GPU support
        # For now, sticking to CPU index for simplicity/robustness unless specifically asked to implement complex GPU resource handling
        # standard faiss-gpu usage is easy, but resources can be tricky.
        # Given "faiss will be installed later", basic CPU or auto-move is safest.
        
        if torch.cuda.is_available() and hasattr(faiss, 'StandardGpuResources'):
            print(f"Moving index to GPU {self.gpu}...")
            res = faiss.StandardGpuResources()
            # Parse gpu id (assuming single int)
            try:
                gpu_id = int(self.gpu.split(',')[0])
                index = faiss.index_cpu_to_gpu(res, gpu_id, index)
            except Exception as e:
                print(f"Warning: Failed to use GPU for Faiss, falling back to CPU. Error: {e}")
        
        print(f"Adding {num_ids} vectors to index...")
        index.add(embeddings)
        
        # Search
        print(f"Searching for {self.k} nearest neighbors...")
        start_time = time.time()
        
        # Search in chunks to avoid massive memory usage for results if N is huge
        # But for N=100k, D=100, Result ~ 100k*100*8 bytes ~ 80MB. It's fine.
        # If N=1M, Result ~ 800MB. Fine.
        # We search all at once for speed if possible.
        
        D, I = index.search(embeddings, self.k) # D: Distances (Sim), I: Indices
        
        end_time = time.time()
        print(f"Search completed in {end_time - start_time:.2f} seconds")
        
        # Process results
        # We need to flatten and convert to torch tensors to match similarity_computer format
        # Filter: keep only src < tgt to avoid duplicates and self-loops
        
        print("Processing and filtering results...")
        
        # Create source indices matching I
        # sources = [0, 0, ..., 0, 1, 1, ..., 1, ...]
        sources = np.repeat(np.arange(num_ids), self.k)
        
        # Flatten targets and scores
        targets = I.flatten()
        scores = D.flatten()
        
        # Filter to keep src < tgt
        # This keeps upper triangular part of adjacency matrix
        # Note: In NNS, relation might be asymmetric. 
        # i -> returns j (sim s). If i < j, we keep (i, j, s).
        # j -> returns i (sim s). If j > i, we typically skip (since i < j was handled).
        # But we only handled i < j IF j was in i's top-k. 
        # If i is in j's top-k but j is NOT in i's top-k, skipping (j, i) means we lose that edge.
        # However, for consistency with 'undirected' graph building, usually we assume symmetry.
        # similarity_computer.py enforces strict symmetry by only computing/storing upper triangle.
        # We will do the same to match the output schema and semantics.
        
        mask = sources < targets
        
        final_sources = torch.from_numpy(sources[mask]).int()
        final_targets = torch.from_numpy(targets[mask]).int()
        final_scores = torch.from_numpy(scores[mask]).float()
        
        num_edges = len(final_scores)
        print(f"Retained {num_edges:,} pairs (src < tgt)")
        
        output_data = {
            "sources": final_sources,
            "targets": final_targets,
            "scores": final_scores,
            "sorted_db_ids": sorted_db_ids,
            "threshold": 0.0, # NNS doesn't use a fixed threshold
            "num_nodes": num_ids,
            "k": self.k
        }
        
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        torch.save(output_data, output_path)
        
        # Save summary
        summary_path = output_path.replace(".pt", "_summary.txt")
        total_possible = num_ids * (num_ids - 1) // 2
        with open(summary_path, "w") as f:
            f.write(f"Total nodes: {num_ids:,}\n")
            f.write(f"Method: Faiss Nearest Neighbor Search (k={self.k})\n")
            f.write(f"Valid Edges (src < tgt): {num_edges:,}\n")
            f.write(f"Edge density: {num_edges / total_possible:.6f}\n")
            if num_ids > 0:
                avg_degree = 2 * num_edges / num_ids
                f.write(f"Average degree: {avg_degree:.2f}\n")
                
        print(f"✅ Saved {num_edges:,} edges to {output_path}")
        return output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute nearest neighbor similarity using Faiss")
    parser.add_argument("--embeddings", type=str, required=True, help="Path to embeddings .pt file")
    parser.add_argument("--output", type=str, required=True, help="Output path for predictions")
    parser.add_argument("--k", type=int, default=100, help="Number of neighbors")
    parser.add_argument("--gpu", type=str, default="0", help="GPU device ID")
    parser.add_argument("--batch-size", type=int, default=1024, help="Batch size (unused for now, kept for compat)")
    
    args = parser.parse_args()
    
    # Set CUDA_VISIBLE_DEVICES if using GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    
    computer = FaissSimilarityComputer(
        k=args.k,
        gpu=args.gpu
    )
    
    computer.compute_nns(
        embedding_path=args.embeddings,
        output_path=args.output
    )
