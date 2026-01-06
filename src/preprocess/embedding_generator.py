"""
Embedding generation for database schemas.

Wraps BGEEmbedder functionality for batch embedding generation.
"""

import argparse
import os
import sys


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

import torch
from tqdm import tqdm
from typing import Optional, Tuple

# Allow imports from parent directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# Module-level worker function for multiprocessing (must be at module level to be picklable)
def _load_schema_batch_worker(args):
    """
    Worker function that processes a batch of db_ids.
    Args is a tuple: (db_id_batch, schema_dir, csv_dir, serialization_mode, sample_size)
    """
    db_id_batch, schema_dir, csv_dir, serialization_mode, sample_size = args
    
    # Import here to avoid issues with multiprocessing
    from model.WKDataset import WKDataset
    from preprocess.schema_serializer import SchemaSerializer
    
    loader = WKDataset(schema_dir=schema_dir, csv_base_dir=csv_dir)
    serializer = SchemaSerializer(mode=serialization_mode, sample_size=sample_size)
    
    results = []
    for db_id in db_id_batch:
        try:
            schema_text = serializer.serialize(loader, db_id)
            results.append((db_id, schema_text))
        except Exception:
            pass
    return results


class EmbeddingGenerator:
    """
    Generate embeddings for all databases using a pretrained model.
    
    Wraps BGEEmbedder.generate_and_save_all_embeddings() with a cleaner interface.
    
    Args:
        model_type: Model type ("bge-m3" or "bge-large-en-v1.5")
        model_path: Path to finetuned model weights (None for pretrained)
        serialization_mode: Schema serialization mode for ablation studies
        sample_size: Number of sample values per column
    
    Example:
        >>> generator = EmbeddingGenerator(model_path="out/model/best")
        >>> generator.generate(
        ...     schema_dir="data/schema",
        ...     csv_dir="data/unzip",
        ...     output_path="data/graph/all_embeddings.pt"
        ... )
    """
    
    def __init__(
        self,
        model_type: str = "bge-m3",
        model_path: Optional[str] = None,
        serialization_mode: str = "full",
        sample_size: int = 3
    ):
        """Initialize the embedding generator."""
        self.model_type = model_type
        self.model_path = model_path
        self.serialization_mode = serialization_mode
        self.sample_size = sample_size
        self._embedder = None
        self._serializer = None
    
    @property
    def embedder(self):
        """Lazy-load the BGE embedder."""
        if self._embedder is None:
            from model.BGEEmbedder import BGEEmbedder
            self._embedder = BGEEmbedder(
                model_type=self.model_type,
                model_path=self.model_path
            )
        return self._embedder
    
    @property
    def serializer(self):
        """Lazy-load the schema serializer."""
        if self._serializer is None:
            from .schema_serializer import SchemaSerializer
            self._serializer = SchemaSerializer(
                mode=self.serialization_mode,
                sample_size=self.sample_size
            )
        return self._serializer
    
    def generate(
        self,
        schema_dir: str,
        csv_dir: str,
        output_path: str,
        batch_size: int = 8,
        db_id_range: Tuple[int, int] = (0, 100000),
        chunk_size: int = 1000,
        num_workers: int = 32
    ) -> str:
        """
        Generate embeddings for all databases in a range.
        
        Args:
            schema_dir: Directory containing schema JSON files
            csv_dir: Directory containing database CSV files
            output_path: Output path for embeddings (directory or .pt file)
            batch_size: Batch size for embedding generation (default: 8)
            db_id_range: Range of database IDs to process (default: 0-100000)
            chunk_size: Intermediate save frequency (default: 1000)
            num_workers: Number of workers for parallel schema loading (default: 32)
        
        Returns:
            Path to the saved embeddings file
        """
        from model.WKDataset import WKDataset
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        print(f"Serialization mode: {self.serialization_mode}")
        print(f"Batch size: {batch_size}")
        
        # Ensure output directory exists
        if output_path.endswith('.pt'):
            output_dir = os.path.dirname(output_path)
            embeddings_file = output_path
        else:
            output_dir = output_path
            embeddings_file = os.path.join(output_dir, "all_embeddings.pt")
        os.makedirs(output_dir, exist_ok=True)
        
        loader = WKDataset(schema_dir=schema_dir, csv_base_dir=csv_dir)
        self.embedder.model.eval()
        
        # Step 1: Collect all valid schemas using multiprocessing
        # Each worker processes a batch of 10 databases for efficiency
        print(f"Step 1: Collecting database schemas with {num_workers} workers...")
        
        # Prepare db_id batches (10 per batch)
        db_ids = [str(i).zfill(5) for i in range(*db_id_range)]
        worker_batch_size = 10
        db_id_batches = [db_ids[i:i+worker_batch_size] for i in range(0, len(db_ids), worker_batch_size)]
        
        # Prepare worker args: each batch includes all info needed to recreate serializer
        worker_args = [
            (batch, schema_dir, csv_dir, self.serialization_mode, self.sample_size)
            for batch in db_id_batches
        ]
        
        # Use multiprocessing for true parallel I/O
        from concurrent.futures import ProcessPoolExecutor, as_completed
        
        all_texts = []
        all_ids = []
        
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(_load_schema_batch_worker, args): args for args in worker_args}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Loading schema batches"):
                results = future.result()
                for db_id, schema_text in results:
                    all_ids.append(db_id)
                    all_texts.append(schema_text)
        
        # Sort by db_id to ensure consistent ordering
        sorted_pairs = sorted(zip(all_ids, all_texts), key=lambda x: x[0])
        all_ids = [p[0] for p in sorted_pairs]
        all_texts = [p[1] for p in sorted_pairs]
        
        print(f"Loaded {len(all_ids)} database schemas")
        
        # Step 2: Process embeddings in batches
        print("Step 2: Generating embeddings in batches...")
        all_embeddings = []
        num_batches = (len(all_texts) + batch_size - 1) // batch_size
        
        with torch.no_grad():
            for batch_idx in tqdm(range(num_batches), desc="Embedding batches"):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(all_texts))
                batch_texts = all_texts[start_idx:end_idx]
                
                try:
                    embs = self.embedder.get_embedding(batch_texts, batch_size=len(batch_texts)).cpu()
                    all_embeddings.append(embs)
                except Exception as e:
                    print(f"❌ Error processing batch {batch_idx}: {e}")
                    # Create zero embeddings for failed batch to maintain alignment
                    continue
        
        # Final save
        if all_embeddings:
            self._save_embeddings(all_embeddings, all_ids, embeddings_file)
            print(f"✅ Final save: {len(all_ids)} embeddings → {embeddings_file}")
        
        return embeddings_file
    
    def _save_embeddings(self, embeddings_list, ids, output_path):
        """Save embeddings to file."""
        current_embeddings = torch.cat(embeddings_list, dim=0)
        db_id_to_index = {db_id: idx for idx, db_id in enumerate(ids)}
        torch.save({
            "embeddings": current_embeddings,
            "db_id_to_index": db_id_to_index
        }, output_path)
    
    @classmethod
    def from_config(cls, config: "PreprocessConfig", model_path: Optional[str] = None) -> "EmbeddingGenerator":
        """Create an EmbeddingGenerator from a PreprocessConfig."""
        from .config import PreprocessConfig
        return cls(
            model_path=model_path,
            serialization_mode=config.serialization_mode,
            sample_size=config.sample_size
        )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate database embeddings")
    parser.add_argument("--schema-dir", type=str, default="data/schema", help="Schema directory")
    parser.add_argument("--csv-dir", type=str, default="data/unzip", help="CSV directory")
    parser.add_argument("--output", type=str, required=True, help="Output path")
    parser.add_argument("--model-path", type=str, default=None, help="Finetuned model path")
    parser.add_argument("--mode", type=str, default="full",
                        choices=["schema_only", "data_only", "full"],
                        help="Schema serialization mode")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--start-id", type=int, default=0, help="Start database ID")
    parser.add_argument("--end-id", type=int, default=100000, help="End database ID")
    parser.add_argument("--num-workers", type=int, default=32, help="Workers for parallel schema loading")
    parser.add_argument("--gpu", type=str, default=None,
                        help="GPU device ID(s) to use (e.g., '0', '1')")
    
    args = parser.parse_args()
    
    generator = EmbeddingGenerator(
        model_path=args.model_path,
        serialization_mode=args.mode
    )
    generator.generate(
        schema_dir=args.schema_dir,
        csv_dir=args.csv_dir,
        output_path=args.output,
        batch_size=args.batch_size,
        db_id_range=(args.start_id, args.end_id),
        num_workers=args.num_workers
    )
