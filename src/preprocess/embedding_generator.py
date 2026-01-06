"""
Embedding generation for database schemas.

Wraps BGEEmbedder functionality for batch embedding generation.
"""

import os
import sys
import torch
from tqdm import tqdm
from typing import Optional, Tuple

# Allow imports from parent directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


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
        chunk_size: int = 1000
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
        
        Returns:
            Path to the saved embeddings file
        """
        from model.WKDataset import WKDataset
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        print(f"Serialization mode: {self.serialization_mode}")
        
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
        
        all_embeddings = []
        all_ids = []
        texts = []
        ids = []
        
        for i in tqdm(range(*db_id_range), desc="Embedding DBs"):
            db_id = str(i).zfill(5)
            try:
                schema_text = self.serializer.serialize(loader, db_id)
                texts.append(schema_text)
                ids.append(db_id)
            except Exception as e:
                print(f"⚠️ Skipped {db_id}: {e}")
                continue
            
            if len(texts) == batch_size or (i == db_id_range[1] - 1):
                try:
                    with torch.no_grad():
                        embs = self.embedder.get_embedding(texts, batch_size=batch_size).cpu()
                    all_embeddings.append(embs)
                    all_ids.extend(ids)
                except Exception as e:
                    print(f"❌ Error processing batch at db_id {db_id}: {e}")
                
                texts = []
                ids = []
            
            # Intermediate save
            if len(all_ids) >= chunk_size:
                self._save_embeddings(all_embeddings, all_ids, embeddings_file)
                print(f"Intermediate save with {len(all_ids)} entries")
        
        # Final save
        if all_ids:
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
        db_id_range=(args.start_id, args.end_id)
    )
