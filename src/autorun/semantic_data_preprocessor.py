"""
Semantic Data Preprocessing for Federated Learning Validation

This module extends the AutomatedDataPreprocessor to use semantic column alignment
based on BGE embeddings instead of simple string matching.

Inspired by DeepJoin baseline approach for table matching.
"""

import os
import sys
import json
import torch
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import torch.nn.functional as F

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from autorun.data_preprocessor import AutomatedDataPreprocessor

# Global debug flag
DEBUG = False

def debug_print(*args, **kwargs):
    """Print only if DEBUG mode is enabled."""
    if DEBUG:
        print(*args, **kwargs)

class SuppressOutput:
    """Context manager to suppress stdout when not in debug mode."""
    def __init__(self):
        self._devnull = None
        self._original_stdout = None
    
    def __enter__(self):
        if not DEBUG:
            import io
            self._original_stdout = sys.stdout
            sys.stdout = io.StringIO()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._original_stdout is not None:
            sys.stdout = self._original_stdout
        return False

class SemanticColumnAligner:
    """
    Semantic column alignment using BGE-M3 embeddings.
    
    This class:
    1. Loads the BGE-M3 model once (original, not fine-tuned)
    2. Generates embeddings for columns (name + sample values)
    3. Performs greedy 1-to-1 matching with similarity threshold
    4. Caches embeddings to disk for reuse when changing threshold
    """
    
    # Default cache directory for embeddings
    DEFAULT_CACHE_DIR = "data/auto_semantic/embedding_cache"
    
    def __init__(self, 
                 model_type: str = "bge-m3",
                 model_path: Optional[str] = None,
                 sample_size: int = 10,
                 similarity_threshold: float = 0.80,
                 cache_dir: Optional[str] = None,
                 device: Optional[str] = None):
        """
        Initialize the semantic column aligner.
        
        Args:
            model_type: Model type to use (default: "bge-m3")
            model_path: Path to model weights (None = use original pretrained)
            sample_size: Number of sample values per column (default: 10)
            similarity_threshold: Minimum similarity for matching (default: 0.80)
            cache_dir: Directory to cache column embeddings (uses default if None)
            device: Device to use (cuda/cpu, auto-detected if None)
        """
        self.model_type = model_type
        self.model_path = model_path  # None = use original BGE-M3, not fine-tuned
        self.sample_size = sample_size
        self.similarity_threshold = similarity_threshold
        self.cache_dir = cache_dir or self.DEFAULT_CACHE_DIR
        
        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Lazy load model
        self._embedder = None
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # In-memory embedding cache: {cache_key: embedding_tensor}
        self._embedding_cache = {}
        
    def _get_cache_path(self, db_id: str, table_title: str) -> str:
        """Get the cache file path for a database table's embeddings."""
        # Sanitize filename
        safe_db_id = db_id.replace("/", "_").replace("\\", "_")
        safe_table = table_title.replace("/", "_").replace("\\", "_").replace(" ", "_")
        return os.path.join(self.cache_dir, f"{safe_db_id}_{safe_table}_embeddings.pt")
    
    def _load_cached_embeddings(self, db_id: str, table_title: str) -> Optional[Tuple[torch.Tensor, List[str]]]:
        """Load cached embeddings from disk if available."""
        cache_path = self._get_cache_path(db_id, table_title)
        if os.path.exists(cache_path):
            try:
                cached = torch.load(cache_path, map_location='cpu')
                debug_print(f"Loaded cached embeddings for {db_id}/{table_title}")
                return cached['embeddings'], cached['col_names']
            except Exception as e:
                debug_print(f"Failed to load cache {cache_path}: {e}")
        return None
    
    def _save_embeddings_to_cache(self, db_id: str, table_title: str, 
                                   embeddings: torch.Tensor, col_names: List[str]):
        """Save embeddings to disk cache."""
        cache_path = self._get_cache_path(db_id, table_title)
        try:
            torch.save({
                'embeddings': embeddings.cpu(),
                'col_names': col_names,
                'sample_size': self.sample_size,
                'model_type': self.model_type
            }, cache_path)
            debug_print(f"Saved embeddings to cache: {cache_path}")
        except Exception as e:
            debug_print(f"Failed to save cache {cache_path}: {e}")
        
    @property
    def embedder(self):
        """Lazy load the BGE embedder."""
        if self._embedder is None:
            from model.BGEEmbedder import BGEEmbedder
            print(f"Loading BGE-M3 model (original, not fine-tuned)...")
            # Use model_path=None to load original pretrained BGE-M3
            self._embedder = BGEEmbedder(
                model_type=self.model_type,
                model_path=self.model_path
            )
            print(f"BGE-M3 model loaded on {self._device}")
        return self._embedder
    
    def format_column_for_embedding(self, 
                                     table_title: str,
                                     col_name: str, 
                                     samples: List) -> str:
        """
        Format a column for embedding generation using DeepJoin paper template.
        
        Template (from DeepJoin paper Section 3.1, Tables 12-13):
        "{table_title}. {column_name} contains {n} values ({max_len}, {min_len}, {avg_len}): {values}."
        
        Args:
            table_title: Table name/title
            col_name: Column name
            samples: List of sample values
            
        Returns:
            Formatted string for embedding
        """
        # Convert samples to strings, filter out None/NaN
        valid_samples = []
        for s in samples:
            if s is not None and pd.notna(s) and str(s).lower() != 'nan':
                valid_samples.append(str(s))
        
        # Get distinct values (up to sample_size for embedding, but count all for n)
        distinct_samples = list(dict.fromkeys(valid_samples))  # Preserve order, remove duplicates
        n_distinct = len(distinct_samples)
        
        # Calculate statistics based on character lengths
        if not distinct_samples:
            return f"{table_title}. {col_name} contains 0 values (0, 0, 0.0): ."
        
        lengths = [len(s) for s in distinct_samples]
        max_len = max(lengths)
        min_len = min(lengths)
        avg_len = round(np.mean(lengths), 1)
        
        # Create value string (take up to sample_size distinct values)
        values_str = ", ".join(distinct_samples[:self.sample_size])
        
        # Format: "{table_title}. {col_name} contains {n} values ({max}, {min}, {avg}): {values}."
        return f"{table_title}. {col_name} contains {n_distinct} values ({max_len}, {min_len}, {avg_len}): {values_str}."
    
    def compute_column_embedding(self, 
                                  table_title: str,
                                  col_name: str,
                                  samples: List) -> torch.Tensor:
        """
        Compute embedding for a single column.
        
        Args:
            table_title: Table name/title
            col_name: Column name
            samples: List of sample values
            
        Returns:
            Embedding tensor of shape [1, embedding_dim]
        """
        text = self.format_column_for_embedding(table_title, col_name, samples)
        with torch.no_grad():
            embedding = self.embedder.get_embedding([text], batch_size=1)
        return embedding
    
    def get_dataframe_embeddings(self, 
                                  df: pd.DataFrame,
                                  table_title: str = "table",
                                  db_id: str = "") -> Tuple[torch.Tensor, List[str]]:
        """
        Get embeddings for all columns in a DataFrame using DeepJoin format.
        
        Embeddings are cached to disk for reuse. If cache exists and column names
        match, cached embeddings are returned without re-computing.
        
        Args:
            df: Input DataFrame
            table_title: Table name/title for the DeepJoin template
            db_id: Database ID for caching and logging
            
        Returns:
            Tuple of (embeddings tensor [num_cols, embedding_dim], column names list)
        """
        col_names = list(df.columns)
        
        # Try to load from disk cache
        if db_id:
            cached = self._load_cached_embeddings(db_id, table_title)
            if cached is not None:
                cached_embeddings, cached_col_names = cached
                # Verify column names match (order matters)
                if cached_col_names == col_names:
                    return cached_embeddings.to(self._device), col_names
                else:
                    debug_print(f"Cache mismatch for {db_id}/{table_title}: columns changed")
        
        # Generate embeddings
        texts = []
        for col in col_names:
            # Get sample values - use more samples to get better stats
            # DeepJoin uses all distinct values for stats, but limited for the value string
            all_samples = df[col].dropna().head(1000).tolist()  # Get up to 1000 for stats
            
            # Format for embedding using DeepJoin template
            text = self.format_column_for_embedding(table_title, col, all_samples)
            texts.append(text)
        
        # Batch encode all columns
        with torch.no_grad():
            embeddings = self.embedder.get_embedding(texts, batch_size=len(texts))
        
        # Save to disk cache
        if db_id:
            self._save_embeddings_to_cache(db_id, table_title, embeddings, col_names)
        
        return embeddings, col_names
    
    def find_semantic_column_matches(self,
                                      df_a: pd.DataFrame,
                                      df_b: pd.DataFrame,
                                      table_title_a: str = "table_a",
                                      table_title_b: str = "table_b",
                                      db_id_a: str = "",
                                      db_id_b: str = "") -> List[Tuple[str, str, float]]:
        """
        Find semantic column matches between two DataFrames using greedy 1-to-1 matching.
        
        Algorithm:
        1. Compute embeddings for all columns in both DataFrames using DeepJoin format
        2. Calculate cosine similarity matrix
        3. Greedy matching: sort by similarity, accept if score > threshold and both cols unused
        
        Args:
            df_a: First DataFrame
            df_b: Second DataFrame
            table_title_a: Table title for first df (used in DeepJoin template)
            table_title_b: Table title for second df (used in DeepJoin template)
            db_id_a: Database ID for first df (for caching)
            db_id_b: Database ID for second df (for caching)
            
        Returns:
            List of (col_a, col_b, similarity_score) tuples for matched columns
        """
        # Get embeddings for both DataFrames using table titles (with caching via db_id)
        emb_a, cols_a = self.get_dataframe_embeddings(df_a, table_title=table_title_a, db_id=db_id_a)
        emb_b, cols_b = self.get_dataframe_embeddings(df_b, table_title=table_title_b, db_id=db_id_b)
        
        debug_print(f"Computing semantic similarity: {len(cols_a)} cols ({table_title_a}) x {len(cols_b)} cols ({table_title_b})")
        
        # Compute cosine similarity matrix: [num_cols_a, num_cols_b]
        # Embeddings are already normalized by BGEEmbedder
        similarity_matrix = torch.mm(emb_a, emb_b.t())
        
        # Greedy 1-to-1 matching
        matches = []
        used_a = set()
        used_b = set()
        
        # Create list of (index_a, index_b, score) candidates
        candidates = []
        for i in range(len(cols_a)):
            for j in range(len(cols_b)):
                score = similarity_matrix[i, j].item()
                if score >= self.similarity_threshold:
                    candidates.append((i, j, score))
        
        # Sort by score descending (best matches first)
        candidates.sort(key=lambda x: x[2], reverse=True)
        
        # Greedy assignment
        for i, j, score in candidates:
            if i not in used_a and j not in used_b:
                # Found a valid 1-to-1 match
                col_name_a = cols_a[i]
                col_name_b = cols_b[j]
                matches.append((col_name_a, col_name_b, score))
                
                # Mark as used
                used_a.add(i)
                used_b.add(j)
        
        debug_print(f"Found {len(matches)} semantic matches (threshold={self.similarity_threshold})")
        if matches:
            debug_print(f"  Top matches: {matches[:5]}{'...' if len(matches) > 5 else ''}")
        
        return matches


class SemanticDataPreprocessor(AutomatedDataPreprocessor):
    """
    Data preprocessor that uses semantic column alignment.
    
    Extends AutomatedDataPreprocessor, overriding the column identification
    method to use BGE embeddings for semantic matching.
    """
    
    def __init__(self,
                 test_size: float = 0.2,
                 random_state: int = 42,
                 min_label_variance: float = 0.01,
                 max_missing_ratio: float = 0.5,
                 join_timeout: int = 300,
                 max_rows: int = 1000000,
                 similarity_threshold: float = 0.80,
                 sample_size: int = 10):
        """
        Initialize the semantic data preprocessor.
        
        Args:
            test_size: Fraction of data for test set (default: 0.2)
            random_state: Random state for reproducibility (default: 42)
            min_label_variance: Minimum variance for label columns (default: 0.01)
            max_missing_ratio: Maximum ratio of missing values (default: 0.5)
            join_timeout: Timeout for table joins in seconds (default: 300)
            max_rows: Maximum rows after joining (default: 1000000)
            similarity_threshold: Minimum similarity for column matching (default: 0.80)
            sample_size: Number of sample values for embedding (default: 10)
        """
        super().__init__(
            test_size=test_size,
            random_state=random_state,
            min_label_variance=min_label_variance,
            max_missing_ratio=max_missing_ratio,
            join_timeout=join_timeout,
            max_rows=max_rows
        )
        
        self.similarity_threshold = similarity_threshold
        self.sample_size = sample_size
        
        # Lazy-loaded semantic aligner (shared across all pairs)
        self._aligner = None
        
        # Current db_ids being processed (for caching in identify_common_columns)
        self._current_db_id1 = ""
        self._current_db_id2 = ""
    
    def set_current_db_ids(self, db_id1: str, db_id2: str):
        """Set db_ids for the current pair being processed (used for caching)."""
        self._current_db_id1 = str(db_id1)
        self._current_db_id2 = str(db_id2)
    
    @property
    def aligner(self) -> SemanticColumnAligner:
        """Lazy load the semantic column aligner."""
        if self._aligner is None:
            self._aligner = SemanticColumnAligner(
                model_type="bge-m3",
                model_path=None,  # Use original pretrained, not fine-tuned
                sample_size=self.sample_size,
                similarity_threshold=self.similarity_threshold
            )
        return self._aligner
    
    def identify_common_columns(self, 
                                 df1: pd.DataFrame, 
                                 df2: pd.DataFrame) -> Tuple[List[str], pd.DataFrame]:
        """
        Identify common columns between two DataFrames using semantic matching.
        
        Overrides the parent method to use BGE embeddings instead of snapshot matching.
        Uses _current_db_id1/2 for embedding caching if set.
        
        Args:
            df1: First DataFrame
            df2: Second DataFrame
            
        Returns:
            Tuple of (common column names from df1, df2 with renamed columns)
        """
        # Get semantic matches (pass db_ids for caching)
        matches = self.aligner.find_semantic_column_matches(
            df1, df2,
            db_id_a=self._current_db_id1,
            db_id_b=self._current_db_id2
        )
        
        if not matches:
            debug_print("Warning: No semantic column matches found above threshold")
            # Fall back to parent's snapshot-based matching
            return super().identify_common_columns(df1, df2)
        
        # Create rename mapping for df2 to match df1 column names
        rename_map = {match[1]: match[0] for match in matches}
        df2_aligned = df2.rename(columns=rename_map)
        
        # Get common column names (using df1 names as reference)
        common_columns = [match[0] for match in matches]
        
        debug_print(f"Semantic matching found {len(common_columns)} common columns")
        debug_print(f"  Matches with similarity: {[(m[0], m[1], f'{m[2]:.3f}') for m in matches[:5]]}")
        
        return common_columns, df2_aligned


def main():
    """
    Main function for semantic data preprocessing.
    
    This can be called as a standalone script to preprocess pairs using 
    semantic column alignment instead of string-based matching.
    """
    import argparse
    import json
    from pathlib import Path
    
    parser = argparse.ArgumentParser(
        description="Semantic Data Preprocessing for FL Validation",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--input", type=str, required=True,
                        help="Path to sampled pairs JSON file")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for processed data")
    parser.add_argument("--test-size", type=float, default=0.2,
                        help="Fraction of data for test set (default: 0.2)")
    parser.add_argument("--random-state", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--min-label-variance", type=float, default=0.01,
                        help="Minimum variance for regression labels (default: 0.01)")
    parser.add_argument("--max-missing-ratio", type=float, default=0.5,
                        help="Maximum missing value ratio (default: 0.5)")
    parser.add_argument("--similarity-threshold", type=float, default=0.80,
                        help="Minimum similarity for column matching (default: 0.80)")
    parser.add_argument("--column-sample-size", type=int, default=10,
                        help="Number of sample values per column for embedding (default: 10)")
    parser.add_argument("--retry", action="store_true",
                        help="Retry failed pairs from previous run")
    parser.add_argument("--debug", action="store_true",
                        help="Enable verbose debug output")
    args = parser.parse_args()
    
    # Set global debug flag
    global DEBUG
    DEBUG = args.debug
    
    print("=" * 60)
    print("SEMANTIC DATA PREPROCESSING FOR FL VALIDATION")
    print("=" * 60)
    print(f"Using BGE-M3 embeddings for column alignment")
    print(f"Similarity threshold: {args.similarity_threshold}")
    print(f"Sample values per column: {args.column_sample_size}")
    print()
    
    # Load pairs - support both preprocessing_summary.json and sampled_pairs.json formats
    print(f"Loading pairs from {args.input}")
    with open(args.input, 'r') as f:
        input_data = json.load(f)
    
    # Detect input format and extract pairs
    pairs = []
    needs_folder_lookup = False
    
    if 'results' in input_data:
        # Format: preprocessing_summary.json from original pipeline
        print("Detected preprocessing_summary.json format (from original pipeline)")
        for result in input_data['results']:
            if result.get('status') == 'success':
                # Extract pair info from the result
                pair = {
                    'db_id1': result['db_id1'],
                    'db_id2': result['db_id2'],
                    'similarity': result.get('similarity', 1.0),
                    'folder1': result.get('folder1', ''),
                    'folder2': result.get('folder2', ''),
                }
                pairs.append(pair)
        print(f"Found {len(pairs)} successful pairs from original preprocessing")
        # Preprocessing summary doesn't have folder info, need to look it up
        needs_folder_lookup = True
    elif 'pairs' in input_data:
        # Format: sampled_pairs.json
        pairs = input_data['pairs']
        print(f"Found {len(pairs)} pairs from sampled_pairs.json")
    elif isinstance(input_data, list):
        # Format: raw list of pairs
        pairs = input_data
        print(f"Found {len(pairs)} pairs from list format")
    else:
        raise ValueError(f"Unknown input format. Expected 'results', 'pairs', or list in {args.input}")
    
    if not pairs:
        print("ERROR: No pairs found to process!")
        return 1
    
    # If folder info is missing, look it up from unzip directory
    if needs_folder_lookup:
        print("Building db_id -> folder mapping...")
        unzip_dir = "data/unzip"
        db_id_to_folder = {}
        
        if os.path.exists(unzip_dir):
            for folder in os.listdir(unzip_dir):
                folder_path = os.path.join(unzip_dir, folder)
                if os.path.isdir(folder_path):
                    try:
                        id_str, _ = folder.split(" ", 1)
                        numeric_id = int(id_str)
                        db_id_to_folder[numeric_id] = folder
                    except (ValueError, IndexError):
                        continue
            print(f"Found {len(db_id_to_folder)} database folders")
        
            # Update pairs with folder info
            valid_pairs = []
            for pair in pairs:
                folder1 = db_id_to_folder.get(pair['db_id1'])
                folder2 = db_id_to_folder.get(pair['db_id2'])
                if folder1 and folder2:
                    pair['folder1'] = folder1
                    pair['folder2'] = folder2
                    valid_pairs.append(pair)
                else:
                    print(f"Warning: Missing folder for pair {pair['db_id1']}-{pair['db_id2']}")
            pairs = valid_pairs
            print(f"After folder lookup: {len(pairs)} valid pairs")
    
    # Initialize semantic preprocessor
    preprocessor = SemanticDataPreprocessor(
        test_size=args.test_size,
        random_state=args.random_state,
        min_label_variance=args.min_label_variance,
        max_missing_ratio=args.max_missing_ratio,
        similarity_threshold=args.similarity_threshold,
        sample_size=args.column_sample_size
    )
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Handle retry mode - load failed pairs from previous run
    if args.retry:
        error_log = output_dir / 'preprocessing_errors.log'
        if error_log.exists():
            failed_pair_ids = preprocessor.parse_failed_pairs_from_log(str(error_log))
            print(f"Retry mode: Found {len(failed_pair_ids)} failed pairs from previous run")
            pairs = [p for p in pairs if f"{p['db_id1']:05d}-{p['db_id2']:05d}" in failed_pair_ids]
            print(f"Will retry {len(pairs)} pairs")
            preprocessor.rotate_error_logs(str(output_dir))
    
    # Process each pair with progress bar
    results = []
    successful = 0
    failed = 0
    task_type_counts = {'classification': 0, 'regression': 0}
    class_counts = {}
    reg_stats = []
    
    # Use tqdm for progress bar
    pbar = tqdm(pairs, desc="Semantic preprocessing", unit="pair")
    
    for pair in pbar:
        pair_id = f"{pair['db_id1']:05d}_{pair['db_id2']:05d}"
        pbar.set_postfix({'pair': pair_id, 'ok': successful, 'fail': failed})
        
        try:
            # Set db_ids for caching before processing
            preprocessor.set_current_db_ids(pair['db_id1'], pair['db_id2'])
            
            # Suppress verbose output from parent class unless in debug mode
            with SuppressOutput():
                result = preprocessor.process_pair(pair, str(output_dir))
            results.append(result)
            
            if result['status'] == 'success':
                successful += 1
                task_type = result['task_type']
                task_type_counts[task_type] += 1
                
                if task_type == 'classification':
                    n_classes = result['processing_params']['n_classes']
                    class_counts[n_classes] = class_counts.get(n_classes, 0) + 1
                else:
                    reg_stats.append({
                        'variance': result['label_metadata']['variance'],
                        'n_unique': result['label_metadata']['n_unique']
                    })
                
                if DEBUG:
                    tqdm.write(f"SUCCESS: {result['pair_id']} ({task_type})")
            else:
                failed += 1
                tqdm.write(f"FAILED: {result['pair_id']} - {result.get('error', 'Unknown error')}")
                
                # Log error to file
                error_log = output_dir / 'preprocessing_errors.log'
                with open(error_log, 'a') as f:
                    f.write(f"{result['pair_id']}: {result.get('error', 'Unknown error')}\n")
                    
        except Exception as e:
            failed += 1
            error_msg = str(e)
            results.append({
                'pair_id': pair_id,
                'status': 'failed',
                'error': error_msg
            })
            tqdm.write(f"FAILED: {pair_id} - {error_msg}")
            
            error_log = output_dir / 'preprocessing_errors.log'
            with open(error_log, 'a') as f:
                f.write(f"{pair_id}: {error_msg}\n")
    
    pbar.close()
    
    # Create summary
    summary = {
        'processing_mode': 'semantic',
        'similarity_threshold': args.similarity_threshold,
        'column_sample_size': args.column_sample_size,
        'total_pairs': len(pairs),
        'processed_pairs': successful,
        'failed_pairs': failed,
        'summary_stats': {
            'total_pairs': len(pairs),
            'successful': successful,
            'failed': failed,
            'success_rate': (successful / len(pairs) * 100) if pairs else 0,
            'task_type_distribution': task_type_counts
        },
        'results': results
    }
    
    # Save summary
    summary_file = output_dir / 'preprocessing_summary.json'
    
    if args.retry and summary_file.exists():
        # Merge with existing summary
        with open(summary_file, 'r') as f:
            existing_summary = json.load(f)
        
        retry_results_lookup = {r['pair_id']: r for r in results}
        updated_results = []
        for existing_result in existing_summary.get('results', []):
            pair_id = existing_result['pair_id']
            if pair_id in retry_results_lookup:
                updated_results.append(retry_results_lookup[pair_id])
            else:
                updated_results.append(existing_result)
        
        successful_total = sum(1 for r in updated_results if r['status'] != 'failed')
        failed_total = sum(1 for r in updated_results if r['status'] == 'failed')
        
        existing_summary['summary_stats']['successful'] = successful_total
        existing_summary['summary_stats']['failed'] = failed_total
        existing_summary['summary_stats']['success_rate'] = (successful_total / len(updated_results) * 100) if updated_results else 0
        existing_summary['results'] = updated_results
        existing_summary['processed_pairs'] = successful_total
        existing_summary['failed_pairs'] = failed_total
        
        summary = existing_summary
    
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    # Print summary
    print(f"\n{'='*60}")
    print("SEMANTIC PREPROCESSING SUMMARY:")
    print(f"{'='*60}")
    print(f"Total pairs: {len(pairs)}")
    print(f"Successful: {successful} ({successful/len(pairs)*100:.1f}%)" if pairs else "")
    print(f"Failed: {failed}" if pairs else "")
    print(f"\nTask type distribution:")
    print(f"  Classification: {task_type_counts['classification']}")
    print(f"  Regression: {task_type_counts['regression']}")
    print(f"\nFiles saved:")
    print(f"  Summary: {summary_file}")
    if failed > 0:
        print(f"  Error log: {output_dir / 'preprocessing_errors.log'}")
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    exit(main())
