"""
GitTables Dataset Loader

A simplified dataset loader for pre-serialized triplets from GitTables.
Compatible with the existing BGEEmbedder training interface.
"""

import json
from typing import Dict, List, Any, Optional


class GitTablesDataset:
    """
    Dataset loader for pre-serialized GitTables triplets.
    
    Unlike WKDataset which loads schemas and serializes on-the-fly,
    this class loads pre-serialized triplets directly.
    """
    
    def __init__(self, triplets_path: Optional[str] = None):
        """
        Initialize the dataset.
        
        Args:
            triplets_path: Optional path to pre-load triplets
        """
        self.triplets = []
        self._text_cache = {}
        
        if triplets_path:
            self.load_triplets(triplets_path)
    
    def load_triplets(self, path: str) -> List[Dict]:
        """Load triplets from JSONL file."""
        self.triplets = []
        with open(path, 'r') as f:
            for line in f:
                if line.strip():
                    triplet = json.loads(line)
                    self.triplets.append(triplet)
        return self.triplets
    
    def get_triplet_text(self, triplet: Dict) -> tuple:
        """
        Get the text for anchor, positive, and negatives.
        
        Args:
            triplet: Dict with 'anchor', 'positive', 'negatives' keys
            
        Returns:
            Tuple of (anchor_text, positive_text, negative_texts)
        """
        anchor = triplet['anchor']
        positive = triplet['positive']
        negatives = triplet['negatives']
        
        return anchor, positive, negatives
    
    def serialize_db(self, text: str, **kwargs) -> str:
        """
        Compatibility method for BGEEmbedder.
        
        For GitTables, the text is already serialized, so we just return it.
        The 'text' parameter here is actually the already-serialized representation.
        """
        return text
    
    def load_database(self, db_id: str) -> Dict:
        """Compatibility method - returns empty dict for GitTables."""
        return {}
    
    def get_schema(self, db_id: str) -> Dict:
        """Compatibility method - returns empty dict for GitTables."""
        return {}


def load_gittables_triplets(path: str) -> List[Dict]:
    """
    Load triplets from a JSONL file.
    
    Each line should be a JSON object with:
    - anchor: str (serialized anchor table/split)
    - positive: str (serialized positive table/split)
    - negatives: List[str] (serialized negative tables/splits)
    - metadata: Optional dict with additional info
    
    Returns:
        List of triplet dictionaries
    """
    triplets = []
    with open(path, 'r') as f:
        for line in f:
            if line.strip():
                triplet = json.loads(line)
                triplets.append(triplet)
    return triplets
