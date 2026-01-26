
import os
from pathlib import Path

# Paths
BASE_DIR = Path(os.getcwd())
DATA_DIR = BASE_DIR / "data"
UNZIP_DIR = Path(os.environ.get("SANTOS_UNZIP_DIR", DATA_DIR / "unzip"))
OUTPUT_DIR = BASE_DIR / "out" / "santos"
INDEX_DIR = OUTPUT_DIR / "index"

# Ensure directories exist
INDEX_DIR.mkdir(parents=True, exist_ok=True)

# File paths
SYNTH_TYPE_KB_PATH = INDEX_DIR / "synth_type_kb.pkl"
SYNTH_RELATION_KB_PATH = INDEX_DIR / "synth_relation_kb.pkl"
SYNTH_TYPE_LOOKUP_PATH = INDEX_DIR / "synth_type_lookup.pkl"
SYNTH_RELATION_LOOKUP_PATH = INDEX_DIR / "synth_relation_lookup.pkl"
SYNTH_RELATION_INVERTED_INDEX_PATH = INDEX_DIR / "synth_relation_inverted_index.pkl"

# Parameters
SAMPLE_SIZE = 3 # Not used in synthesis directly but consistent with other parts
NUM_WORKERS = int(os.environ.get("SANTOS_NUM_WORKERS", 32))  # Default 32 workers
BATCH_SIZE = 1000 # Files per batch for parallel processing
CHUNK_SIZE = int(os.environ.get("SANTOS_CHUNK_SIZE", 10))  # Files per chunk for Pass 2 parallelization
