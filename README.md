# WikiDBs - Database Similarity Graph and Federated Learning

A graph-based database analysis system for WikiData databases that performs similarity analysis, matching, and federated learning experiments.

## Project Overview

WikiDBs constructs a similarity graph over databases extracted from WikiData, then uses this graph for federated learning validation experiments. The pipeline consists of:

1. **Schema Embedding**: Convert database schemas to vector embeddings using BGE-M3
2. **Similarity Graph**: Compute pairwise similarity and build a graph structure
3. **Federated Learning**: Train and evaluate FL algorithms on similar database pairs

## Quick Start

```bash
# Install dependencies
pip install -e .

# Set environment for running scripts
export PYTHONPATH=src
```

## Preprocessing Pipeline

The `src/preprocess/` module provides a unified, extensible pipeline for data processing.

### Components

| Module | Description |
|--------|-------------|
| `schema_serializer.py` | Serialize schemas with ablation modes |
| `triplet_generator.py` | Generate contrastive learning triplets |
| `embedding_generator.py` | Batch embedding generation |
| `similarity_computer.py` | All-pairs similarity computation |
| `edge_filter.py` | Filter edges by threshold |
| `graph_builder.py` | Build DGL graphs |

### Schema Serialization with Ablation Modes

The `SchemaSerializer` supports three modes for ablation studies:

```python
from preprocess import SchemaSerializer
from model.WKDataset import WKDataset

loader = WKDataset(schema_dir="data/schema", csv_base_dir="data/unzip")

# Full mode (default) - backward compatible
serializer = SchemaSerializer(mode="full", sample_size=3)
text = serializer.serialize(loader, "00001")

# Schema only - table and column names
serializer = SchemaSerializer(mode="schema_only")
text = serializer.serialize(loader, "00001")

# Data only - sample values only
serializer = SchemaSerializer(mode="data_only", sample_size=3)
text = serializer.serialize(loader, "00001")
```

### Configuration

All hyperparameters centralized in `PreprocessConfig`:

```python
from preprocess import PreprocessConfig

config = PreprocessConfig(
    serialization_mode="full",  # "schema_only", "data_only", "full"
    sample_size=3,              # representative values per column
    num_negatives=6,            # negatives per triplet
    similarity_threshold=0.6713 # edge filtering threshold
)
```

### End-to-End Pipeline

```bash
# 1. Generate triplets for training
python -m preprocess.triplet_generator \
    --qid-pairs data/qid_pairs.csv \
    --negative-pool data/negative_candidates.csv \
    --output-dir data/split_triplets \
    --num-negatives 6

# 2. Train BGE-M3 embedding model
python src/script/train_bge_softmax.py

# 3. Generate embeddings for all databases
python -m preprocess.embedding_generator \
    --schema-dir data/schema \
    --csv-dir data/unzip \
    --output data/graph/all_embeddings.pt \
    --model-path out/model/best

# 4. Compute all-pairs similarity
python -m preprocess.similarity_computer \
    --embeddings data/graph/all_embeddings.pt \
    --output data/graph/all_predictions.pt \
    --threshold 0.6713

# 5. Filter edges at desired threshold
python -m preprocess.edge_filter \
    --predictions data/graph/all_predictions.pt \
    --output data/graph/filtered_edges.csv \
    --threshold 0.94

# 6. Build DGL graph
python -m preprocess.graph_builder \
    --edges data/graph/filtered_edges.csv \
    --embeddings data/graph/all_embeddings.pt \
    --output data/graph/graph.dgl
```

## BGE-M3 Embedding Model

### Training

The model is finetuned using InfoNCE loss with triplet data:

```bash
cd src && export PYTHONPATH=.
python script/train_bge_softmax.py
```

Training uses:
- **Data**: Triplets from `data/split_triplets/`
- **Loss**: InfoNCE with temperature=0.5
- **Optimizer**: AdamW with lr=1e-5
- **Output**: Finetuned model in `out/col_matcher_bge-m3_*/weights/`

### Evaluation

```bash
# Evaluate on test triplets with ROC/AUC
python -c "
from model.BGEEmbedder import BGEEmbedder
embedder = BGEEmbedder(model_path='out/model/best')
embedder.test_scalable(
    test_path='data/split_triplets/triplets_test_seed42.jsonl',
    embedding_path='data/graph/all_embeddings.pt',
    save_dir='out/test_results'
)
"
```

Outputs: `roc_curve.png`, `predictions.csv`, `summary.txt`

## Federated Learning Experiments

### Available Algorithms

| Algorithm | Type | Script |
|-----------|------|--------|
| FedAvg | Horizontal FL | `src/demo/train_fedavg.py` |
| FedProx | Horizontal FL | `src/demo/train_fedprox.py` |
| SCAFFOLD | Horizontal FL | `src/demo/train_scaffold.py` |
| FedOV | One-shot FL | `src/demo/train_fedov.py` |
| SplitNN | Vertical FL | `src/demo/train_splitnn.py` |

### Running Experiments

```bash
# Single experiment
python src/demo/train_fedavg.py --seed 0 --databases 02799 79665

# Automated validation pipeline
./run_automated_fl_validation.sh \
    --min-similarity 0.98 \
    --sample-size 2000 \
    --task-types "fedprox scaffold fedov"
```

### Generate Results Tables

```bash
python src/summary/run_and_generate_horizontal_table.py --show-std --num-seeds 5
python src/summary/run_and_generate_vertical_table.py --show-std --num-seeds 5
```

## Directory Structure

```
wikidbs/
├── src/
│   ├── preprocess/       # Unified preprocessing pipeline
│   ├── model/            # Neural network models
│   ├── demo/             # Training scripts
│   ├── analysis/         # Graph analysis
│   └── summary/          # Results aggregation
├── data/
│   ├── schema/           # Database schema JSON files
│   ├── unzip/            # Raw database CSV files
│   ├── split_triplets/   # Training triplets
│   └── graph/            # Embeddings and graphs
└── results/              # Experiment results
```

## Ablation Studies

To compare different schema representations:

```python
from preprocess import SchemaSerializer, PreprocessConfig

# Run embedding generation with different modes
for mode in ["schema_only", "data_only", "full"]:
    config = PreprocessConfig(serialization_mode=mode)
    # Generate embeddings and evaluate...
```

## Requirements

- Python 3.12+
- CUDA 12.1+
- PyTorch with CUDA support
- DGL (Deep Graph Library)
- RAPIDS cuML (optional, for GPU acceleration)

## License

[Add license information]
