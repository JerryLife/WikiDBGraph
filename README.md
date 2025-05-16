# WikiDBGraph

A tool for analyzing and visualizing Wikidata relationships.

## Project Structure

The project is organized into the following main directories:

- `src/`
  - `analysis/` - Contains analysis tools and scripts
    - `EdgeProperties.py` - Analyzes relationships between database schemas
    - `NodeProperties.py` - Analyzes individual database properties
    - `add_properties.py` - Adds computed properties to the graph
    - `build_raw_graph.py` - Constructs the initial graph structure
    - `filter_edges.py` - Filters graph edges based on criteria
  - `model/` - Data models and graph structures
  - `script/` - Utility scripts for data processing
  - `summary/` - Summary generation and reporting tools
  - `utils/` - Helper functions and utilities

## Installation

This project requires Python 3.12 or higher. Install using pip:

```bash
pip install .
```

### Dependencies

The project depends on the following main packages:
- torch ~=2.4.0
- torchvision
- torchdata
- dgl (with CUDA 12.1 support)
- numpy
- pandas
- matplotlib
- scikit-learn
- orjson
- tqdm

## Analysis Components

The analysis module provides several key functionalities:

1. **Edge Properties Analysis** (`EdgeProperties.py`):
   - Calculates structural properties between database schemas
   - Computes Jaccard indices for table names, columns, and data types
   - Measures graph edit distance between schemas
   - Analyzes data type distributions using Hellinger distance

2. **Node Properties Analysis** (`NodeProperties.py`):
   - Analyzes individual database properties
   - Processes database, table, and column information
   - Manages foreign key relationships

3. **Graph Construction and Processing**:
   - `build_raw_graph.py`: Creates the initial graph structure
   - `add_properties.py`: Enhances the graph with computed properties
   - `filter_edges.py`: Applies filtering criteria to graph edges

## Usage

[Usage instructions to be added]

## Features

- Graph-based analysis of Wikipedia database relationships
- Structural and semantic property analysis
- Statistical analysis of database schemas
- Graph visualization capabilities

## Requirements

- Python >= 3.12
- CUDA 12.1 (for GPU acceleration)
- See `pyproject.toml` for complete dependency list


## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
