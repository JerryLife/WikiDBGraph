# Automated Federated Learning Validation System

This system provides a comprehensive pipeline for automated federated learning validation experiments on database pairs with high similarity.

## Overview

The system performs the following steps:

1. **Pair Sampling**: Samples database pairs within a specified similarity range (0.98-1.0 by default)
2. **Data Preprocessing**: Automatically joins tables, selects regression labels, cleans data, and splits into train/test sets
3. **Parallel Training**: Runs FedAvg, Solo, and Combined training experiments in parallel across multiple GPUs
4. **Results Collection**: Aggregates results and generates comprehensive reports

## Components

### 1. Pair Sampler (`pair_sampler.py`)
- Filters database pairs by similarity threshold
- Validates data quality (minimum rows, common columns)
- Samples specified number of pairs for experiments
- Outputs: `sampled_pairs.json`

### 2. Data Preprocessor (`data_preprocessor.py`)  
- Loads and joins all tables from each database
- Identifies common columns between database pairs
- Automatically selects regression labels with sufficient variance
- Cleans data (fills missing values, normalizes labels to [0,1])
- Splits data into train/test sets
- Outputs: Individual pair directories with train/test CSV files and config.json

### 3. Enhanced FedAvg Trainer (`enhanced_fedavg.py`)
- Parameterized version of the original training scripts
- Supports regression tasks with MSE, RMSE, MAE, and R² metrics
- Runs three training modes:
  - **FedAvg**: Federated averaging across clients
  - **Solo**: Individual client training
  - **Combined**: Centralized training on combined data
- Outputs: Results JSON with all metrics and training history

### 4. GPU Scheduler (`gpu_scheduler.py`)
- Manages parallel execution across multiple GPUs
- Load balancing with configurable concurrent tasks per GPU
- Task queuing and status monitoring
- Automatic retry and error handling
- Outputs: Execution logs and final report

### 5. Main Orchestration Script (`run_automated_fl_validation.sh`)
- Coordinates the entire pipeline
- Supports resume functionality and step skipping
- Comprehensive logging and error handling
- GPU utilization monitoring

## Usage

### Basic Usage
```bash
# Run with default parameters (200 pairs, similarity 0.98-1.0)
./run_automated_fl_validation.sh
```

### Custom Parameters
```bash
# Custom similarity range and sample size
./run_automated_fl_validation.sh \\
    --min-similarity 0.95 \\
    --max-similarity 0.99 \\
    --sample-size 100 \\
    --num-gpus 4 \\
    --max-concurrent 2
```

### Resume/Skip Steps
```bash
# Resume from last successful step
./run_automated_fl_validation.sh --resume

# Skip sampling if pairs already exist
./run_automated_fl_validation.sh --skip-sampling

# Only run training on preprocessed data
./run_automated_fl_validation.sh --skip-sampling --skip-preprocessing
```

## Parameters

### Sampling Parameters
- `--min-similarity`: Minimum similarity threshold (default: 0.98)
- `--max-similarity`: Maximum similarity threshold (default: 1.0)  
- `--min-rows`: Minimum table rows requirement (default: 100)
- `--sample-size`: Number of pairs to sample (default: 200)

### GPU Parameters
- `--num-gpus`: Number of GPUs to use (default: 4)
- `--max-concurrent`: Max concurrent tasks per GPU (default: 2)
- `--timeout`: Training timeout in seconds (default: 7200)

### Other Parameters
- `--seed`: Random seed for reproducibility (default: 42)

## Output Structure

```
out/autorun/
├── sampled_pairs.json              # Sampled database pairs
├── logs/                           # All execution logs
│   ├── sampling.log
│   ├── preprocessing.log  
│   ├── training.log
│   └── {pair_id}_{task}_gpu{id}.log
└── results/                        # Training results
    ├── {pair_id}_fedavg_results.json
    └── execution_report.json       # Final summary

data/auto/
├── preprocessing_summary.json      # Preprocessing overview
└── {db_id1}_{db_id2}/             # Individual pair data
    ├── config.json                 # Pair configuration
    ├── {db_id1}_train.csv         # Training data
    ├── {db_id1}_test.csv          # Test data
    ├── {db_id2}_train.csv
    └── {db_id2}_test.csv
```

## Result Format

Each experiment generates a comprehensive results file:

```json
{
  "pair_id": "02799_79665",
  "db_id1": 2799,
  "db_id2": 79665,
  "similarity": 0.9894,
  "label_column": "encoded_protein_length",
  "num_features": 15,
  "results": {
    "fedavg": {
      "mse": 0.0234,
      "rmse": 0.1529,
      "mae": 0.1203,
      "r2": 0.8567
    },
    "solo": {
      "client_0": {"mse": 0.0456, "r2": 0.7234},
      "client_1": {"mse": 0.0389, "r2": 0.7891}
    },
    "combined": {
      "mse": 0.0198,
      "rmse": 0.1407,
      "mae": 0.1156,
      "r2": 0.8798
    }
  }
}
```

## Performance Optimization

### GPU Utilization
- The system supports up to 4 GPUs by default
- Each GPU can run multiple concurrent experiments
- Automatic load balancing distributes tasks evenly
- Memory management prevents OOM errors

### Parallel Processing
- Data preprocessing is sequential but optimized
- Training runs in parallel across all available GPUs
- Task queuing ensures continuous GPU utilization
- Automatic cleanup of completed tasks

### Resource Monitoring
- Real-time GPU utilization tracking
- Memory usage monitoring
- Automatic task termination on timeout
- Comprehensive logging for debugging

## Error Handling

### Robustness Features
- Automatic retry on transient failures
- Graceful handling of corrupted data
- Skip corrupted database pairs
- Continue processing remaining pairs on individual failures

### Logging
- Comprehensive logs for each step
- Individual log files per training task
- Error messages with context
- Performance metrics and timing

## Customization

### Extending Training Methods
Add new training methods by:
1. Creating a new trainer module
2. Adding it to the GPU scheduler
3. Modifying the orchestration script

### Custom Metrics
Extend the results format by:
1. Adding metrics to the enhanced trainer
2. Updating the results schema
3. Modifying analysis scripts

### Different Data Types
Adapt for other data types by:
1. Modifying the data preprocessor
2. Adjusting label selection criteria
3. Updating training objectives

## Requirements

### System Requirements
- Linux/Unix environment
- Python 3.8+
- CUDA-capable GPUs (recommended)
- Sufficient disk space for processed data

### Python Dependencies
- torch
- sklearn
- pandas
- numpy
- Other dependencies from requirements.txt

### Data Requirements
- WikiDB database dump in data/unzip/
- Graph similarity data in data/graph/
- Sufficient database pairs with required similarity

## Troubleshooting

### Common Issues
1. **No suitable pairs found**: Adjust similarity thresholds or min_rows requirement
2. **GPU memory errors**: Reduce max_concurrent_per_gpu or model size
3. **Preprocessing failures**: Check data quality and column alignment
4. **Training timeouts**: Increase timeout or reduce model complexity

### Debug Steps
1. Check logs in out/autorun/logs/
2. Verify GPU availability with nvidia-smi
3. Test with smaller sample size first
4. Use --resume to continue from last successful step

## Citation

If you use this system in your research, please cite:
[Paper citation will be added when published]