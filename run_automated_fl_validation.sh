#!/bin/bash

#
# Automated Federated Learning Validation Pipeline
#
# This script orchestrates the complete FL validation process:
# 1. Sample database pairs with similarity filtering
# 2. Preprocess data for FL training
# 3. Run parallel FL experiments on multiple GPUs
#
# Usage: ./run_automated_fl_validation.sh [options]
#

# Note: Not using 'set -e' to allow partial preprocessing success

# Default parameters
MIN_SIMILARITY=0.98
MAX_SIMILARITY=1.0
MIN_ROWS=100
SAMPLE_SIZE=2000
SEED=42
NUM_GPUS=4
GPU_IDS="0,1,2,3"  # Empty means use 0 to NUM_GPUS-1
MAX_CONCURRENT_PER_GPU=5
TIMEOUT=3600  # 1 hour default timeout
TASK_TYPES="fedprox scaffold fedov"  # Default: all algorithms except fedtree

# Directories
BASE_DIR=$(pwd)
DATA_DIR="data/auto"
OUTPUT_DIR="out/autorun"
LOG_DIR="out/autorun/logs"
RESULTS_DIR="out/autorun/results"

# Files
PAIRS_FILE="$OUTPUT_DIR/sampled_pairs.json"
PREPROCESSING_SUMMARY="$DATA_DIR/preprocessing_summary.json"
EXECUTION_REPORT="$RESULTS_DIR/execution_report.json"

# Colors for output
RED='\\033[0;31m'
GREEN='\\033[0;32m'
YELLOW='\\033[1;33m'
BLUE='\\033[0;34m'
NC='\\033[0m' # No Color

# Logging function
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] SUCCESS:${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING:${NC} $1"
}

log_error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR:${NC} $1"
}

# Help function
show_help() {
    cat << EOF
Automated Federated Learning Validation Pipeline

Usage: $0 [OPTIONS]

OPTIONS:
    --min-similarity FLOAT    Minimum similarity threshold (default: $MIN_SIMILARITY)
    --max-similarity FLOAT    Maximum similarity threshold (default: $MAX_SIMILARITY)
    --min-rows INT            Minimum table rows requirement (default: $MIN_ROWS)
    --sample-size INT         Number of pairs to sample (default: $SAMPLE_SIZE)
    --seed INT                Random seed (default: $SEED)
    --num-gpus INT            Number of GPUs to use (default: $NUM_GPUS)
                              Only used if --gpu-ids is not specified
    --gpu-ids IDS             Comma-separated list of specific GPU IDs to use (e.g., "0,2,5")
                              Sets CUDA_VISIBLE_DEVICES to restrict visible GPUs
                              If specified, overrides --num-gpus (count is auto-detected)
                              Default: use GPUs 0 to NUM_GPUS-1
    --max-concurrent INT      Max concurrent tasks per GPU (default: $MAX_CONCURRENT_PER_GPU)
    --timeout INT             Timeout in seconds (default: $TIMEOUT)
    --task-types TYPES        Space-separated list of algorithms to run
                              Available: fedavg, fedprox, scaffold, fedov, fedtree, solo, combined
                              Special: 'all' runs all algorithms including fedtree
                              Default: "$TASK_TYPES"
    --skip-sampling           Force skip pair sampling (auto-skips if pairs.json exists)
    --skip-preprocessing      Force skip preprocessing (auto-skips if preprocessing_summary.json exists)
    --skip-training           Skip training step
    --force-rerun             Delete all existing data and results, then run from scratch
    --resume                  Resume from last successful step (deprecated, auto-detection is default)
    --help                    Show this help message

EXAMPLES:
    # Run with default parameters (all algorithms except fedtree)
    $0
    
    # Run with custom similarity range and sample size
    $0 --min-similarity 0.95 --max-similarity 0.99 --sample-size 100
    
    # Run only the new FL algorithms (FedProx, SCAFFOLD, FedOV)
    $0 --task-types "fedprox scaffold fedov solo combined"
    
    # Run only FedProx
    $0 --task-types "fedprox solo combined"
    
    # Run all algorithms including FedTree
    $0 --task-types all
    
    # Run only FedTree with solo and combined baselines
    $0 --task-types "fedtree solo combined"
    
    # Use specific GPUs (0, 2, and 5) via CUDA_VISIBLE_DEVICES
    $0 --gpu-ids "0,2,5"
    
    # Use only GPU 1 (useful when other GPUs are busy)
    $0 --gpu-ids "1"
    
    # Use GPUs 2 and 3 with FedProx only
    $0 --gpu-ids "2,3" --task-types "fedprox solo combined"
    
    # Run again (auto-detects existing pairs and preprocessed data)
    $0
    
    # Force resample and reprocess (ignoring existing data)
    $0 --force-rerun

OUTPUT:
    Results will be saved in: $RESULTS_DIR
    Logs will be saved in: $LOG_DIR
    Processed data will be in: $DATA_DIR

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --min-similarity)
            MIN_SIMILARITY="$2"
            shift 2
            ;;
        --max-similarity)
            MAX_SIMILARITY="$2"
            shift 2
            ;;
        --min-rows)
            MIN_ROWS="$2"
            shift 2
            ;;
        --sample-size)
            SAMPLE_SIZE="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --num-gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --gpu-ids)
            GPU_IDS="$2"
            shift 2
            ;;
        --max-concurrent)
            MAX_CONCURRENT_PER_GPU="$2"
            shift 2
            ;;
        --timeout)
            TIMEOUT="$2"
            shift 2
            ;;
        --task-types)
            TASK_TYPES="$2"
            shift 2
            ;;
        --skip-sampling)
            SKIP_SAMPLING=1
            shift
            ;;
        --skip-preprocessing)
            SKIP_PREPROCESSING=1
            shift
            ;;
        --skip-training)
            SKIP_TRAINING=1
            shift
            ;;
        --force-rerun)
            FORCE_RERUN=1
            shift
            ;;
        --resume)
            RESUME=1
            shift
            ;;
        --help)
            show_help
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Process task types
if [[ "$TASK_TYPES" == "all" ]]; then
    TASK_TYPES="fedavg fedprox scaffold fedov fedtree solo combined"
    log "Running all algorithms including FedTree"
fi

# Process GPU configuration
if [[ -n "$GPU_IDS" ]]; then
    # GPU IDs specified, convert to array and count
    IFS=',' read -ra GPU_ARRAY <<< "$GPU_IDS"
    NUM_GPUS=${#GPU_ARRAY[@]}
    log "Using specific GPUs: $GPU_IDS (count: $NUM_GPUS)"
    # Set CUDA_VISIBLE_DEVICES to restrict GPU visibility
    export CUDA_VISIBLE_DEVICES="$GPU_IDS"
    log "Set CUDA_VISIBLE_DEVICES=$GPU_IDS"
else
    log "Using GPUs 0-$((NUM_GPUS-1)) (count: $NUM_GPUS)"
    # Optionally set CUDA_VISIBLE_DEVICES to sequential GPUs
    if [[ $NUM_GPUS -gt 0 ]]; then
        GPU_RANGE=$(seq -s, 0 $((NUM_GPUS-1)))
        export CUDA_VISIBLE_DEVICES="$GPU_RANGE"
        log "Set CUDA_VISIBLE_DEVICES=$GPU_RANGE"
    fi
fi

# Check if we're in the right directory
if [[ ! -f "src/autorun/pair_sampler.py" ]]; then
    log_error "Please run this script from the project root directory"
    exit 1
fi

# Check Python environment
if ! python -c "import torch, sklearn, pandas, numpy" &> /dev/null; then
    log_error "Required Python packages not found. Please install dependencies."
    exit 1
fi

# Check GPU availability
if ! python -c "import torch; print(f'GPUs available: {torch.cuda.device_count()}')" 2>/dev/null | grep -q "GPUs available: [1-9]"; then
    log_warning "No CUDA GPUs detected. Training may be slow."
fi

# Create directories
log "Creating output directories..."
mkdir -p "$DATA_DIR" "$OUTPUT_DIR" "$LOG_DIR" "$RESULTS_DIR"

# Function to check if step should be skipped
should_skip_step() {
    local step=$1
    local file=$2
    
    if [[ -n "$RESUME" && -f "$file" ]]; then
        log_warning "Resuming: skipping $step (output exists: $file)"
        return 0
    fi
    
    return 1
}

# Handle force rerun - delete existing data
if [[ -n "$FORCE_RERUN" ]]; then
    log_warning "Force rerun mode enabled - removing existing data"
    
    if [[ -f "$PAIRS_FILE" ]]; then
        log "Removing existing pairs file: $PAIRS_FILE"
        rm -f "$PAIRS_FILE"
    fi
    
    if [[ -f "$PREPROCESSING_SUMMARY" ]]; then
        log "Removing existing preprocessing summary: $PREPROCESSING_SUMMARY"
        rm -f "$PREPROCESSING_SUMMARY"
    fi
    
    if [[ -d "$DATA_DIR" ]]; then
        log "Removing existing data directory: $DATA_DIR"
        rm -rf "$DATA_DIR"
    fi
    
    if [[ -d "$RESULTS_DIR" ]]; then
        log "Removing existing results directory: $RESULTS_DIR"
        rm -rf "$RESULTS_DIR"
    fi
    
    log_success "Cleaned up existing data for fresh run"
fi

# Step 1: Sample database pairs
if [[ -z "$SKIP_SAMPLING" ]]; then
    # Auto-detect: skip if pairs file already exists
    if [[ -f "$PAIRS_FILE" ]]; then
        log "Pairs file already exists: $PAIRS_FILE"
        log "Skipping pair sampling step (use --skip-sampling to suppress this check)"
    else
        log "Step 1: Sampling database pairs..."
        log "Parameters: similarity=[$MIN_SIMILARITY, $MAX_SIMILARITY], min_rows=$MIN_ROWS, sample_size=$SAMPLE_SIZE"
        
        cd "$BASE_DIR"
        export PYTHONPATH=src
        
        python src/autorun/pair_sampler.py \
            --min-similarity "$MIN_SIMILARITY" \
            --max-similarity "$MAX_SIMILARITY" \
            --min-rows "$MIN_ROWS" \
            --sample-size "$SAMPLE_SIZE" \
            --seed "$SEED" \
            --output "$PAIRS_FILE" \
            2>&1 | tee "$LOG_DIR/sampling.log"
        
        sampling_exit_code=${PIPESTATUS[0]}
        
        if [[ $sampling_exit_code -eq 0 ]]; then
            log_success "Pair sampling completed successfully"
        else
            log_error "Pair sampling failed (exit code: $sampling_exit_code)"
            exit 1
        fi
    fi
else
    log "Skipping pair sampling step"
fi

# Step 2: Preprocess data
if [[ -z "$SKIP_PREPROCESSING" ]]; then
    # Auto-detect: skip if preprocessing summary exists with successful results
    if [[ -f "$PREPROCESSING_SUMMARY" ]]; then
        # Check if there are any successful preprocessed pairs
        successful_pairs=$(python -c "
import json, sys, os
try:
    if os.path.exists('$PREPROCESSING_SUMMARY'):
        with open('$PREPROCESSING_SUMMARY', 'r') as f:
            summary = json.load(f)
        # Check summary_stats.successful field
        stats = summary.get('summary_stats', {})
        print(stats.get('successful', 0))
    else:
        print(0)
except Exception:
    print(0)
" 2>/dev/null)
        
        if [[ -z "$successful_pairs" ]]; then
            successful_pairs=0
        fi
        
        if [[ "$successful_pairs" -gt 0 ]]; then
            log "Preprocessing summary already exists with $successful_pairs successful pairs"
            log "Data directory: $DATA_DIR"
            log "Skipping preprocessing step (use --skip-preprocessing to suppress this check)"
        else
            log "Preprocessing summary exists but has no successful pairs, reprocessing..."
            log "Step 2: Preprocessing data for FL training..."
            
            if [[ ! -f "$PAIRS_FILE" ]]; then
                log_error "Pairs file not found: $PAIRS_FILE"
                log_error "Please run sampling step first or provide existing pairs file"
                exit 1
            fi
            
            cd "$BASE_DIR"
            export PYTHONPATH=src
            
            python src/autorun/data_preprocessor.py \
                --input "$PAIRS_FILE" \
                --output-dir "$DATA_DIR" \
                --test-size 0.2 \
                --random-state "$SEED" \
                --min-label-variance 0.01 \
                --max-missing-ratio 0.5 \
                2>&1 | tee "$LOG_DIR/preprocessing.log"
            
            preprocessing_exit_code=${PIPESTATUS[0]}
            
            # Log status based on exit code
            if [[ $preprocessing_exit_code -ne 0 ]]; then
                log_warning "Data preprocessing script finished with a non-zero exit code ($preprocessing_exit_code)."
                log_warning "This is expected if some pairs fail. Checking summary file for successes..."
            else
                log_success "Data preprocessing script finished successfully."
            fi
            
            # Re-check successful pairs after reprocessing
            successful_pairs=$(python -c "
import json, sys, os
try:
    if os.path.exists('$PREPROCESSING_SUMMARY'):
        with open('$PREPROCESSING_SUMMARY', 'r') as f:
            summary = json.load(f)
        # Check summary_stats.successful field
        stats = summary.get('summary_stats', {})
        print(stats.get('successful', 0))
    else:
        print(0)
except Exception:
    print(0)
" 2>/dev/null)

            if [[ -z "$successful_pairs" ]]; then
                successful_pairs=0
            fi

            if [[ "$successful_pairs" -gt 0 ]]; then
                log_success "Found $successful_pairs successfully preprocessed pairs. Proceeding to training."
            else
                log_error "No pairs were successfully preprocessed. Check '$LOG_DIR/preprocessing.log'."
                log_warning "Skipping the training step as there is no data to train on."
                SKIP_TRAINING=1
            fi
        fi
    else
        log "Step 2: Preprocessing data for FL training..."
        
        if [[ ! -f "$PAIRS_FILE" ]]; then
            log_error "Pairs file not found: $PAIRS_FILE"
            log_error "Please run sampling step first or provide existing pairs file"
            exit 1
        fi
        
        cd "$BASE_DIR"
        export PYTHONPATH=src
        
        python src/autorun/data_preprocessor.py \
            --input "$PAIRS_FILE" \
            --output-dir "$DATA_DIR" \
            --test-size 0.2 \
            --random-state "$SEED" \
            --min-label-variance 0.01 \
            --max-missing-ratio 0.5 \
            2>&1 | tee "$LOG_DIR/preprocessing.log"
        
        preprocessing_exit_code=${PIPESTATUS[0]}
        
        # Log status based on exit code, but don't exit or skip yet.
        if [[ $preprocessing_exit_code -ne 0 ]]; then
            log_warning "Data preprocessing script finished with a non-zero exit code ($preprocessing_exit_code)."
            log_warning "This is expected if some pairs fail. Checking summary file for successes..."
        else
            log_success "Data preprocessing script finished successfully."
        fi

        # Now, decide whether to continue based on the *results* in the summary file.
        # This is the single source of truth for continuing to the training step.
        successful_pairs=$(python -c "
import json, sys, os
try:
    if os.path.exists('$PREPROCESSING_SUMMARY'):
        with open('$PREPROCESSING_SUMMARY', 'r') as f:
            summary = json.load(f)
        print(summary.get('processed_pairs', 0))
    else:
        print(0)
except Exception:
    print(0)
" 2>/dev/null)

        # Handle the case where the python command itself failed
        if [[ -z "$successful_pairs" ]]; then
            successful_pairs=0
        fi

        if [[ "$successful_pairs" -gt 0 ]]; then
            log_success "Found $successful_pairs successfully preprocessed pairs. Proceeding to training."
        else
            log_error "No pairs were successfully preprocessed. Check '$LOG_DIR/preprocessing.log'."
            log_warning "Skipping the training step as there is no data to train on."
            SKIP_TRAINING=1
        fi
    fi
else
    log "Skipping data preprocessing step"
fi

# Step 3: Run parallel FL training
if [[ -z "$SKIP_TRAINING" ]]; then
    if should_skip_step "FL training" "$EXECUTION_REPORT"; then
        log "Skipping FL training step"
    else
        log "Step 3: Running parallel FL training..."
        log "GPU configuration: $NUM_GPUS GPUs, max $MAX_CONCURRENT_PER_GPU concurrent tasks per GPU"
        log "Algorithms to run: $TASK_TYPES"
        
        if [[ ! -f "$PREPROCESSING_SUMMARY" ]]; then
            log_error "Preprocessing summary not found: $PREPROCESSING_SUMMARY"
            log_error "Please run preprocessing step first"
            exit 1
        fi
        
        cd "$BASE_DIR"
        export PYTHONPATH=src
        
        # Show GPU status
        if command -v nvidia-smi &> /dev/null; then
            log "Current GPU status:"
            nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
        fi
        
        # Note: GPU selection is controlled via CUDA_VISIBLE_DEVICES environment variable
        # which was set above based on --gpu-ids parameter
        # The GPU scheduler will only see the GPUs specified in CUDA_VISIBLE_DEVICES
        log "Environment: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-not set}"
        
        # Execute GPU scheduler (CUDA_VISIBLE_DEVICES already set above)
        python src/autorun/gpu_scheduler.py \
            --preprocessing-summary "$PREPROCESSING_SUMMARY" \
            --data-dir "$DATA_DIR" \
            --num-gpus "$NUM_GPUS" \
            --max-concurrent-per-gpu "$MAX_CONCURRENT_PER_GPU" \
            --output-dir "$RESULTS_DIR" \
            --log-dir "$LOG_DIR" \
            --timeout "$TIMEOUT" \
            --task-types $TASK_TYPES \
            2>&1 | tee "$LOG_DIR/training.log"
        
        training_exit_code=${PIPESTATUS[0]}
        
        if [[ $training_exit_code -eq 0 ]]; then
            log_success "FL training completed successfully"
        else
            log_warning "FL training completed with some failures (exit code: $training_exit_code)"
        fi
    fi
else
    log "Skipping FL training step"
    if [[ -n "$SKIP_TRAINING" ]]; then
        log_warning "Training was skipped due to preprocessing failures"
    fi
fi

# Final summary
log "==============================================="
log "AUTOMATED FL VALIDATION COMPLETED"
log "==============================================="

# Count preprocessing results
if [[ -f "$PREPROCESSING_SUMMARY" ]]; then
    log "Preprocessing Results Summary:"
    python -c "
import json, os
try:
    with open('$PREPROCESSING_SUMMARY', 'r') as f:
        summary = json.load(f)
    
    processed = summary.get('processed_pairs', 0)
    failed = summary.get('failed_pairs', 0)
    total = processed + failed
    
    print(f'  Total pairs attempted: {total}')
    print(f'  Successfully processed: {processed}')
    print(f'  Failed preprocessing: {failed}')
    
    if total > 0:
        success_rate = (processed / total) * 100
        print(f'  Success rate: {success_rate:.1f}%')
        
except Exception as e:
    print(f'  Error reading preprocessing summary: {e}')
"
fi

# Count training results
if [[ -f "$EXECUTION_REPORT" ]]; then
    log "Training Results Summary:"
    python -c "
import json
try:
    with open('$EXECUTION_REPORT', 'r') as f:
        report = json.load(f)
        
    completed = len(report.get('completed_tasks', []))
    failed = len(report.get('failed_tasks', []))
    
    print(f'  Completed training tasks: {completed}')
    print(f'  Failed training tasks: {failed}')

    if 'summary' in report:
        summary = report['summary']
        queue_size = summary.get('queue_size', 0)
        running = summary.get('running_tasks', 0)
        print(f'  Queue remaining: {queue_size}')
        print(f'  Still running: {running}')
        
except Exception as e:
    print(f'  Error reading execution report: {e}')
"
fi

# Show output locations
log "Output Locations:"
log "  Sampled pairs: $PAIRS_FILE"
log "  Processed data: $DATA_DIR"
log "  Training results: $RESULTS_DIR"
log "  Logs: $LOG_DIR"
log "  Execution report: $EXECUTION_REPORT"

# Check for results files
result_count=$(find "$RESULTS_DIR" -name "*.json" -type f | wc -l)
log "Total result files generated: $result_count"

# Determine overall success based on results
if [[ $result_count -gt 0 ]]; then
    log_success "Pipeline completed successfully with $result_count result files!"
    
    # Check if we had partial preprocessing failures
    if [[ -f "$PREPROCESSING_SUMMARY" ]]; then
        failed_pairs=$(python -c "
import json
try:
    with open('$PREPROCESSING_SUMMARY', 'r') as f:
        summary = json.load(f)
    print(summary.get('failed_pairs', 0))
except:
    print(0)
" 2>/dev/null || echo "0")
        
        if [[ "$failed_pairs" -gt 0 ]]; then
            log_warning "Note: Some pairs failed preprocessing but training proceeded on successful pairs"
        fi
    fi
    
    log "Next steps:"
    log "  1. Analyze results in: $RESULTS_DIR"
    log "  2. Check logs for detailed information: $LOG_DIR"
    log "  3. Use results for downstream analysis or paper writing"
    if [[ "$failed_pairs" -gt 0 ]]; then
        log "  4. Review preprocessing errors in: $DATA_DIR/preprocessing_errors.log"
    fi
else
    log_warning "Pipeline completed but no result files found"
    log "This could indicate:"
    log "  1. All preprocessing failed - check: $DATA_DIR/preprocessing_errors.log"
    log "  2. Training failed - check: $LOG_DIR/training.log"
    log "  3. Configuration issues - check logs in: $LOG_DIR"
fi

log "Done."