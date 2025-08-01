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
MAX_CONCURRENT_PER_GPU=4
TIMEOUT=3600  # 1 hour default timeout

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
    --max-concurrent INT      Max concurrent tasks per GPU (default: $MAX_CONCURRENT_PER_GPU)
    --timeout INT             Timeout in seconds (default: $TIMEOUT)
    --skip-sampling           Skip pair sampling step
    --skip-preprocessing      Skip data preprocessing step
    --skip-training           Skip training step
    --resume                  Resume from last successful step
    --help                    Show this help message

EXAMPLES:
    # Run with default parameters
    $0
    
    # Run with custom similarity range and sample size
    $0 --min-similarity 0.95 --max-similarity 0.99 --sample-size 100
    
    # Resume from preprocessing (if sampling already done)
    $0 --skip-sampling
    
    # Only run training on already preprocessed data
    $0 --skip-sampling --skip-preprocessing

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
        --max-concurrent)
            MAX_CONCURRENT_PER_GPU="$2"
            shift 2
            ;;
        --timeout)
            TIMEOUT="$2"
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

# Step 1: Sample database pairs
if [[ -z "$SKIP_SAMPLING" ]]; then
    if should_skip_step "pair sampling" "$PAIRS_FILE"; then
        log "Skipping pair sampling step"
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
    if should_skip_step "data preprocessing" "$PREPROCESSING_SUMMARY"; then
        log "Skipping data preprocessing step"
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
        
        python src/autorun/gpu_scheduler.py \
            --preprocessing-summary "$PREPROCESSING_SUMMARY" \
            --data-dir "$DATA_DIR" \
            --num-gpus "$NUM_GPUS" \
            --max-concurrent-per-gpu "$MAX_CONCURRENT_PER_GPU" \
            --output-dir "$RESULTS_DIR" \
            --log-dir "$LOG_DIR" \
            --timeout "$TIMEOUT" \
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