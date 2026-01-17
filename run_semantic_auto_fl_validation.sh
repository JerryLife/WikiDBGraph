#!/bin/bash

#
# Semantic-Based Automated Federated Learning Validation Pipeline
#
# This script is a semantic variant of run_automated_fl_validation.sh
# It uses BGE embeddings for column alignment instead of string matching.
#
# The FL training step is IDENTICAL to the original for fair comparison.
#
# Usage: ./run_semantic_auto_fl_validation.sh [options]
#

# Note: Not using 'set -e' to allow partial preprocessing success

# Default parameters (ALIGNED with original run_automated_fl_validation.sh for fair comparison)
MIN_SIMILARITY=0.98
MAX_SIMILARITY=1.0
MIN_ROWS=100
SAMPLE_SIZE=2000
SEED=42
NUM_GPUS=2
GPU_IDS="2,3"  # Empty means use 0 to NUM_GPUS-1
MAX_CONCURRENT_PER_GPU=5
TIMEOUT=3600  # 1 hour default timeout
TASK_TYPES="fedprox scaffold fedov"  # Algorithms to compare (same as will be compared with raw)

# Semantic-specific parameters
SEMANTIC_THRESHOLD=0.80  # Similarity threshold for column matching
COLUMN_SAMPLE_SIZE=10     # Number of sample values per column for embedding

# Directories (different from original to avoid conflicts)
BASE_DIR=$(pwd)
DATA_DIR="data/auto_semantic"
OUTPUT_DIR="out/autorun_semantic"
LOG_DIR="out/autorun_semantic/logs"
RESULTS_DIR="out/autorun_semantic/results"

# Shared pairs file (can reuse from original pipeline)
PAIRS_FILE="out/autorun/sampled_pairs.json"
PREPROCESSING_SUMMARY="$DATA_DIR/preprocessing_summary.json"
EXECUTION_REPORT="$RESULTS_DIR/execution_report.json"

# Colors for output
RED='\\033[0;31m'
GREEN='\\033[0;32m'
YELLOW='\\033[1;33m'
BLUE='\\033[0;34m'
MAGENTA='\\033[0;35m'
NC='\\033[0m' # No Color

# Logging function
log() {
    echo -e "${MAGENTA}[SEMANTIC]${NC} ${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

log_success() {
    echo -e "${MAGENTA}[SEMANTIC]${NC} ${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] SUCCESS:${NC} $1"
}

log_warning() {
    echo -e "${MAGENTA}[SEMANTIC]${NC} ${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING:${NC} $1"
}

log_error() {
    echo -e "${MAGENTA}[SEMANTIC]${NC} ${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR:${NC} $1"
}

# Help function
show_help() {
    cat << EOF
Semantic-Based Automated Federated Learning Validation Pipeline

This script uses BGE embeddings for column alignment (vs string matching in original).
The FL training step is IDENTICAL to run_automated_fl_validation.sh for fair comparison.

Usage: $0 [OPTIONS]

OPTIONS:
    --min-similarity FLOAT    Minimum similarity threshold (default: $MIN_SIMILARITY)
    --max-similarity FLOAT    Maximum similarity threshold (default: $MAX_SIMILARITY)
    --min-rows INT            Minimum table rows requirement (default: $MIN_ROWS)
    --sample-size INT         Number of pairs to sample (default: $SAMPLE_SIZE)
    --seed INT                Random seed (default: $SEED)
    --num-gpus INT            Number of GPUs to use (default: $NUM_GPUS)
    --gpu-ids IDS             Comma-separated list of specific GPU IDs to use
    --max-concurrent INT      Max concurrent tasks per GPU (default: $MAX_CONCURRENT_PER_GPU)
    --timeout INT             Timeout in seconds (default: $TIMEOUT)
    --task-types TYPES        Space-separated list of algorithms to run
    --semantic-threshold FLOAT  Similarity threshold for column matching (default: $SEMANTIC_THRESHOLD)
    --column-sample-size INT  Sample values per column for embedding (default: $COLUMN_SAMPLE_SIZE)
    --skip-sampling           Skip pair sampling (uses existing pairs from original pipeline)
    --skip-preprocessing      Skip preprocessing step
    --skip-training           Skip training step
    --force-rerun             Delete all existing semantic data and results
    --debug                   Enable verbose debug output
    --help                    Show this help message

EXAMPLES:
    # Run semantic pipeline using same pairs as original
    $0
    
    # Run with higher similarity threshold for stricter matching
    $0 --semantic-threshold 0.85
    
    # Run only preprocessing (no FL training) for debugging
    $0 --skip-training
    
    # Force fresh preprocessing
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
        --semantic-threshold)
            SEMANTIC_THRESHOLD="$2"
            shift 2
            ;;
        --column-sample-size)
            COLUMN_SAMPLE_SIZE="$2"
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
        --debug)
            DEBUG_MODE=1
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
    IFS=',' read -ra GPU_ARRAY <<< "$GPU_IDS"
    NUM_GPUS=${#GPU_ARRAY[@]}
    log "Using specific GPUs: $GPU_IDS (count: $NUM_GPUS)"
    export CUDA_VISIBLE_DEVICES="$GPU_IDS"
    log "Set CUDA_VISIBLE_DEVICES=$GPU_IDS"
else
    log "Using GPUs 0-$((NUM_GPUS-1)) (count: $NUM_GPUS)"
    if [[ $NUM_GPUS -gt 0 ]]; then
        GPU_RANGE=$(seq -s, 0 $((NUM_GPUS-1)))
        export CUDA_VISIBLE_DEVICES="$GPU_RANGE"
        log "Set CUDA_VISIBLE_DEVICES=$GPU_RANGE"
    fi
fi

# Check if we're in the right directory
if [[ ! -f "src/autorun/semantic_data_preprocessor.py" ]]; then
    log_error "Please run this script from the project root directory"
    log_error "Make sure src/autorun/semantic_data_preprocessor.py exists"
    exit 1
fi

# Check Python environment
if ! python -c "import torch, sklearn, pandas, numpy" &> /dev/null; then
    log_error "Required Python packages not found. Please install dependencies."
    exit 1
fi

# Check GPU availability
if ! python -c "import torch; print(f'GPUs available: {torch.cuda.device_count()}')" 2>/dev/null | grep -q "GPUs available: [1-9]"; then
    log_warning "No CUDA GPUs detected. BGE embeddings may be slow."
fi

# Create directories
log "Creating output directories..."
mkdir -p "$DATA_DIR" "$OUTPUT_DIR" "$LOG_DIR" "$RESULTS_DIR"

# Handle force rerun - delete existing semantic data
if [[ -n "$FORCE_RERUN" ]]; then
    log_warning "Force rerun mode enabled - removing existing semantic data"
    
    if [[ -f "$PREPROCESSING_SUMMARY" ]]; then
        log "Removing existing preprocessing summary: $PREPROCESSING_SUMMARY"
        rm -f "$PREPROCESSING_SUMMARY"
    fi
    
    if [[ -d "$DATA_DIR" ]]; then
        log "Removing existing semantic data directory: $DATA_DIR"
        rm -rf "$DATA_DIR"
        mkdir -p "$DATA_DIR"
    fi
    
    if [[ -d "$RESULTS_DIR" ]]; then
        log "Removing existing semantic results directory: $RESULTS_DIR"
        rm -rf "$RESULTS_DIR"
        mkdir -p "$RESULTS_DIR"
    fi
    
    log_success "Cleaned up existing semantic data for fresh run"
fi

# Step 1: Require existing sampled pairs from original pipeline
# This script REQUIRES sampled_pairs.json from run_automated_fl_validation.sh
log "Step 1: Checking for existing sampled pairs from original pipeline..."

if [[ ! -f "$PAIRS_FILE" ]]; then
    log_error "Sampled pairs file not found: $PAIRS_FILE"
    log_error "Please run ./run_automated_fl_validation.sh first to generate pairs."
    log_error "This semantic script REUSES sampled pairs from the original pipeline for fair comparison."
    exit 1
fi

# Count pairs from sampled pairs file
sampled_pairs_count=$(python -c "
import json, os
try:
    with open('$PAIRS_FILE', 'r') as f:
        pairs = json.load(f)
    print(len(pairs))
except Exception:
    print(0)
" 2>/dev/null)

if [[ -z "$sampled_pairs_count" || "$sampled_pairs_count" -eq 0 ]]; then
    log_error "No pairs found in sampled pairs file: $PAIRS_FILE"
    log_error "Please run ./run_automated_fl_validation.sh first."
    exit 1
fi

log_success "Found $sampled_pairs_count sampled pairs in $PAIRS_FILE"
log "Semantic preprocessing will process these pairs using BGE embeddings for column alignment"

# Step 2: Semantic Preprocessing (key difference from original)
if [[ -z "$SKIP_PREPROCESSING" ]]; then
    if [[ -f "$PREPROCESSING_SUMMARY" ]]; then
        # Check if there are any successful preprocessed pairs
        successful_pairs=$(python -c "
import json, sys, os
try:
    if os.path.exists('$PREPROCESSING_SUMMARY'):
        with open('$PREPROCESSING_SUMMARY', 'r') as f:
            summary = json.load(f)
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
            log "Semantic preprocessing already exists with $successful_pairs successful pairs"
            log "Skipping preprocessing step (use --force-rerun to reprocess)"
        else
            log "Preprocessing summary exists but has no successful pairs, reprocessing..."
            NEED_PREPROCESSING=1
        fi
    else
        NEED_PREPROCESSING=1
    fi
    
    if [[ -n "$NEED_PREPROCESSING" ]]; then
        log "Step 2: Semantic preprocessing with BGE embeddings..."
        log "Parameters: semantic_threshold=$SEMANTIC_THRESHOLD, column_sample_size=$COLUMN_SAMPLE_SIZE"
        
        cd "$BASE_DIR"
        export PYTHONPATH=src
        
        # Build debug flag for python script
        DEBUG_FLAG=""
        if [[ -n "$DEBUG_MODE" ]]; then
            DEBUG_FLAG="--debug"
        fi
        
        # Run semantic preprocessing using sampled pairs file (same as original pipeline)
        python src/autorun/semantic_data_preprocessor.py \
            --input "$PAIRS_FILE" \
            --output-dir "$DATA_DIR" \
            --test-size 0.2 \
            --random-state "$SEED" \
            --min-label-variance 0.01 \
            --max-missing-ratio 0.5 \
            --similarity-threshold "$SEMANTIC_THRESHOLD" \
            --column-sample-size "$COLUMN_SAMPLE_SIZE" \
            $DEBUG_FLAG \
            2>&1 | tee "$LOG_DIR/semantic_preprocessing.log"
        
        preprocessing_exit_code=${PIPESTATUS[0]}
        
        if [[ $preprocessing_exit_code -ne 0 ]]; then
            log_warning "Semantic preprocessing finished with non-zero exit code ($preprocessing_exit_code)"
            log_warning "This is expected if some pairs fail. Checking summary file..."
        else
            log_success "Semantic preprocessing completed"
        fi
        
        # Check for successful pairs
        successful_pairs=$(python -c "
import json, sys, os
try:
    if os.path.exists('$PREPROCESSING_SUMMARY'):
        with open('$PREPROCESSING_SUMMARY', 'r') as f:
            summary = json.load(f)
        stats = summary.get('summary_stats', {})
        print(stats.get('successful', summary.get('processed_pairs', 0)))
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
            log_error "No pairs were successfully preprocessed. Check '$LOG_DIR/semantic_preprocessing.log'."
            log_warning "Skipping the training step as there is no data to train on."
            SKIP_TRAINING=1
        fi
    fi
else
    log "Skipping semantic preprocessing step"
fi

# Step 3: Run parallel FL training (IDENTICAL to original)
if [[ -z "$SKIP_TRAINING" ]]; then
    if [[ -f "$EXECUTION_REPORT" ]]; then
        log "Execution report already exists, skipping training"
    else
        log "Step 3: Running parallel FL training (same as original)..."
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
        
        # Execute GPU scheduler (IDENTICAL to original)
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
fi

# Final summary
log "==============================================="
log "SEMANTIC FL VALIDATION COMPLETED"
log "==============================================="

# Count preprocessing results
if [[ -f "$PREPROCESSING_SUMMARY" ]]; then
    log "Semantic Preprocessing Results Summary:"
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
        
except Exception as e:
    print(f'  Error reading execution report: {e}')
"
fi

# Show output locations
log "Output Locations:"
log "  Sampled pairs: $PAIRS_FILE"
log "  Semantic processed data: $DATA_DIR"
log "  Training results: $RESULTS_DIR"
log "  Logs: $LOG_DIR"
log "  Execution report: $EXECUTION_REPORT"

# Check for results files
result_count=$(find "$RESULTS_DIR" -name "*.json" -type f 2>/dev/null | wc -l)
log "Total result files generated: $result_count"

# Comparison hint
log ""
log "To compare with original (string-based) approach:"
log "  1. Run: ./run_automated_fl_validation.sh --sample-size $SAMPLE_SIZE"
log "  2. Compare: diff data/auto/preprocessing_summary.json $PREPROCESSING_SUMMARY"
log ""

log "Done."
