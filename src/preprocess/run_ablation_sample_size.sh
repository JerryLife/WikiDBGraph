#!/bin/bash
#
# Ablation Study: Sample Size
#
# Compares different sample sizes (number of representative values per column).
# Each configuration gets its own triplets, model, and embeddings in its output directory.
#
# Usage: ./run_ablation_sample_size.sh [--gpu ID] [--skip-training]
#

set -e

BASE_DIR=$(pwd)

# Default parameters
GPU_ID="0"
SKIP_TRAINING=""
LR="1e-05"
EPOCHS=10
BATCH_SIZE=32
SEED=0

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

log() { echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --gpu) GPU_ID="$2"; shift 2 ;;
        --skip-training) SKIP_TRAINING="--skip-training"; shift ;;
        --lr) LR="$2"; shift 2 ;;
        --epochs) EPOCHS="$2"; shift 2 ;;
        --batch-size) BATCH_SIZE="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

export PYTHONPATH="${BASE_DIR}/src:${PYTHONPATH}"

# Sample sizes to test
SAMPLE_SIZES=(1 5 10)

log "=========================================="
log "ABLATION STUDY: Sample Size"
log "=========================================="
log "Sample sizes: ${SAMPLE_SIZES[*]}"
log "GPU: $GPU_ID"
log "Training: ${SKIP_TRAINING:-enabled}"
log "Output format: out/graph_full_ss{size}_neg6/"
log "=========================================="

mkdir -p out

for size in "${SAMPLE_SIZES[@]}"; do
    log "Running sample_size: $size"
    
    # Run full pipeline with training for each sample size
    # Each gets its own triplets, model, and embeddings
    ./src/preprocess/run_preprocess.sh \
        --mode "full" \
        --sample-size "$size" \
        --num-negatives 6 \
        --gpu "$GPU_ID" \
        --lr "$LR" \
        --epochs "$EPOCHS" \
        --batch-size "$BATCH_SIZE" \
        --seed "$SEED" \
        $SKIP_TRAINING \
        2>&1 | tee "out/ablation_sample_size_${size}.log"
    
    log_success "Completed sample_size: $size"
done

log "=========================================="
log "Generating comparison plot..."
log "=========================================="

mkdir -p fig

# Generate comparison plot
python -m preprocess.summary.plot_ablation_sample_size \
    --results-dir "out" \
    --sizes ${SAMPLE_SIZES[*]} \
    --output "fig/ablation_sample_size.png"

log_success "Ablation study complete!"
log "Results in: out/graph_full_ss*_neg6/"
log "Plot in: fig/ablation_sample_size.png"
