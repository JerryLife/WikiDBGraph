#!/bin/bash
#
# Ablation Study: Number of Negatives
#
# Compares different numbers of negative samples per triplet.
# Each configuration gets its own triplets, model, and embeddings in its output directory.
#
# Usage: ./run_ablation_num_negatives.sh [--gpu ID] [--skip-training]
#

set -e

BASE_DIR=$(pwd)

# Default parameters
GPU_ID="0"
SKIP_TRAINING=""
LR="1e-05"
EPOCHS=10
BATCH_SIZE=32

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
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

export PYTHONPATH="${BASE_DIR}/src:${PYTHONPATH}"

# Number of negatives to test
NUM_NEGATIVES=(2 4 6 10 15)

log "=========================================="
log "ABLATION STUDY: Number of Negatives"
log "=========================================="
log "Num negatives: ${NUM_NEGATIVES[*]}"
log "GPU: $GPU_ID"
log "Training: ${SKIP_TRAINING:-enabled}"
log "Output format: out/graph_full_ss3_neg{num}/"
log "=========================================="

mkdir -p out

for num in "${NUM_NEGATIVES[@]}"; do
    log "Running num_negatives: $num"
    
    # Run full pipeline with training for each num_negatives value
    # Each gets its own triplets, model, and embeddings
    ./src/preprocess/run_preprocess.sh \
        --mode "full" \
        --sample-size 3 \
        --num-negatives "$num" \
        --gpu "$GPU_ID" \
        --lr "$LR" \
        --epochs "$EPOCHS" \
        --batch-size "$BATCH_SIZE" \
        $SKIP_TRAINING \
        2>&1 | tee "out/ablation_num_negatives_${num}.log"
    
    log_success "Completed num_negatives: $num"
done

log "=========================================="
log "Generating comparison plot..."
log "=========================================="

mkdir -p fig

# Generate comparison plot
python -m preprocess.summary.plot_ablation_num_negatives \
    --results-dir "out" \
    --nums ${NUM_NEGATIVES[*]} \
    --output "fig/ablation_num_negatives.png"

log_success "Ablation study complete!"
log "Results in: out/graph_full_ss3_neg*/"
log "Plot in: fig/ablation_num_negatives.png"
