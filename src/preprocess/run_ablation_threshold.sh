#!/bin/bash
#
# Threshold Ablation Study
#
# Analyzes how similarity threshold affects graph density metrics:
# - Number of edges
# - Average degree  
# - Node coverage (% of nodes with at least one edge)
#
# This script reads from an existing all_exhaustive_predictions.pt file
# and computes statistics for multiple thresholds efficiently (no regeneration).
#
# Usage: ./run_ablation_threshold.sh [options]
#

set -e

# Default paths (ss3 and neg6 as default)
BASE_DIR=$(pwd)
PREDICTIONS_PATH="${BASE_DIR}/out/graph_full_ss3_neg6/all_exhaustive_predictions.pt"
OUTPUT_DIR="${BASE_DIR}/fig/ablation_threshold"

# Default thresholds to analyze (scores >= threshold become edges)
THRESHOLDS="0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.93 0.96 0.98"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log() { echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

show_help() {
    cat << EOF
Threshold Ablation Study Script

Analyzes how similarity threshold affects graph density.
Reads predictions ONCE and computes stats for multiple thresholds efficiently.

Usage: $0 [OPTIONS]

OPTIONS:
    --predictions PATH      Path to predictions file (default: out/graph_full_ss3_neg6/all_exhaustive_predictions.pt)
    --output-dir DIR        Output directory for plots (default: fig/ablation_threshold)
    --thresholds T1 T2 ...  Space-separated thresholds to analyze (default: 0.5 0.55 ... 0.95)
    --help                  Show this help message

EXAMPLES:
    # Run with defaults (ss3, neg6)
    $0

    # Custom predictions file
    $0 --predictions out/graph_full_ss5_neg10/all_exhaustive_predictions.pt

    # Custom thresholds
    $0 --thresholds 0.6 0.7 0.8 0.9
EOF
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --predictions) PREDICTIONS_PATH="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --thresholds) 
            shift
            THRESHOLDS=""
            while [[ $# -gt 0 ]] && [[ ! "$1" =~ ^-- ]]; do
                THRESHOLDS="$THRESHOLDS $1"
                shift
            done
            ;;
        --help) show_help; exit 0 ;;
        *) log_error "Unknown option: $1"; show_help; exit 1 ;;
    esac
done

# Validate predictions file exists
if [[ ! -f "$PREDICTIONS_PATH" ]]; then
    log_error "Predictions file not found: $PREDICTIONS_PATH"
    log_error "Please generate predictions first using similarity_computer.py:"
    log_error "  python -m preprocess.similarity_computer \\"
    log_error "    --embeddings out/graph_full_ss3_neg6/database_embeddings.pt \\"
    log_error "    --output $PREDICTIONS_PATH \\"
    log_error "    --threshold 0.5 --chunk-size 1024 --gpu 0"
    exit 1
fi

# Set Python path
export PYTHONPATH="${BASE_DIR}/src:${PYTHONPATH}"

log "=========================================="
log "THRESHOLD ABLATION STUDY"
log "=========================================="
log "Predictions: $PREDICTIONS_PATH"
log "Output directory: $OUTPUT_DIR"
log "Thresholds: $THRESHOLDS"
log "=========================================="

# Run threshold analysis (reads predictions ONCE, filters by each threshold)
log "Analyzing threshold effects on graph density..."
python -m preprocess.summary.plot_ablation_threshold \
    --predictions "$PREDICTIONS_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --thresholds $THRESHOLDS

if [[ $? -ne 0 ]]; then
    log_error "Threshold analysis failed!"
    exit 1
fi

# Summary
log "=========================================="
log_success "ABLATION COMPLETE"
log "=========================================="
log "Outputs:"
log "  Plots: $OUTPUT_DIR/"
log "    - threshold_vs_edges.png"
log "    - threshold_vs_degree.png"
log "    - threshold_vs_coverage.png"
log "  Results: $OUTPUT_DIR/threshold_ablation_results.txt"
log "=========================================="
