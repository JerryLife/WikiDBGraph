#!/bin/bash
# santoseval.sh: Run the SANTOS baseline evaluation

set -e

# Default Params
DATA_DIR="data/unzip" # Overwrite in config or via python args
TRIPLETS="out/graph_full_ss3_neg6/triplets/triplets_test_seed0.jsonl"
OUTPUT_DIR="out/santos_eval"
LIMIT="" # Empty for all

mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "SANTOS Evaluation Pipeline"
echo "=========================================="

# 1. Synthesize KB
# This will use the config.py paths, but we can rely on it reading from data/unzip
echo "[1/3] Synthesizing Knowledge Base..."
# python -m baseline.santos.synthesize_kb # This takes arguments? No, config based.
# To support custom data dir for testing, we might need to modify synthesize_kb to accept args override.
# But for now, let's assume config.py is source of truth or we edit it/mock it.
# Actually, for the dry run verification, I intended to pass args.
# My synthesize_kb.py uses config.UNZIP_DIR directly.
# I should update synthesize_kb.py to accept --data-dir override for flexibility.
# But for now, assuming the user standard run:
python -m src.baseline.santos.synthesize_kb

# 2. Score Pairs
echo "[2/3] Scoring Test Pairs..."
# Limit arg
LIMIT_ARG=""
if [ -n "$LIMIT" ]; then
    LIMIT_ARG="--limit $LIMIT"
fi

python -m src.baseline.santos.score_pairs \
    --triplets "$TRIPLETS" \
    --output "$OUTPUT_DIR/scores.csv" \
    $LIMIT_ARG

# 3. Evaluate
echo "[3/3] Calculating AUC..."
python -m src.baseline.santos.evaluate_auc \
    --scores "$OUTPUT_DIR/scores.csv" \
    --plot "$OUTPUT_DIR/roc_curve.png"

echo "=========================================="
echo "Done. Results in $OUTPUT_DIR"
