#!/bin/bash
# Run all incomplete HPO combinations
# Created: 2026-01-14

set -e

OUTPUT_DIR="outputs/hpo_inference_combos"
mkdir -p "$OUTPUT_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$OUTPUT_DIR/hpo_all_combos_$TIMESTAMP.log"

echo "Starting HPO for all incomplete combinations at $(date)" | tee "$LOG_FILE"
echo "Log file: $LOG_FILE"
echo ""

# Function to run HPO for a combination
run_hpo() {
    local retriever=$1
    local reranker=$2
    local env=${3:-""}  # Optional conda environment

    echo "========================================"  | tee -a "$LOG_FILE"
    echo "Running: $retriever + $reranker" | tee -a "$LOG_FILE"
    echo "Started: $(date)" | tee -a "$LOG_FILE"
    echo "========================================"  | tee -a "$LOG_FILE"

    if [ -n "$env" ]; then
        echo "Using conda env: $env" | tee -a "$LOG_FILE"
        mamba run -n "$env" python scripts/hpo_retriever_reranker_combinations.py \
            --output_dir "$OUTPUT_DIR" \
            --retrievers "$retriever" \
            --rerankers "$reranker" \
            --n_trials 50 2>&1 | tee -a "$LOG_FILE"
    else
        python scripts/hpo_retriever_reranker_combinations.py \
            --output_dir "$OUTPUT_DIR" \
            --retrievers "$retriever" \
            --rerankers "$reranker" \
            --n_trials 50 2>&1 | tee -a "$LOG_FILE"
    fi

    echo "Completed: $(date)" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"
}

# ============================================================
# MISSING INFERENCE HPO COMBINATIONS
# ============================================================

# 1. qwen3-embed-4b combinations (current env should work)
echo "=== QWEN3-EMBED-4B COMBINATIONS ===" | tee -a "$LOG_FILE"
run_hpo "qwen3-embed-4b" "jina-reranker-v3"
run_hpo "qwen3-embed-4b" "bge-reranker-v2-m3"

# 2. llama-embed-8b + jina-reranker-v3
echo "=== LLAMA-EMBED-8B COMBINATIONS ===" | tee -a "$LOG_FILE"
run_hpo "llama-embed-8b" "jina-reranker-v3"

# 3. nv-embed-v2 combinations (requires nv-embed-v2 conda env)
echo "=== NV-EMBED-V2 COMBINATIONS (using nv-embed-v2 env) ===" | tee -a "$LOG_FILE"
run_hpo "nv-embed-v2" "jina-reranker-v3" "nv-embed-v2"
run_hpo "nv-embed-v2" "bge-reranker-v2-m3" "nv-embed-v2"
run_hpo "nv-embed-v2" "jina-reranker-v2" "nv-embed-v2"

echo "========================================"  | tee -a "$LOG_FILE"
echo "ALL INFERENCE HPO COMPLETED at $(date)" | tee -a "$LOG_FILE"
echo "========================================"  | tee -a "$LOG_FILE"
