#!/bin/bash
# Run incomplete finetuning HPO combinations (with NO_EVIDENCE)
# Created: 2026-01-14

set -e

OUTPUT_DIR="outputs/hpo_finetuning_with_no_evidence"
mkdir -p "$OUTPUT_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$OUTPUT_DIR/hpo_finetuning_$TIMESTAMP.log"

echo "Starting Finetuning HPO at $(date)" | tee "$LOG_FILE"
echo "Log file: $LOG_FILE"
echo ""

# Function to run finetuning HPO for a combination
run_finetuning_hpo() {
    local retriever=$1
    local reranker=$2
    local n_trials=${3:-50}
    local env=${4:-""}  # Optional conda environment for retriever cache

    echo "========================================"  | tee -a "$LOG_FILE"
    echo "Finetuning HPO: $retriever + $reranker" | tee -a "$LOG_FILE"
    echo "Trials: $n_trials" | tee -a "$LOG_FILE"
    echo "Started: $(date)" | tee -a "$LOG_FILE"
    echo "========================================"  | tee -a "$LOG_FILE"

    # Step 1: Build retriever cache (may need special env)
    if [ -n "$env" ]; then
        echo "Building cache using conda env: $env" | tee -a "$LOG_FILE"
        mamba run -n "$env" python scripts/hpo_finetuning_combo.py \
            --retriever "$retriever" \
            --reranker "$reranker" \
            --stage cache \
            --with_no_evidence 2>&1 | tee -a "$LOG_FILE"
    fi

    # Step 2: Run training HPO (main env)
    python scripts/hpo_finetuning_combo.py \
        --retriever "$retriever" \
        --reranker "$reranker" \
        --n_trials "$n_trials" \
        --stage train \
        --with_no_evidence 2>&1 | tee -a "$LOG_FILE"

    echo "Completed: $(date)" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"
}

# ============================================================
# INCOMPLETE FINETUNING HPO COMBINATIONS (with NO_EVIDENCE)
# ============================================================

# Check and complete nv-embed-v2 + jina-reranker-v2 (missing from results)
echo "=== NV-EMBED-V2 + JINA-RERANKER-V2 ===" | tee -a "$LOG_FILE"
run_finetuning_hpo "nv-embed-v2" "jina-reranker-v2" 50 "nv-embed-v2"

# Additional finetuning for other priority combinations
echo "=== Additional Finetuning Combinations ===" | tee -a "$LOG_FILE"

# qwen3-embed-0.6b combinations (good inference results)
run_finetuning_hpo "qwen3-embed-0.6b" "jina-reranker-v2" 50
run_finetuning_hpo "qwen3-embed-0.6b" "bge-reranker-v2-m3" 50

echo "========================================"  | tee -a "$LOG_FILE"
echo "ALL FINETUNING HPO COMPLETED at $(date)" | tee -a "$LOG_FILE"
echo "========================================"  | tee -a "$LOG_FILE"
