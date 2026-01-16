#!/bin/bash
# Run all incomplete HPO: Inference first, then Finetuning
# Created: 2026-01-14

set -e

echo "=============================================="
echo "RUNNING ALL INCOMPLETE HPO COMBINATIONS"
echo "Started: $(date)"
echo "=============================================="

# Run inference HPO first
echo ""
echo ">>> PHASE 1: INFERENCE HPO <<<"
bash scripts/run_incomplete_hpo.sh

# Run finetuning HPO after inference completes
echo ""
echo ">>> PHASE 2: FINETUNING HPO <<<"
bash scripts/run_incomplete_finetuning_hpo.sh

echo ""
echo "=============================================="
echo "ALL HPO COMPLETED"
echo "Finished: $(date)"
echo "=============================================="
