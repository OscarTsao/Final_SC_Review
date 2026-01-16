#!/bin/bash
# Master script for post-processing optimization pipeline
#
# This script runs the complete optimization pipeline:
# 1. Build OOF cache (if not exists)
# 2. Run NE gate HPO
# 3. Run Dynamic-K HPO
# 4. Generate deployment config and assessment
#
# Usage:
#   bash scripts/run_postproc_optimization.sh [profile]
#
# Arguments:
#   profile: Deployment profile to use (default: high_recall_low_hallucination)
#            Options: high_recall_low_hallucination, ultra_safe, cheap

set -e

PROFILE="${1:-high_recall_low_hallucination}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "========================================"
echo "POST-PROCESSING OPTIMIZATION PIPELINE"
echo "========================================"
echo "Profile: ${PROFILE}"
echo "Timestamp: ${TIMESTAMP}"
echo "========================================"

# Paths
CANDIDATES="outputs/retrieval_candidates/retrieval_candidates.pkl"
MODEL_DIR="outputs/training/no_evidence_reranker"
OOF_CACHE="outputs/oof_cache/oof_predictions.parquet"
TARGETS_CONFIG="configs/deployment_targets.yaml"

# Output directories
OOF_DIR="outputs/oof_cache"
NE_HPO_DIR="outputs/hpo_ne_gate"
DK_HPO_DIR="outputs/hpo_dynamic_k"
DEPLOY_DIR="outputs/deployment_optimized/${PROFILE}_${TIMESTAMP}"

# Check prerequisites
if [ ! -f "$CANDIDATES" ]; then
    echo "ERROR: Retrieval candidates not found: $CANDIDATES"
    echo "Run the retrieval pipeline first."
    exit 1
fi

if [ ! -d "$MODEL_DIR" ]; then
    echo "ERROR: Reranker model not found: $MODEL_DIR"
    echo "Train the reranker first."
    exit 1
fi

if [ ! -f "$TARGETS_CONFIG" ]; then
    echo "ERROR: Deployment targets config not found: $TARGETS_CONFIG"
    exit 1
fi

# Step 1: Build OOF cache (if needed)
echo ""
echo "========================================"
echo "STEP 1: Build OOF Cache"
echo "========================================"

if [ -f "$OOF_CACHE" ]; then
    echo "OOF cache already exists: $OOF_CACHE"
    echo "Skipping cache build. Delete the file to rebuild."
else
    echo "Building OOF cache..."
    python scripts/build_oof_cache.py \
        --candidates "$CANDIDATES" \
        --model_dir "$MODEL_DIR" \
        --outdir "$OOF_DIR" \
        --include_no_evidence \
        --batch_size 64
fi

# Verify cache exists
if [ ! -f "$OOF_CACHE" ]; then
    echo "ERROR: Failed to build OOF cache"
    exit 1
fi

# Step 2: NE Gate HPO
echo ""
echo "========================================"
echo "STEP 2: NE Gate HPO"
echo "========================================"

python scripts/hpo_ne_gate.py \
    --oof_cache "$OOF_CACHE" \
    --targets "$TARGETS_CONFIG" \
    --profile "$PROFILE" \
    --outdir "$NE_HPO_DIR" \
    --n_steps 200

# Step 3: Dynamic-K HPO
echo ""
echo "========================================"
echo "STEP 3: Dynamic-K HPO"
echo "========================================"

python scripts/hpo_dynamic_k.py \
    --oof_cache "$OOF_CACHE" \
    --targets "$TARGETS_CONFIG" \
    --profile "$PROFILE" \
    --outdir "$DK_HPO_DIR" \
    --n_steps 100

# Step 4: Generate Deployment Config
echo ""
echo "========================================"
echo "STEP 4: Generate Deployment Config"
echo "========================================"

python scripts/generate_deployment_config.py \
    --ne_hpo "${NE_HPO_DIR}/results.json" \
    --dk_hpo "${DK_HPO_DIR}/results.json" \
    --oof_cache "$OOF_CACHE" \
    --targets "$TARGETS_CONFIG" \
    --profile "$PROFILE" \
    --outdir "$DEPLOY_DIR"

# Summary
echo ""
echo "========================================"
echo "OPTIMIZATION COMPLETE"
echo "========================================"
echo ""
echo "Outputs:"
echo "  OOF Cache: $OOF_CACHE"
echo "  NE Gate HPO: ${NE_HPO_DIR}/results.json"
echo "  Dynamic-K HPO: ${DK_HPO_DIR}/results.json"
echo "  Deployment Config: ${DEPLOY_DIR}/deployment_config.yaml"
echo "  Comparison Report: ${DEPLOY_DIR}/comparison_report.md"
echo ""
echo "Next steps:"
echo "  1. Review the comparison report"
echo "  2. If targets are not met, consider Phase 3 (advanced improvements)"
echo "  3. Run final assessment with the deployment config"
echo ""
