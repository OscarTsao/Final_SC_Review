#!/bin/bash
# Full Reranker Research Pipeline
# Implements Section 12 of reranker_research_plan.md:
# Run all reranker HPO → Test all retriever+reranker combinations → Find best config

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Configuration
OUTPUT_BASE="outputs/reranker_research"
N_TRIALS=50
EPOCHS=3
MAX_LENGTH=256

# Rerankers to run HPO for (cross-encoder models that support training)
RERANKERS=(
    "BAAI/bge-reranker-v2-m3"
    "jinaai/jina-reranker-v2-base-multilingual"
    "cross-encoder/ms-marco-MiniLM-L-12-v2"
)

RERANKER_NAMES=(
    "bge-reranker-v2-m3"
    "jina-reranker-v2"
    "ms-marco-minilm"
)

# Retrievers to test combinations with
RETRIEVERS=(
    "bge-m3"
    "bge-large-en-v1.5"
    "e5-large-v2"
    "bm25"
)

# Top-K values
TOP_K_RETRIEVER="50 100 200"
TOP_K_RERANK="10 20 50"

echo "================================================================================"
echo "FULL RERANKER RESEARCH PIPELINE"
echo "Timestamp: $(date -Iseconds)"
echo "================================================================================"
echo ""
echo "Configuration:"
echo "  Output directory: $OUTPUT_BASE"
echo "  N trials: $N_TRIALS"
echo "  Epochs: $EPOCHS"
echo "  Rerankers: ${RERANKER_NAMES[@]}"
echo "  Retrievers: ${RETRIEVERS[@]}"
echo ""

mkdir -p "$OUTPUT_BASE"

# ==============================================================================
# Phase 1: Run HPO for all rerankers
# ==============================================================================
echo ""
echo "================================================================================"
echo "PHASE 1: RERANKER HPO WITH LOSS FUNCTION SWEEP"
echo "================================================================================"

for i in "${!RERANKERS[@]}"; do
    model="${RERANKERS[$i]}"
    name="${RERANKER_NAMES[$i]}"

    output_dir="$OUTPUT_BASE/hpo_${name}"
    db_path="$output_dir/optuna.db"

    echo ""
    echo "--------------------------------------------------------------------------------"
    echo "Training HPO: $name ($model)"
    echo "Output: $output_dir"
    echo "--------------------------------------------------------------------------------"

    mkdir -p "$output_dir"

    # Check if HPO already completed
    if [ -f "$output_dir/best_config.json" ]; then
        echo "HPO already completed for $name, skipping..."
        continue
    fi

    # Run HPO with all loss function sweeps
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python scripts/reranker/train_maxout.py \
        --model "$model" \
        --output_dir "$output_dir" \
        --hpo \
        --n_trials $N_TRIALS \
        --epochs $EPOCHS \
        --max_length $MAX_LENGTH \
        --use_gradient_checkpointing \
        --num_workers 18 \
        --study_name "${name}_loss_hpo" \
        --storage "sqlite:///$db_path" \
        2>&1 | tee "$output_dir/hpo.log"

    # Export best config
    echo "Exporting best config for $name..."
    python -c "
import optuna
import json
from pathlib import Path

storage = 'sqlite:///$db_path'
study = optuna.load_study(study_name='${name}_loss_hpo', storage=storage)

best_trial = study.best_trial
config = {
    'model': '$model',
    'name': '$name',
    'best_value': best_trial.value,
    'best_params': best_trial.params,
    'n_trials': len(study.trials),
}

with open('$output_dir/best_config.json', 'w') as f:
    json.dump(config, f, indent=2)

print(f'Best trial: {best_trial.number}')
print(f'Best value: {best_trial.value:.6f}')
print(f'Best params: {best_trial.params}')
"

    echo "HPO completed for $name"
done

# ==============================================================================
# Phase 2: Evaluate all retriever+reranker combinations
# ==============================================================================
echo ""
echo "================================================================================"
echo "PHASE 2: RETRIEVER + RERANKER COMBINATION SWEEP"
echo "================================================================================"

SWEEP_OUTPUT="$OUTPUT_BASE/combination_sweep"
mkdir -p "$SWEEP_OUTPUT"

python scripts/eval_retriever_reranker_combinations.py \
    --output_dir "$SWEEP_OUTPUT" \
    --retrievers ${RETRIEVERS[@]} \
    --rerankers ${RERANKER_NAMES[@]} \
    --top_k_retriever $TOP_K_RETRIEVER \
    --top_k_rerank $TOP_K_RERANK \
    --split val \
    2>&1 | tee "$SWEEP_OUTPUT/sweep.log"

# ==============================================================================
# Phase 3: Generate final report
# ==============================================================================
echo ""
echo "================================================================================"
echo "PHASE 3: GENERATING FINAL REPORT"
echo "================================================================================"

python -c "
import json
import pandas as pd
from pathlib import Path

output_base = Path('$OUTPUT_BASE')

# Load HPO results
hpo_results = []
for name in ${RERANKER_NAMES[@]@Q}.split():
    config_path = output_base / f'hpo_{name}' / 'best_config.json'
    if config_path.exists():
        with open(config_path) as f:
            hpo_results.append(json.load(f))

# Load combination sweep results
sweep_path = output_base / 'combination_sweep' / 'sweep_summary.json'
if sweep_path.exists():
    with open(sweep_path) as f:
        sweep_results = json.load(f)
else:
    sweep_results = None

# Generate report
report = {
    'timestamp': '$(date -Iseconds)',
    'hpo_results': hpo_results,
    'combination_sweep': sweep_results,
}

report_path = output_base / 'final_report.json'
with open(report_path, 'w') as f:
    json.dump(report, f, indent=2)

print('Final report saved to', report_path)

# Print summary
print()
print('=' * 80)
print('FINAL RESEARCH SUMMARY')
print('=' * 80)

print()
print('HPO Results (Best val_loss per reranker):')
print('-' * 60)
for r in hpo_results:
    print(f\"  {r['name']}: {r['best_value']:.6f}\")
    params = r['best_params']
    print(f\"    loss_type: {params.get('listwise_type', 'N/A')}\")
    print(f\"    curriculum: {params.get('curriculum_enabled', False)}\")

if sweep_results:
    print()
    print('Best Overall Retriever+Reranker Combination:')
    print('-' * 60)
    best = sweep_results['best_overall']
    print(f\"  Retriever: {best['retriever']}\")
    print(f\"  Reranker: {best['reranker']}\")
    print(f\"  nDCG@10: {best['ndcg_at_10']:.4f}\")
    print(f\"  MRR@10: {best['mrr_at_10']:.4f}\")
    print(f\"  Recall@10: {best['recall_at_10']:.4f}\")
    print(f\"  False Evidence Rate: {best['false_evidence_rate']:.4f}\")
"

echo ""
echo "================================================================================"
echo "RERANKER RESEARCH PIPELINE COMPLETE"
echo "Results saved to: $OUTPUT_BASE"
echo "================================================================================"
