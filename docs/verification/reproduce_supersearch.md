# Supersearch Reproduction and Leakage Verification

## Summary

**CONFIRMED: Label leakage exists in the supersearch NE detection pipeline.**

The previous supersearch results showing TPR ~99.93% at FPR ~3.05% were artificially inflated due to
label leakage. Features like `mrr`, `recall_at_*`, and `gold_rank` were computed using ground truth
labels and then used as model inputs.

## Leakage Evidence

### Audit Results (2026-01-16)

```
Feature Store: outputs/feature_store/20260115_190449/full_features.parquet
Total features: 33
LEAKY features: 8
DEPLOYABLE features: 15
```

### Key Leaky Features

| Feature | AUC vs Label | Source |
|---------|-------------|--------|
| **mrr** | **0.9554** | Computed from gold_sentence_ids |
| recall_at_10 | 0.8985 | Computed from gold_sentence_ids |
| recall_at_5 | 0.8038 | Computed from gold_sentence_ids |
| recall_at_3 | 0.7295 | Computed from gold_sentence_ids |
| recall_at_1 | 0.5993 | Computed from gold_sentence_ids |
| min_gold_rank | 0.3114 | Rank of first gold in candidate list |
| max_gold_rank | 0.3276 | Rank of last gold in candidate list |
| mean_gold_rank | 0.3208 | Mean rank of gold sentences |

**The `mrr` feature alone has AUC=0.9554, which explains the "too good" results.**

### Leakage Mechanism

In `scripts/supersearch/build_feature_store.py`, the function `compute_rank_features()`:

```python
def compute_rank_features(
    candidate_ids: List[str],
    gold_ids: List[str],        # <-- LEAKAGE: uses ground truth
    candidate_scores: np.ndarray,
) -> Dict[str, float]:
    ...
    # Find ranks of gold sentences
    for rank, idx in enumerate(sorted_indices):
        if candidate_ids[idx] in gold_set:  # <-- LEAKAGE
            gold_ranks.append(rank + 1)

    features["mrr"] = float(1.0 / min(gold_ranks))  # <-- LEAKY FEATURE
    features[f"recall_at_{k}"] = ...                # <-- LEAKY FEATURE
```

These features are then included in the model input (line 268-272):
```python
rank_features = compute_rank_features(
    candidate_ids,
    gold_sent_ids,              # <-- LEAKAGE: passing gold labels
    candidate_scores,
)
```

## Reproduction Commands

### 1. Run Original Supersearch (with leakage)

```bash
# Build feature store (includes leaky features)
python scripts/supersearch/build_feature_store.py \
    --groundtruth data/groundtruth/evidence_sentence_groundtruth.csv \
    --criteria data/DSM5/MDD_Criteira.json \
    --cache_dir data/cache/oof_cache \
    --output_dir outputs/feature_store

# Run supersearch (uses leaky features)
python scripts/supersearch/run_supersearch.py \
    --feature_store outputs/feature_store/20260115_190449 \
    --output_dir outputs/supersearch
```

### 2. Run Leakage Audit

```bash
python scripts/verification/audit_feature_leakage.py \
    --feature_store outputs/feature_store/20260115_190449/full_features.parquet \
    --output_dir outputs/verification_fix/20260116_213647 \
    --auc_threshold 0.90
```

## Expected Clean Performance

With **only deployable features** (max AUC ~0.55), we expect:
- AUROC: ~0.55-0.65 (significantly lower than previous ~0.99)
- TPR@5%FPR: ~60-70% (not 99.93%)

This is because the deployable features (score statistics, entropy, etc.) have weak
individual discriminative power (AUC ~0.5).

## Fix Applied

See `scripts/verification/build_deployable_features.py` for the corrected feature pipeline
that separates:
- `mode="deployable"`: Only inference-time features
- `mode="evaluation"`: Includes metrics for reporting (never for modeling)

## Artifacts

- `outputs/verification_fix/20260116_213647/leakage_audit_report.md`
- `outputs/verification_fix/20260116_213647/leaky_features.csv`
- `outputs/verification_fix/20260116_213647/deployable_features.csv`
