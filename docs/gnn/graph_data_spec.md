# GNN Graph Data Specification

## Overview

This document specifies the graph data format for GNN-based NE detection and Dynamic-K selection. Each query (post_id, criterion_id) becomes a graph where candidates are nodes.

## Feature Provenance

### CRITICAL: Label Leakage Prevention

**FORBIDDEN FEATURES** (derived from gold labels):
- `is_gold`, `groundtruth`
- `mrr`, `recall_at_*`, `map_at_*`, `ndcg_at_*`
- `gold_rank`, `min_gold_rank`, `mean_gold_rank`
- `n_gold_sentences`, `gold_sentence_ids`

These features are **ONLY** available at training time and cause severe data leakage if used as model input. The `tests/test_gnn_no_leakage.py` tests verify compliance.

### Allowed Node Features (Inference-Time)

| Feature | Dim | Source | Description |
|---------|-----|--------|-------------|
| `embedding` | 1024 | BGE-M3 | Dense sentence embedding |
| `reranker_score_norm` | 1 | Jina-v3 | MinMax normalized reranker score |
| `rank_percentile` | 1 | Derived | Position in ranking (0=top, 1=bottom) |
| `gap_to_prev` | 1 | Derived | Score difference to higher-ranked candidate |
| `gap_to_next` | 1 | Derived | Score difference to lower-ranked candidate |
| `score_zscore` | 1 | Derived | Z-score of reranker score |
| `score_minmax` | 1 | Derived | MinMax normalized score (duplicate for compatibility) |
| `above_mean` | 1 | Derived | Binary: score > mean |
| `below_median` | 1 | Derived | Binary: score < median |

**Total node feature dimension**: 1024 + 8 = 1032 (with embeddings) or 8 (without)

### Edge Features

| Feature | Dim | Source | Description |
|---------|-----|--------|-------------|
| `cosine_similarity` | 1 | Embeddings | Cosine similarity between node embeddings |
| `sequence_distance` | 1 | Sentence IDs | Normalized distance in sentence order |

## Graph Construction

### Node Creation
- Each candidate sentence becomes a node
- Nodes are ordered by reranker score (descending)
- Max candidates per query: 20 (configurable)

### Edge Types

1. **Semantic kNN** (`semantic_knn`)
   - Connect each node to k=5 nearest neighbors by cosine similarity
   - Minimum similarity threshold: 0.5
   - Directed edges (i → j if j is neighbor of i)

2. **Adjacency** (`adjacency`)
   - Connect consecutive sentences in original post order
   - Bidirectional edges
   - Only connects sentences with |sid_i - sid_j| ≤ 1

3. **Full** (`full`)
   - All pairs connected (optional, for ablation)
   - Creates dense graph

## Data Format

### PyG Data Object

```python
Data(
    x=[n_nodes, node_feature_dim],      # Node features (NO gold labels!)
    edge_index=[2, n_edges],             # COO format edges
    edge_attr=[n_edges, edge_feature_dim], # Edge features

    # Metadata
    query_id=str,                        # "{post_id}_{criterion_id}"
    candidate_uids=[str],                # List of sentence UIDs
    reranker_scores=[n_nodes],           # Original reranker scores
    n_candidates=int,                    # Number of candidates

    # Labels (training only, NOT features)
    y=[1],                               # Graph-level: has_evidence
    node_labels=[n_nodes],               # Node-level: is_gold (optional)

    # Fold info
    fold_id=int,                         # 0-4 for 5-fold CV
    post_id=str,                         # For post-id disjoint splits
    criterion_id=str,                    # DSM-5 criterion ID
)
```

### File Structure

```
data/cache/gnn/<timestamp>/
├── fold_0.pt           # Graphs for fold 0
├── fold_1.pt           # Graphs for fold 1
├── fold_2.pt           # Graphs for fold 2
├── fold_3.pt           # Graphs for fold 3
├── fold_4.pt           # Graphs for fold 4
├── uid_to_idx.json     # UID → embedding index mapping
└── metadata.json       # Dataset metadata
```

### Metadata Schema

```json
{
    "timestamp": "20240117_120000",
    "n_folds": 5,
    "seed": 42,
    "total_queries": 14770,
    "total_candidates": 295400,
    "has_evidence_rate": 0.45,
    "top_k_candidates": 20,
    "knn_k": 5,
    "knn_threshold": 0.5,
    "embedding_dim": 1024,
    "fold_stats": {
        "0": {"n_queries": 2954, "n_has_evidence": 1329, "n_candidates": 59080},
        ...
    },
    "config": {...},
    "source_cache": "data/cache/oof_cache/...",
    "source_embeddings": "data/cache/bge_m3/dense.npy"
}
```

## Cross-Validation Protocol

### Post-ID Disjoint Splits
- All queries from the same post go to the same fold
- Prevents information leakage across train/val
- ~2,954 queries per fold (14,770 / 5)

### Nested Tuning (Inner Split)
- Within each training fold: 70% train, 30% threshold tuning
- Threshold tuning set used for:
  - NE gate threshold selection
  - Dynamic-K threshold optimization
  - Hyperparameter search

### Evaluation Protocol
1. Train on folds [0,1,2,3], validate on fold 4
2. Train on folds [0,1,2,4], validate on fold 3
3. ... (5-fold rotation)
4. Report mean ± std across all folds

## Graph Statistics (Baseline Features)

For non-GNN baselines, extract these graph-level statistics:

| Feature | Description |
|---------|-------------|
| `edge_density` | n_edges / (n_nodes * (n_nodes-1)) |
| `avg_degree` | Mean out-degree |
| `max_degree` | Maximum out-degree |
| `clustering_proxy` | Edge density (simplified) |
| `score_degree_corr` | Correlation(degree, score) |
| `avg_neighbor_sim` | Mean cosine similarity among top-5 |

## Usage Example

```python
from final_sc_review.gnn.graphs.builder import GraphBuilder
from final_sc_review.gnn.config import GraphConstructionConfig

# Build graphs
config = GraphConstructionConfig(
    embedding_path=Path("data/cache/bge_m3/dense.npy"),
    knn_k=5,
    knn_threshold=0.5,
)
builder = GraphBuilder(config)
builder.load_embeddings()

graph = builder.build_graph(
    candidate_uids=["post1_0", "post1_1", "post1_2"],
    reranker_scores=np.array([0.9, 0.7, 0.5]),
    sentence_ids=[0, 1, 2],
    query_id="post1_A1",
    has_evidence=True,  # Label, NOT a feature
)

# Verify no leakage
from final_sc_review.gnn.graphs.features import assert_no_leakage
feature_names = builder.node_extractor.get_feature_names()
assert_no_leakage({n: 0 for n in feature_names})  # Raises if leakage
```

## Verification

Run leakage tests before any training:

```bash
pytest tests/test_gnn_no_leakage.py -v
```

All tests **MUST** pass before proceeding with GNN training.
