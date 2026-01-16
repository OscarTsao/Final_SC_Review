# GNN-Based NE Detection and Dynamic-K Selection: Final Report

## Executive Summary

This report presents the results of applying Graph Neural Networks to the S-C evidence retrieval pipeline for:
1. **No-Evidence (NE) Detection**: Identifying queries with no relevant evidence
2. **Dynamic-K Selection**: Adaptively selecting the number of evidence candidates

**Key Results**: [To be filled after experiments]
- Baseline (RF on features): AUROC=0.60, TPR@5%FPR=10.95%
- Best GNN Model: AUROC=?, TPR@5%FPR=?

## 1. Introduction

### 1.1 Problem Statement

Given a (post, criterion) query with ~20 candidate sentences:
- Predict whether ANY candidate is relevant (has_evidence)
- Select optimal K candidates to return

### 1.2 Baseline Performance

| Metric | RF Baseline | Target |
|--------|-------------|--------|
| AUROC | 0.596 | >0.65 |
| TPR@5%FPR | 10.95% | >15% |
| Evidence Recall | - | >93% |
| Avg K | - | ≤5 |

### 1.3 Constraints

- NO label leakage (forbidden: mrr, recall_at_*, gold_rank)
- Post-id disjoint 5-fold CV
- Dynamic-K: k_min=2, k_max=10, k_max_ratio=0.5

## 2. Methodology

### 2.1 Graph Construction

Each query becomes a graph where:
- **Nodes**: Candidate sentences
- **Node Features**: BGE-M3 embedding (1024d) + reranker score + rank features
- **Edges**: Semantic kNN (k=5, threshold=0.5) + sentence adjacency

### 2.2 Models

| Model | Task | Architecture |
|-------|------|--------------|
| P1 NE Gate | Graph classification | GAT + Attention Pooling + MLP |
| P2 Dynamic-K | Node scoring | GAT + Node MLP |
| P3 Reranker | Score refinement | GAT + Score adjustment head |
| P4 Hetero | Cross-criterion | HeteroGNN with typed edges |

### 2.3 Training

- Loss: Focal loss (gamma=2.0, alpha=0.25)
- Optimizer: AdamW (lr=1e-4, weight_decay=1e-5)
- Early stopping: patience=10, min_delta=1e-4
- 5-fold CV with nested threshold tuning

## 3. Results

### 3.1 Graph Statistics Baseline

| Feature Set | AUROC | TPR@5%FPR |
|-------------|-------|-----------|
| Score features only | ? | ? |
| Graph stats only | ? | ? |
| Score + Graph stats | ? | ? |

**Insight**: [Does graph structure provide signal beyond scores?]

### 3.2 P1 NE Gate GNN

| Variant | AUROC | TPR@5%FPR | AUPRC |
|---------|-------|-----------|-------|
| GCN | ? | ? | ? |
| GraphSAGE | ? | ? | ? |
| GAT (default) | ? | ? | ? |

**Best Configuration**: [Architecture details]

### 3.3 P2 Dynamic-K Selection

| Policy | Evidence Recall | Avg K | F1 |
|--------|-----------------|-------|-----|
| Fixed K=5 | ? | 5.0 | ? |
| Threshold (tau=0.5) | ? | ? | ? |
| Mass (gamma=0.9) | ? | ? | ? |

### 3.4 Ablation Study

| Component | AUROC Δ | TPR@5%FPR Δ |
|-----------|---------|-------------|
| - Embeddings | ? | ? |
| - Graph edges | ? | ? |
| - Score features | ? | ? |
| GCN vs GAT | ? | ? |
| Mean vs Attention pool | ? | ? |

## 4. Analysis

### 4.1 When Does GNN Help?

[Analysis of cases where GNN improves over baseline]

### 4.2 Failure Cases

[Analysis of queries where GNN performs worse]

### 4.3 Graph Structure Insights

[What edge patterns correlate with has_evidence?]

## 5. Conclusions

### 5.1 Key Findings

1. [Finding 1]
2. [Finding 2]
3. [Finding 3]

### 5.2 Recommendations

For deployment:
- [ ] Model selection
- [ ] Threshold configuration
- [ ] Dynamic-K policy

### 5.3 Future Work

1. Pre-training on larger corpus
2. Joint NE + Dynamic-K training
3. Explainable attention visualization

## 6. Leaderboard

| Method | AUROC | TPR@5%FPR | Evidence Recall | Avg K |
|--------|-------|-----------|-----------------|-------|
| Baseline (rf_100) | 0.596 | 10.95% | - | - |
| Graph Stats | ? | ? | - | - |
| P1 NE Gate (GAT) | ? | ? | - | - |
| P2 Dynamic-K | ? | ? | ? | ? |
| P3 Reranker | ? | ? | ? | ? |
| P4 Hetero | ? | ? | ? | ? |

## Appendix

### A. Hyperparameter Settings

```yaml
graph_construction:
  embedding_dim: 1024
  knn_k: 5
  knn_threshold: 0.5
  edge_types: [semantic_knn, adjacency]

model:
  gnn_type: gat
  hidden_dim: 256
  num_layers: 3
  num_heads: 4
  dropout: 0.3
  pooling: attention

training:
  learning_rate: 1e-4
  batch_size: 32
  max_epochs: 100
  patience: 10
  focal_gamma: 2.0
  focal_alpha: 0.25
```

### B. Compute Resources

- GPU: [Specify]
- Training time per fold: [Specify]
- Total experiment time: [Specify]

### C. Code Location

- Graph builder: `src/final_sc_review/gnn/graphs/builder.py`
- Models: `src/final_sc_review/gnn/models/`
- Training: `scripts/gnn/train_eval_ne_gnn.py`
- Ablation: `scripts/gnn/ablation_study.py`

---

*Report generated: [Date]*
*Experiment ID: [Timestamp]*
