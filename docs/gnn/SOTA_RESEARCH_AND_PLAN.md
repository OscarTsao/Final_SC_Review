# SOTA Research: GNN for Document Evidence Retrieval

## Overview

This document surveys state-of-the-art GNN methods for evidence retrieval and no-evidence detection, informing the design choices in our implementation.

## Problem Setting

Given a query (DSM-5 criterion) and a set of candidate sentences from a post:
1. **NE Detection**: Predict P(has_evidence) - whether any candidate is relevant
2. **Dynamic-K Selection**: Determine how many candidates to return
3. **Reranking**: Refine initial reranker scores using graph context

## Related Work

### 1. GNN for Document Understanding

**Graph-Based Evidence Aggregation (GBEA)**
- Nodes: sentences/passages, Edges: semantic similarity
- Aggregates evidence across related passages
- Key insight: Evidence often spans multiple related sentences

**HiGNN (Hierarchical GNN)**
- Multi-level graphs: word → sentence → paragraph
- Bottom-up and top-down message passing
- Useful for capturing local and global context

### 2. GNN for Retrieval Reranking

**SetRank (NeurIPS 2020)**
- Treats candidate set as a graph
- Learns inter-document relationships
- Graph attention for score aggregation

**LightGCN (SIGIR 2020)**
- Simplified GCN without feature transformation
- Effective for collaborative filtering patterns
- Can capture candidate similarity patterns

**GraphRetriever (ACL 2021)**
- Iterative graph construction and retrieval
- Entity-centric evidence graphs
- Multi-hop reasoning capability

### 3. Graph Classification for Set Prediction

**Deep Sets (NeurIPS 2017)**
- Permutation-invariant set functions
- ρ(Σ φ(x_i)) architecture
- Our mean/max pooling follows this

**Set Transformer (ICML 2019)**
- Attention-based set pooling
- Induced Set Attention Block (ISAB)
- Our attention pooling is related

**GIN (Graph Isomorphism Network)**
- Maximally expressive GNN
- WL test equivalence
- Could be added as alternative

### 4. Threshold/Dynamic-K Selection

**Learning to Stop (ICLR 2019)**
- Neural stopping policies
- Reinforcement learning for K selection
- Related to our Dynamic-K P2 model

**Calibration-based Selection**
- Predict confidence, threshold for K
- Our mass-based policy (cumsum(p) >= gamma)

## Design Choices

### Architecture Selection

| Component | Our Choice | Alternatives | Rationale |
|-----------|------------|--------------|-----------|
| GNN Layers | GAT | GCN, SAGE | Attention weights improve interpretability |
| Pooling | Attention | Mean, Max, Set2Set | Learnable importance weighting |
| Depth | 3 layers | 2-4 | Balance expressivity and oversmoothing |
| Edge Types | kNN + Adjacency | Full, kNN only | Capture semantic + structural |

### Key Design Decisions

1. **Graph per Query**: Each (post_id, criterion) becomes a separate graph
   - Pros: Clean isolation, batch parallelism
   - Cons: Cannot model cross-query patterns within post

2. **Node Features**: Embeddings + scores + rank features
   - NO gold labels (prevents leakage)
   - Rich inference-time signals

3. **Edge Construction**:
   - Semantic kNN: captures meaning similarity
   - Adjacency: captures discourse structure
   - Threshold: prevents noise from weak similarities

4. **Focal Loss**: Addresses class imbalance (~45% positive)
   - gamma=2.0 focuses on hard examples
   - alpha=0.25 upweights rare class

### P4 Heterogeneous Design

Cross-criterion reasoning via typed message passing:
- Criterion nodes carry query context
- Sentence nodes aggregate from multiple criteria
- Edge types preserve semantic roles

## Baseline Comparison

| Method | Type | Expected AUROC | Notes |
|--------|------|----------------|-------|
| Random | - | 0.50 | Baseline |
| Max Score | Threshold | 0.55-0.60 | Strong baseline |
| RF on Features | ML | 0.60 | Current baseline |
| Graph Stats + RF | ML | 0.60-0.65 | If graph signal exists |
| P1 NE Gate GNN | GNN | 0.62-0.68 | Target improvement |
| P4 Hetero GNN | GNN | 0.65-0.70 | With cross-criterion |

## Implementation Notes

### Preventing Oversmoothing

- Residual connections in all layers
- Layer normalization after each GNN layer
- Dropout (0.3) for regularization
- Limited depth (3 layers default)

### Computational Efficiency

- Batch processing via PyG's DataLoader
- Sparse edge storage
- Attention only within local neighborhoods
- Caching of embeddings

### Hyperparameter Ranges

```yaml
learning_rate: [1e-5, 1e-3]
hidden_dim: [128, 256, 512]
num_layers: [2, 3, 4]
dropout: [0.1, 0.3, 0.5]
num_heads: [2, 4, 8]
knn_k: [3, 5, 10]
knn_threshold: [0.3, 0.5, 0.7]
```

## Evaluation Protocol

### Metrics

1. **NE Detection**:
   - AUROC (primary)
   - TPR@5%FPR (deployment constraint)
   - AUPRC (imbalanced data)

2. **Dynamic-K**:
   - Evidence Recall
   - Average K
   - F1 at selected K

3. **Reranking**:
   - MRR, NDCG@K
   - Recall@K with refined scores

### Cross-Validation

- 5-fold by post_id (disjoint)
- Inner 70/30 split for threshold tuning
- Report mean ± std across folds

## Future Directions

1. **Graph Structure Learning**: Learn edge weights jointly
2. **Multi-Task Learning**: Joint NE + Dynamic-K training
3. **Pre-training**: Self-supervised on unlabeled posts
4. **Explainability**: Attention visualization for evidence chains

## References

1. Kipf & Welling (2017). Semi-Supervised Classification with GCNs
2. Veličković et al. (2018). Graph Attention Networks
3. Hamilton et al. (2017). Inductive Representation Learning on Large Graphs
4. Lee et al. (2019). Set Transformer
5. Zaheer et al. (2017). Deep Sets
6. He et al. (2020). LightGCN
