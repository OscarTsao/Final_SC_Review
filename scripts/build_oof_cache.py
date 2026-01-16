#!/usr/bin/env python3
"""
Build Out-of-Fold (OOF) Cache for Fast Post-Processing HPO.

This script generates a cache containing all reranker predictions for each fold's
validation data, enabling fast iteration on post-processing strategies without
re-running the reranker.

Features:
- Loads pre-computed retrieval candidates (5-fold CV)
- Runs reranker inference on validation queries only (out-of-fold)
- Optionally includes NO_EVIDENCE pseudo-candidate scoring
- Stores per-query records with full score information
- Saves manifest for reproducibility

Usage:
    python scripts/build_oof_cache.py \
        --candidates outputs/retrieval_candidates/retrieval_candidates.pkl \
        --model_dir outputs/training/no_evidence_reranker \
        --outdir outputs/oof_cache \
        --include_no_evidence

Output structure:
    outputs/oof_cache/
    ├── oof_predictions.parquet
    └── manifest.json

Note: Uses pickle for loading retrieval candidates (existing pipeline format).
This is safe as we control the data source.
"""

import argparse
import json
import os
# NOTE: pickle is used here intentionally to load pre-computed candidates
# This is safe as we control the data source
import pickle
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from scipy.special import expit
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# NO_EVIDENCE pseudo-candidate token (must match training)
NO_EVIDENCE_TOKEN = "[NO_EVIDENCE]"

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)


class RerankerScorer:
    """Efficient batch scoring for reranker model."""

    def __init__(self, model_dir: str, device: str = "cuda", dtype_name: str = "float16"):
        self.device = device
        self.model_dir = model_dir
        self.dtype_name = dtype_name
        if dtype_name == "float16":
            self.dtype = torch.float16
        elif dtype_name == "bfloat16":
            self.dtype = torch.bfloat16
        else:
            self.dtype = torch.float32
        self.model = None
        self.tokenizer = None

    def load(self):
        """Load model and tokenizer (lazy loading)."""
        if self.model is None:
            print(f"Loading reranker from {self.model_dir}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_dir,
                torch_dtype=self.dtype,
            )
            self.model = self.model.to(self.device)
            self.model.requires_grad_(False)
            print(f"  Model loaded on {self.device} with dtype {self.dtype_name}")

    def score(
        self,
        query: str,
        texts: List[str],
        batch_size: int = 32,
        max_length: int = 384,
    ) -> List[float]:
        """Score query-text pairs and return logits."""
        self.load()
        scores = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            pairs = [[query, t] for t in batch_texts]
            inputs = self.tokenizer(
                pairs,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits.squeeze(-1).cpu().numpy()
                if logits.ndim == 0:
                    logits = [float(logits)]
                else:
                    logits = logits.tolist()
                scores.extend(logits)
        return scores

    def score_batch_queries(
        self,
        queries: List[str],
        texts_per_query: List[List[str]],
        batch_size: int = 64,
        max_length: int = 384,
    ) -> List[List[float]]:
        """Batch score multiple queries efficiently.

        Flattens all pairs, scores in batches, then reconstitutes per-query results.
        """
        self.load()

        # Build flat list of all pairs with query indices
        all_pairs = []
        query_boundaries = [0]
        for q_idx, (query, texts) in enumerate(zip(queries, texts_per_query)):
            for text in texts:
                all_pairs.append([query, text])
            query_boundaries.append(len(all_pairs))

        if not all_pairs:
            return [[] for _ in queries]

        # Score all pairs in batches
        all_scores = []
        for i in tqdm(range(0, len(all_pairs), batch_size), desc="Scoring batches", leave=False):
            batch_pairs = all_pairs[i : i + batch_size]
            inputs = self.tokenizer(
                batch_pairs,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits.squeeze(-1).cpu().numpy()
                if logits.ndim == 0:
                    logits = [float(logits)]
                else:
                    logits = logits.tolist()
                all_scores.extend(logits)

        # Reconstitute per-query scores
        results = []
        for i in range(len(queries)):
            start = query_boundaries[i]
            end = query_boundaries[i + 1]
            results.append(all_scores[start:end])

        return results


def extract_score_features(scores: List[float]) -> Dict[str, float]:
    """Extract features from score distribution for NE classifier input."""
    if not scores:
        return {
            "max_score": 0.0,
            "min_score": 0.0,
            "mean_score": 0.0,
            "std_score": 0.0,
            "top3_mean": 0.0,
            "score_range": 0.0,
            "top1_minus_top2": 0.0,
            "top1_minus_mean": 0.0,
            "n_candidates": 0,
        }

    scores_arr = np.array(sorted(scores, reverse=True))
    n = len(scores_arr)

    max_score = float(scores_arr[0])
    min_score = float(scores_arr[-1])
    mean_score = float(scores_arr.mean())
    std_score = float(scores_arr.std()) if n > 1 else 0.0

    top3_mean = float(scores_arr[: min(3, n)].mean())
    score_range = max_score - min_score
    top1_minus_top2 = float(scores_arr[0] - scores_arr[1]) if n > 1 else 0.0
    top1_minus_mean = max_score - mean_score

    return {
        "max_score": max_score,
        "min_score": min_score,
        "mean_score": mean_score,
        "std_score": std_score,
        "top3_mean": top3_mean,
        "score_range": score_range,
        "top1_minus_top2": top1_minus_top2,
        "top1_minus_mean": top1_minus_mean,
        "n_candidates": n,
    }


def build_oof_cache(
    candidates_path: str,
    model_dir: str,
    output_dir: Path,
    include_no_evidence: bool = True,
    batch_size: int = 64,
    device: str = "cuda",
    dtype_name: str = "float16",
) -> pd.DataFrame:
    """Build OOF cache for all folds.

    Args:
        candidates_path: Path to retrieval candidates pickle
        model_dir: Path to trained reranker model
        output_dir: Output directory for cache
        include_no_evidence: Include NO_EVIDENCE pseudo-candidate scoring
        batch_size: Batch size for reranker inference
        device: Device for inference
        dtype_name: Data type for model

    Returns:
        DataFrame with all OOF predictions
    """
    print("Loading retrieval candidates...")
    with open(candidates_path, "rb") as f:
        data = pickle.load(f)

    scorer = RerankerScorer(model_dir, device, dtype_name)

    all_records = []
    fold_names = sorted([k for k in data.keys() if k.startswith("fold_")])

    print(f"\nProcessing {len(fold_names)} folds...")

    for fold_name in fold_names:
        fold_data = data[fold_name]
        fold_id = int(fold_name.split("_")[1])
        val_data = fold_data.get("val_data", [])
        train_posts = set(fold_data.get("train_posts", []))
        val_posts = set(fold_data.get("val_posts", []))

        # Sanity check: no overlap between train and val posts
        overlap = train_posts & val_posts
        if overlap:
            raise ValueError(
                f"CRITICAL: {fold_name} has {len(overlap)} overlapping posts! "
                "This would cause data leakage."
            )

        print(f"\n[{fold_name}] Processing {len(val_data)} validation queries...")
        print(f"  Train posts: {len(train_posts)}, Val posts: {len(val_posts)}")

        # Collect queries and candidates for batch processing
        queries = []
        texts_per_query = []
        query_meta = []

        for query_entry in val_data:
            query_text = query_entry.get("query", "")
            post_id = query_entry.get("post_id", "")
            criterion_id = query_entry.get("criterion_id", "")
            is_no_evidence = query_entry.get("is_no_evidence", True)
            gold_uids = query_entry.get("gold_uids", [])
            candidates = query_entry.get("candidates", [])

            # Sanity check: val query should be from val post
            if post_id not in val_posts:
                raise ValueError(
                    f"Query post_id {post_id} not in val_posts for {fold_name}!"
                )

            # Extract candidate info
            candidate_uids = [c.get("sent_uid", "") for c in candidates]
            candidate_texts = [c.get("text", "") for c in candidates]
            candidate_labels = [c.get("label", 0) for c in candidates]
            retriever_scores = [c.get("score", 0.0) for c in candidates]

            # Add NO_EVIDENCE pseudo-candidate if requested
            if include_no_evidence:
                candidate_texts_with_ne = candidate_texts + [NO_EVIDENCE_TOKEN]
            else:
                candidate_texts_with_ne = candidate_texts

            queries.append(query_text)
            texts_per_query.append(candidate_texts_with_ne)
            query_meta.append({
                "fold_id": fold_id,
                "post_id": post_id,
                "criterion_id": criterion_id,
                "gt_has_evidence": not is_no_evidence,
                "gold_uids": gold_uids,
                "candidate_uids": candidate_uids,
                "candidate_labels": candidate_labels,
                "retriever_scores": retriever_scores,
                "n_candidates": len(candidates),
            })

        # Batch score all queries
        print(f"  Scoring {len(queries)} queries with reranker...")
        all_scores = scorer.score_batch_queries(
            queries, texts_per_query, batch_size=batch_size
        )

        # Process results and build records
        for q_idx, (meta, scores) in enumerate(zip(query_meta, all_scores)):
            n_real_candidates = meta["n_candidates"]

            # Split scores into real candidates and NO_EVIDENCE
            if include_no_evidence and scores:
                reranker_scores = scores[:n_real_candidates]
                ne_score = scores[n_real_candidates] if len(scores) > n_real_candidates else None
            else:
                reranker_scores = scores
                ne_score = None

            # Extract score features
            score_features = extract_score_features(reranker_scores)

            # Compute calibrated probabilities (sigmoid on raw logits as baseline)
            calibrated_probs = expit(np.array(reranker_scores)).tolist() if reranker_scores else []

            # Build record
            record = {
                "fold_id": meta["fold_id"],
                "post_id": meta["post_id"],
                "criterion_id": meta["criterion_id"],
                "gt_has_evidence": meta["gt_has_evidence"],
                "n_candidates": n_real_candidates,
                # Store as JSON strings for parquet compatibility
                "gold_uids": json.dumps(meta["gold_uids"]),
                "candidate_uids": json.dumps(meta["candidate_uids"]),
                "candidate_labels": json.dumps(meta["candidate_labels"]),
                "retriever_scores": json.dumps(meta["retriever_scores"]),
                "reranker_scores": json.dumps(reranker_scores),
                "calibrated_probs": json.dumps(calibrated_probs),
                # Score features for NE detection
                "max_score": score_features["max_score"],
                "min_score": score_features["min_score"],
                "mean_score": score_features["mean_score"],
                "std_score": score_features["std_score"],
                "top3_mean": score_features["top3_mean"],
                "score_range": score_features["score_range"],
                "top1_minus_top2": score_features["top1_minus_top2"],
                "top1_minus_mean": score_features["top1_minus_mean"],
            }

            # Add NO_EVIDENCE score if available
            if include_no_evidence:
                record["ne_score"] = ne_score
                record["ne_calibrated_prob"] = expit(ne_score) if ne_score is not None else None
                # Compute margin: best_real_score - ne_score
                if ne_score is not None and reranker_scores:
                    record["best_minus_ne"] = max(reranker_scores) - ne_score
                else:
                    record["best_minus_ne"] = None

            all_records.append(record)

        print(f"  Collected {len(all_records)} total records so far")

    # Create DataFrame
    df = pd.DataFrame(all_records)

    # Verify fold distribution
    print("\n" + "=" * 60)
    print("OOF Cache Summary")
    print("=" * 60)
    print(f"Total queries: {len(df)}")
    print(f"\nPer-fold distribution:")
    for fold_id in sorted(df["fold_id"].unique()):
        fold_df = df[df["fold_id"] == fold_id]
        n_has_evidence = fold_df["gt_has_evidence"].sum()
        n_no_evidence = len(fold_df) - n_has_evidence
        print(f"  Fold {fold_id}: {len(fold_df)} queries ({n_has_evidence} has-ev, {n_no_evidence} no-ev)")

    print(f"\nClass balance:")
    n_total_has_ev = df["gt_has_evidence"].sum()
    n_total_no_ev = len(df) - n_total_has_ev
    print(f"  Has evidence: {n_total_has_ev} ({n_total_has_ev/len(df)*100:.1f}%)")
    print(f"  No evidence: {n_total_no_ev} ({n_total_no_ev/len(df)*100:.1f}%)")

    if include_no_evidence:
        print(f"\nNO_EVIDENCE pseudo-candidate:")
        print(f"  Mean NE score: {df['ne_score'].mean():.4f}")
        print(f"  Mean best_minus_ne margin: {df['best_minus_ne'].mean():.4f}")

    return df


def generate_manifest(
    output_dir: Path,
    args: argparse.Namespace,
    df: pd.DataFrame,
) -> Dict[str, Any]:
    """Generate manifest.json with reproducibility information."""
    manifest = {
        "created_at": datetime.now().isoformat(),
        "script": "scripts/build_oof_cache.py",
        "args": vars(args),
        "seed": SEED,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "platform": platform.platform(),
        "git_branch": "unknown",
        "git_commit": "unknown",
        "stats": {
            "total_queries": len(df),
            "n_folds": len(df["fold_id"].unique()),
            "n_has_evidence": int(df["gt_has_evidence"].sum()),
            "n_no_evidence": int((~df["gt_has_evidence"]).sum()),
            "has_evidence_ratio": float(df["gt_has_evidence"].mean()),
        },
    }

    try:
        manifest["git_branch"] = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=PROJECT_ROOT,
        ).decode().strip()
        manifest["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
        ).decode().strip()
    except Exception:
        pass

    return manifest


def main():
    parser = argparse.ArgumentParser(
        description="Build OOF cache for post-processing HPO",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--candidates",
        type=str,
        default="outputs/retrieval_candidates/retrieval_candidates.pkl",
        help="Path to retrieval candidates pickle",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="outputs/training/no_evidence_reranker",
        help="Path to trained reranker model",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="outputs/oof_cache",
        help="Output directory for OOF cache",
    )
    parser.add_argument(
        "--include_no_evidence",
        action="store_true",
        default=True,
        help="Include NO_EVIDENCE pseudo-candidate scoring",
    )
    parser.add_argument(
        "--no_no_evidence",
        action="store_true",
        help="Disable NO_EVIDENCE pseudo-candidate scoring",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for reranker inference",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for inference",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "float32", "bfloat16"],
        help="Data type for model",
    )
    args = parser.parse_args()

    # Handle --no_no_evidence flag
    if args.no_no_evidence:
        args.include_no_evidence = False

    output_dir = Path(args.outdir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("BUILD OUT-OF-FOLD (OOF) CACHE")
    print("=" * 70)
    print(f"Candidates: {args.candidates}")
    print(f"Model: {args.model_dir}")
    print(f"Output: {output_dir}")
    print(f"Include NO_EVIDENCE: {args.include_no_evidence}")
    print(f"Device: {args.device}")
    print(f"Dtype: {args.dtype}")
    print("=" * 70)

    # Build cache
    df = build_oof_cache(
        candidates_path=args.candidates,
        model_dir=args.model_dir,
        output_dir=output_dir,
        include_no_evidence=args.include_no_evidence,
        batch_size=args.batch_size,
        device=args.device,
        dtype_name=args.dtype,
    )

    # Save parquet
    parquet_path = output_dir / "oof_predictions.parquet"
    df.to_parquet(parquet_path, index=False)
    print(f"\nSaved OOF cache: {parquet_path}")
    print(f"  Size: {parquet_path.stat().st_size / 1024 / 1024:.2f} MB")

    # Generate and save manifest
    manifest = generate_manifest(output_dir, args, df)
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved manifest: {manifest_path}")

    print("\n" + "=" * 70)
    print("OOF CACHE BUILD COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
