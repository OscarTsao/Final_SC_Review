#!/usr/bin/env python3
"""
Build BGE-M3 embedding caches for the sentence corpus.

Creates cached embeddings in data/cache/bge_m3/:
- dense.npy: Dense embeddings
- sparse.pkl: Sparse BM25-style weights
- colbert.pkl: ColBERT token vectors
- fingerprint.json: Cache validation hash

Usage:
    python scripts/retriever/build_cache.py
    python scripts/retriever/build_cache.py --config configs/default.yaml
    python scripts/retriever/build_cache.py --rebuild
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import yaml


def get_repo_root() -> Path:
    """Find repository root."""
    current = Path(__file__).resolve().parent
    while current != current.parent:
        if (current / ".git").exists():
            return current
        current = current.parent
    return Path(__file__).resolve().parent.parent.parent


def main():
    parser = argparse.ArgumentParser(description="Build BGE-M3 embedding caches")
    parser.add_argument("--config", type=str, default=None, help="Config YAML path")
    parser.add_argument("--rebuild", action="store_true", help="Force cache rebuild")
    args = parser.parse_args()

    repo_root = get_repo_root()

    # Load config
    if args.config:
        config_path = Path(args.config)
    else:
        config_path = repo_root / "configs" / "default.yaml"

    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)
    else:
        config = {}

    # Extract paths from config or use defaults
    paths = config.get("paths", {})
    corpus_path = Path(paths.get(
        "sentence_corpus",
        repo_root / "data" / "groundtruth" / "sentence_corpus.jsonl"
    ))
    cache_dir = Path(paths.get(
        "cache_dir",
        repo_root / "data" / "cache" / "bge_m3"
    ))

    # Import retriever
    sys.path.insert(0, str(repo_root / "src"))
    from final_sc_review.retriever.bge_m3 import BgeM3Retriever
    from final_sc_review.data.io import load_sentence_corpus

    print("="*60)
    print("BUILD BGE-M3 EMBEDDING CACHE")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Corpus: {corpus_path}")
    print(f"Cache dir: {cache_dir}")
    print(f"Rebuild: {args.rebuild}")
    print("="*60)

    # Load corpus
    if not corpus_path.exists():
        print(f"[ERROR] Corpus not found: {corpus_path}")
        sys.exit(1)

    print("\n[1/3] Loading sentence corpus...")
    sentences = load_sentence_corpus(corpus_path)
    print(f"  Loaded {len(sentences)} sentences")

    # Check if cache already exists
    cache_dir = Path(cache_dir)
    fingerprint_path = cache_dir / "fingerprint.json"

    if fingerprint_path.exists() and not args.rebuild:
        with open(fingerprint_path) as f:
            fp = json.load(f)
        print(f"\n[INFO] Cache already exists")
        print(f"  Created: {fp.get('created_at', 'unknown')}")
        print(f"  Corpus hash: {fp.get('corpus_hash', 'unknown')[:16]}...")
        print(f"  N sentences: {fp.get('n_sentences', 'unknown')}")
        print("\n[SKIP] Use --rebuild to force rebuild")
        return

    # Get model config
    models_cfg = config.get("models", {})
    model_name = models_cfg.get("bge_model", "BAAI/bge-m3")
    batch_size = models_cfg.get("batch_size_embed", 64)
    max_length = models_cfg.get("max_passage_length", 512)

    print(f"\n[2/3] Initializing BGE-M3 retriever...")
    print(f"  Model: {model_name}")
    print(f"  Batch size: {batch_size}")
    print(f"  Max length: {max_length}")

    # Initialize retriever (this will build/load cache)
    retriever = BgeM3Retriever(
        sentences=sentences,
        cache_dir=cache_dir,
        model_name=model_name,
        batch_size=batch_size,
        passage_max_length=max_length,
        rebuild_cache=args.rebuild,
    )

    print(f"\n[3/3] Cache build complete!")
    print(f"  Dense shape: {retriever.dense_vecs.shape}")
    print(f"  Sparse vectors: {len(retriever.sparse_weights)}")
    print(f"  ColBERT vectors: {len(retriever.colbert_vecs)}")

    # Verify cache files
    cache_files = ["dense.npy", "sparse.pkl", "colbert.pkl", "fingerprint.json"]
    print("\n[VERIFY] Cache files:")
    for fname in cache_files:
        fpath = cache_dir / fname
        if fpath.exists():
            size_mb = fpath.stat().st_size / (1024 * 1024)
            print(f"  {fname}: {size_mb:.1f} MB")
        else:
            print(f"  {fname}: MISSING")

    print("\n" + "="*60)
    print("[SUCCESS] Cache build complete")
    print("="*60)


if __name__ == "__main__":
    main()
