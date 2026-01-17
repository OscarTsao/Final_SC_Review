#!/usr/bin/env python3
"""LLM Integration Pilot Experiment.

Test LLM reranker and verifier on a small sample (DEV split or first 50 queries)
to validate implementation and estimate costs before running full 5-fold CV.

Usage:
    export GEMINI_API_KEY="your-api-key"
    python scripts/llm_integration/run_llm_pilot.py --n_samples 50

Estimated cost: ~$0.50 for 50 queries (Gemini Flash)
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from final_sc_review.llm.gemini_client import GeminiClient
from final_sc_review.llm.reranker import LLMReranker
from final_sc_review.llm.verifier import LLMVerifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_text_data():
    """Load sentence corpus, posts, and criteria for text lookup.

    Returns:
        (sentence_dict, post_dict, criterion_dict)
    """
    # Load sentence corpus
    corpus_file = Path("data/groundtruth/sentence_corpus.jsonl")
    sentence_dict = {}
    with open(corpus_file) as f:
        for line in f:
            sent = json.loads(line)
            sentence_dict[sent["sent_uid"]] = sent["text"]

    # Load posts (reconstruct from sentences for now - not ideal but works for pilot)
    post_dict = {}
    # We'll populate this on-the-fly from sentences

    # Load criteria
    criteria_file = Path("data/DSM5/MDD_Criteira.json")
    with open(criteria_file) as f:
        criteria_data = json.load(f)

    criterion_dict = {}
    for criterion in criteria_data["criteria"]:
        criterion_dict[criterion["id"]] = criterion["text"]

    logger.info(f"Loaded {len(sentence_dict)} sentences, {len(post_dict)} posts, {len(criterion_dict)} criteria")

    return sentence_dict, post_dict, criterion_dict


def load_sample_graphs(graph_cache_dir: Path, n_samples: int = 50):
    """Load a sample of graphs for pilot testing.

    Args:
        graph_cache_dir: Path to graph cache directory
        n_samples: Number of samples to load

    Returns:
        List of PyG Data objects
    """
    # Find latest graph cache
    cache_dirs = sorted([d for d in graph_cache_dir.iterdir() if d.is_dir()])
    if not cache_dirs:
        raise ValueError(f"No graph cache found in {graph_cache_dir}")

    latest_cache = cache_dirs[-1]
    logger.info(f"Loading graphs from {latest_cache}")

    # Load fold 0 (arbitrary choice for pilot)
    fold_file = latest_cache / "fold_0.pt"
    if not fold_file.exists():
        raise ValueError(f"Fold file not found: {fold_file}")

    data = torch.load(fold_file, weights_only=False)

    # Extract graphs list from dict
    if isinstance(data, dict) and 'graphs' in data:
        graphs = data['graphs']
    else:
        graphs = data

    logger.info(f"Loaded {len(graphs)} graphs from fold 0")

    # Take first n_samples
    sample_graphs = graphs[:n_samples]
    logger.info(f"Using {len(sample_graphs)} graphs for pilot")

    return sample_graphs


def extract_graph_info(graph: Data, sentence_dict: dict, post_dict: dict, criterion_dict: dict):
    """Extract information from PyG graph for LLM processing.

    Args:
        graph: PyG Data object
        sentence_dict: Mapping from sent_uid to sentence text
        post_dict: Mapping from post_id to post text
        criterion_dict: Mapping from criterion_id to criterion text

    Returns:
        Dictionary with query info and candidates
    """
    # Extract UIDs and scores from graph
    candidate_uids = graph.candidate_uids
    reranker_scores = graph.reranker_scores.tolist() if hasattr(graph.reranker_scores, 'tolist') else graph.reranker_scores
    post_id = graph.post_id
    criterion_id = graph.criterion_id
    has_evidence = int(graph.y[0].item())

    # Get text from dictionaries
    candidate_texts = [sentence_dict.get(uid, f"[Missing: {uid}]") for uid in candidate_uids]

    # Reconstruct post text from candidate sentences (simple approach for pilot)
    # In full implementation, load actual post text from source
    if post_id in post_dict:
        query_text = post_dict[post_id]
    else:
        query_text = " ".join(candidate_texts)  # Use candidates as proxy

    criterion_text = criterion_dict.get(criterion_id, f"[Missing criterion: {criterion_id}]")

    # Ground truth labels
    node_labels = graph.node_labels.tolist() if hasattr(graph.node_labels, 'tolist') else graph.node_labels
    gold_ids = [uid for uid, label in zip(candidate_uids, node_labels) if label == 1]

    return {
        "query_text": query_text,
        "criterion_text": criterion_text,
        "candidates": candidate_texts,
        "candidate_ids": candidate_uids,
        "reranker_scores": reranker_scores,
        "has_evidence": has_evidence,
        "gold_ids": set(gold_ids) if gold_ids else set(),
    }


def run_llm_reranker_pilot(
    graphs: list,
    client: GeminiClient,
    sentence_dict: dict,
    post_dict: dict,
    criterion_dict: dict,
    top_m: int = 10,
):
    """Run LLM reranker on sample graphs.

    Args:
        graphs: List of PyG graphs
        client: Gemini client
        sentence_dict: Sentence text lookup
        post_dict: Post text lookup
        criterion_dict: Criterion text lookup
        top_m: Number of top candidates to rerank

    Returns:
        List of results with reranked candidates
    """
    logger.info("="*60)
    logger.info("PILOT: LLM Reranker")
    logger.info("="*60)

    reranker = LLMReranker(client, top_m=top_m)

    results = []
    for graph in tqdm(graphs, desc="Reranking"):
        info = extract_graph_info(graph, sentence_dict, post_dict, criterion_dict)

        try:
            reranked_indices, llm_scores = reranker.rerank(
                query_text=info["query_text"],
                criterion_text=info["criterion_text"],
                candidates=info["candidates"],
                candidate_ids=info["candidate_ids"],
                current_scores=info["reranker_scores"],
            )

            results.append({
                "query_text": info["query_text"][:100],
                "criterion_text": info["criterion_text"][:100],
                "n_candidates": len(info["candidates"]),
                "reranked_top5": [info["candidates"][i][:80] for i in reranked_indices[:5]],
                "llm_scores_top5": [llm_scores[i] for i in reranked_indices[:5]],
                "success": True,
            })

        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            results.append({
                "query_text": info["query_text"][:100],
                "error": str(e),
                "success": False,
            })

    success_rate = sum(r["success"] for r in results) / len(results)
    logger.info(f"Reranker success rate: {success_rate:.1%}")

    return results


def run_llm_verifier_pilot(
    graphs: list,
    client: GeminiClient,
    sentence_dict: dict,
    post_dict: dict,
    criterion_dict: dict,
    verification_mode: str = "all",
):
    """Run LLM verifier on sample graphs.

    Args:
        graphs: List of PyG graphs
        client: Gemini client
        sentence_dict: Sentence text lookup
        post_dict: Post text lookup
        criterion_dict: Criterion text lookup
        verification_mode: Verification mode

    Returns:
        List of results with verification outputs
    """
    logger.info("="*60)
    logger.info("PILOT: LLM Verifier")
    logger.info("="*60)

    verifier = LLMVerifier(client, verification_mode=verification_mode)

    results = []
    for graph in tqdm(graphs, desc="Verifying"):
        info = extract_graph_info(graph, sentence_dict, post_dict, criterion_dict)

        # Take top-5 for verification pilot
        top_k = min(5, len(info["candidates"]))
        sorted_indices = np.argsort(info["reranker_scores"])[::-1]
        top_indices = sorted_indices[:top_k]

        top_candidates = [info["candidates"][i] for i in top_indices]
        top_ids = [info["candidate_ids"][i] for i in top_indices]

        try:
            supports, confidence = verifier.verify_batch(
                query_text=info["query_text"],
                criterion_text=info["criterion_text"],
                candidates=top_candidates,
                candidate_ids=top_ids,
                ne_prob=1.0,  # Assume positive for pilot
            )

            n_supported = sum(supports)

            results.append({
                "query_text": info["query_text"][:100],
                "criterion_text": info["criterion_text"][:100],
                "n_candidates": len(top_candidates),
                "n_supported": n_supported,
                "supports": supports,
                "confidence": confidence,
                "verified_candidates": [
                    (top_candidates[i][:80], supports[i], confidence[i])
                    for i in range(len(top_candidates))
                ],
                "success": True,
            })

        except Exception as e:
            logger.error(f"Verification failed: {e}")
            results.append({
                "query_text": info["query_text"][:100],
                "error": str(e),
                "success": False,
            })

    success_rate = sum(r["success"] for r in results) / len(results)
    logger.info(f"Verifier success rate: {success_rate:.1%}")

    avg_supported = np.mean([r["n_supported"] for r in results if r["success"]])
    logger.info(f"Average candidates supported: {avg_supported:.2f}")

    return results


def main():
    parser = argparse.ArgumentParser(description="LLM Integration Pilot")
    parser.add_argument("--n_samples", type=int, default=50, help="Number of samples to test")
    parser.add_argument("--top_m", type=int, default=10, help="Top-M for reranking")
    parser.add_argument("--test_reranker", action="store_true", help="Test reranker")
    parser.add_argument("--test_verifier", action="store_true", help="Test verifier")
    parser.add_argument("--output_dir", type=str, default="outputs/llm_pilot")
    args = parser.parse_args()

    # Default: test both if neither specified
    if not args.test_reranker and not args.test_verifier:
        args.test_reranker = True
        args.test_verifier = True

    # Setup output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("="*60)
    logger.info("LLM Integration Pilot Experiment")
    logger.info("="*60)
    logger.info(f"Samples: {args.n_samples}")
    logger.info(f"Output: {output_dir}")

    # Check API key
    if not os.getenv("GEMINI_API_KEY"):
        logger.error("GEMINI_API_KEY not set. Please export it before running.")
        logger.info("Get your API key from: https://makersuite.google.com/app/apikey")
        return

    # Initialize Gemini client
    try:
        client = GeminiClient()
        logger.info("✓ Gemini client initialized")
    except Exception as e:
        logger.error(f"Failed to initialize Gemini client: {e}")
        return

    # Load text data
    try:
        sentence_dict, post_dict, criterion_dict = load_text_data()
        logger.info("✓ Loaded text data")
    except Exception as e:
        logger.error(f"Failed to load text data: {e}")
        import traceback
        traceback.print_exc()
        return

    # Load sample graphs
    try:
        graph_cache_dir = Path("data/cache/gnn")
        graphs = load_sample_graphs(graph_cache_dir, n_samples=args.n_samples)
        logger.info(f"✓ Loaded {len(graphs)} sample graphs")
    except Exception as e:
        logger.error(f"Failed to load graphs: {e}")
        logger.info("Note: This pilot requires pre-built graph cache from GNN experiments")
        return

    results = {}

    # Test reranker
    if args.test_reranker:
        try:
            reranker_results = run_llm_reranker_pilot(
                graphs, client, sentence_dict, post_dict, criterion_dict, top_m=args.top_m
            )
            results["reranker"] = reranker_results

            # Save reranker results
            with open(output_dir / "reranker_results.json", "w") as f:
                json.dump(reranker_results, f, indent=2)

            logger.info(f"✓ Reranker results saved to {output_dir}/reranker_results.json")

        except Exception as e:
            logger.error(f"Reranker pilot failed: {e}")
            import traceback
            traceback.print_exc()

    # Test verifier
    if args.test_verifier:
        try:
            verifier_results = run_llm_verifier_pilot(
                graphs, client, sentence_dict, post_dict, criterion_dict
            )
            results["verifier"] = verifier_results

            # Save verifier results
            with open(output_dir / "verifier_results.json", "w") as f:
                json.dump(verifier_results, f, indent=2)

            logger.info(f"✓ Verifier results saved to {output_dir}/verifier_results.json")

        except Exception as e:
            logger.error(f"Verifier pilot failed: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    logger.info("="*60)
    logger.info("Pilot Complete!")
    logger.info("="*60)
    logger.info(f"Output directory: {output_dir}")

    if args.test_reranker:
        n_success = sum(r["success"] for r in results.get("reranker", []))
        logger.info(f"Reranker: {n_success}/{args.n_samples} successful")

    if args.test_verifier:
        n_success = sum(r["success"] for r in results.get("verifier", []))
        logger.info(f"Verifier: {n_success}/{args.n_samples} successful")

    logger.info("\nNext steps:")
    logger.info("1. Review results in output directory")
    logger.info("2. If successful, run full 5-fold CV with:")
    logger.info("   python scripts/llm_integration/run_llm_full_eval.py")


if __name__ == "__main__":
    main()
