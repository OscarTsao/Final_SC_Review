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

    graphs = torch.load(fold_file)
    logger.info(f"Loaded {len(graphs)} graphs from fold 0")

    # Take first n_samples
    sample_graphs = graphs[:n_samples]
    logger.info(f"Using {len(sample_graphs)} graphs for pilot")

    return sample_graphs


def extract_graph_info(graph: Data):
    """Extract information from PyG graph for LLM processing.

    Args:
        graph: PyG Data object

    Returns:
        Dictionary with query info and candidates
    """
    # Extract node features
    n_nodes = graph.x.shape[0]

    # Assuming graph has these attributes (from GNN pipeline)
    # You may need to adjust based on actual graph structure
    candidate_texts = graph.candidate_texts if hasattr(graph, 'candidate_texts') else [f"Sentence {i}" for i in range(n_nodes)]
    candidate_ids = graph.candidate_ids if hasattr(graph, 'candidate_ids') else [f"uid_{i}" for i in range(n_nodes)]
    reranker_scores = graph.reranker_scores if hasattr(graph, 'reranker_scores') else np.random.rand(n_nodes).tolist()

    # Query info
    query_text = graph.query_text if hasattr(graph, 'query_text') else "Sample query text"
    criterion_text = graph.criterion_text if hasattr(graph, 'criterion_text') else "Sample criterion"

    # Ground truth
    has_evidence = graph.has_evidence if hasattr(graph, 'has_evidence') else 1
    gold_ids = graph.gold_ids if hasattr(graph, 'gold_ids') else []

    return {
        "query_text": query_text,
        "criterion_text": criterion_text,
        "candidates": candidate_texts,
        "candidate_ids": candidate_ids,
        "reranker_scores": reranker_scores,
        "has_evidence": has_evidence,
        "gold_ids": set(gold_ids) if gold_ids else set(),
    }


def run_llm_reranker_pilot(
    graphs: list,
    client: GeminiClient,
    top_m: int = 10,
):
    """Run LLM reranker on sample graphs.

    Args:
        graphs: List of PyG graphs
        client: Gemini client
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
        info = extract_graph_info(graph)

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
    verification_mode: str = "all",
):
    """Run LLM verifier on sample graphs.

    Args:
        graphs: List of PyG graphs
        client: Gemini client
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
        info = extract_graph_info(graph)

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
            reranker_results = run_llm_reranker_pilot(graphs, client, top_m=args.top_m)
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
            verifier_results = run_llm_verifier_pilot(graphs, client)
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
