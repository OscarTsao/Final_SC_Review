#!/usr/bin/env python3
"""
Step 1: Retrieve candidates using NV-Embed-v2.

This script runs in the nv-embed-v2 conda environment (transformers 4.44.2).
It retrieves candidates for each (post_id, criterion) pair and saves them to disk
for the reranking step.

Usage:
    conda activate nv-embed-v2
    python scripts/step1_retrieve_candidates.py --output data/cache/retrieval_candidates.json
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

from tqdm.auto import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from final_sc_review.data.io import load_criteria, load_groundtruth, load_sentence_corpus
from final_sc_review.retriever.zoo import RetrieverZoo


def build_query_data(groundtruth, criteria_map, include_no_evidence=True):
    """Build query data from groundtruth for all posts."""
    queries = {}

    for row in groundtruth:
        key = (row.post_id, row.criterion_id)
        if key not in queries:
            queries[key] = {
                'post_id': row.post_id,
                'criterion_id': row.criterion_id,
                'criterion_text': criteria_map.get(row.criterion_id, row.criterion_id),
                'gold_uids': set(),
            }

        if row.groundtruth == 1:
            queries[key]['gold_uids'].add(row.sent_uid)

    result = []
    for query_data in queries.values():
        query_data['is_no_evidence'] = len(query_data['gold_uids']) == 0
        query_data['gold_uids'] = list(query_data['gold_uids'])  # Convert to list for JSON
        if include_no_evidence or not query_data['is_no_evidence']:
            result.append(query_data)

    return result


def main():
    parser = argparse.ArgumentParser(description="Step 1: Retrieve candidates using NV-Embed-v2")
    parser.add_argument("--output", type=str, default="data/cache/retrieval_candidates.json",
                        help="Output file for retrieval candidates")
    parser.add_argument("--retriever", type=str, default="nv-embed-v2",
                        help="Retriever name")
    parser.add_argument("--top_k", type=int, default=20,
                        help="Number of candidates to retrieve per query")
    parser.add_argument("--data_dir", type=str, default="data",
                        help="Data directory")
    args = parser.parse_args()

    data_dir = project_root / args.data_dir
    output_path = project_root / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading data...")
    groundtruth = load_groundtruth(data_dir / "groundtruth/evidence_sentence_groundtruth.csv")
    sentences = load_sentence_corpus(data_dir / "groundtruth/sentence_corpus.jsonl")
    criteria = load_criteria(data_dir / "DSM5/MDD_Criteira.json")

    print(f"Groundtruth rows: {len(groundtruth)}")
    print(f"Sentences: {len(sentences)}")
    print(f"Criteria: {len(criteria)}")

    # Build lookup maps
    sentences_by_post = defaultdict(list)
    for sent in sentences:
        sentences_by_post[sent.post_id].append(sent)

    criteria_map = {c.criterion_id: c.text for c in criteria}

    # Build queries
    queries = build_query_data(groundtruth, criteria_map, include_no_evidence=True)
    print(f"Total queries: {len(queries)}")

    # Initialize retriever
    print(f"\nInitializing retriever: {args.retriever}")
    cache_dir = data_dir / "cache"
    retriever_zoo = RetrieverZoo(sentences=sentences, cache_dir=cache_dir)
    retriever = retriever_zoo.get_retriever(args.retriever)

    # Encode corpus
    print("Encoding corpus (using cache if available)...")
    retriever.encode_corpus(rebuild=False)
    print("Retriever ready!")

    # Retrieve candidates for each query
    print(f"\nRetrieving top-{args.top_k} candidates for each query...")
    all_candidates = []

    for query in tqdm(queries, desc="Retrieving"):
        post_id = query['post_id']
        criterion_text = query['criterion_text']

        # Get sentences for this post
        post_sentences = sentences_by_post.get(post_id, [])
        if len(post_sentences) < 2:
            continue

        # Retrieve candidates
        try:
            results = retriever.retrieve_within_post(
                query=criterion_text,
                post_id=post_id,
                top_k=min(args.top_k, len(post_sentences)),
            )
        except Exception as e:
            print(f"Error retrieving for {post_id}/{query['criterion_id']}: {e}")
            continue

        if len(results) < 2:
            continue

        candidates = []
        for r in results:
            candidates.append({
                'sent_uid': r.sent_uid,
                'text': r.text,
                'retriever_score': float(r.score),
            })

        all_candidates.append({
            'post_id': post_id,
            'criterion_id': query['criterion_id'],
            'criterion_text': criterion_text,
            'gold_uids': query['gold_uids'],
            'is_no_evidence': query['is_no_evidence'],
            'candidates': candidates,
        })

    # Save results
    print(f"\nSaving {len(all_candidates)} query results to {output_path}")
    with open(output_path, 'w') as f:
        json.dump({
            'retriever': args.retriever,
            'top_k': args.top_k,
            'n_queries': len(all_candidates),
            'queries': all_candidates,
        }, f, indent=2)

    print("Done!")

    # Summary stats
    n_with_evidence = sum(1 for q in all_candidates if not q['is_no_evidence'])
    print(f"\nSummary:")
    print(f"  Total queries: {len(all_candidates)}")
    print(f"  Queries with evidence: {n_with_evidence}")
    print(f"  Queries without evidence: {len(all_candidates) - n_with_evidence}")


if __name__ == "__main__":
    main()
