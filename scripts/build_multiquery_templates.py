#!/usr/bin/env python3
"""Build multi-query templates (paraphrases) for each DSM-5 criterion.

Generates rule-based and synonym-based paraphrases for multi-query retrieval.
Does NOT use external APIs unless --allow_external_api is set.

Usage:
    python scripts/build_multiquery_templates.py --output outputs/maxout/multiquery/criteria_paraphrases.json
    python scripts/build_multiquery_templates.py --n_paraphrases 12 --allow_external_api
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from final_sc_review.data.io import load_criteria


# Synonym mappings for rule-based paraphrasing
SYNONYM_MAP = {
    # Emotions/mood
    "depressed": ["sad", "down", "low", "unhappy", "melancholic"],
    "sad": ["unhappy", "sorrowful", "dejected", "down"],
    "hopeless": ["despairing", "without hope", "despondent"],
    "empty": ["hollow", "void", "numb"],
    "tearful": ["crying", "weepy", "in tears"],

    # Interest/pleasure
    "interest": ["enthusiasm", "motivation", "engagement"],
    "pleasure": ["enjoyment", "satisfaction", "joy"],
    "diminished": ["reduced", "decreased", "lessened", "lowered"],

    # Physical symptoms
    "weight loss": ["losing weight", "weight reduction", "getting thinner"],
    "weight gain": ["gaining weight", "weight increase", "getting heavier"],
    "appetite": ["hunger", "desire to eat", "eating habits"],
    "insomnia": ["difficulty sleeping", "can't sleep", "sleeplessness", "trouble sleeping"],
    "hypersomnia": ["sleeping too much", "excessive sleep", "oversleeping"],
    "fatigue": ["tiredness", "exhaustion", "lack of energy", "feeling drained"],

    # Psychomotor
    "agitation": ["restlessness", "fidgeting", "nervousness"],
    "retardation": ["slowing down", "sluggishness", "moving slowly"],

    # Cognitive
    "worthlessness": ["feeling worthless", "low self-worth", "feeling useless"],
    "guilt": ["feeling guilty", "self-blame", "remorse"],
    "concentrate": ["focus", "pay attention", "think clearly"],
    "indecisiveness": ["difficulty making decisions", "can't decide", "uncertain"],

    # Suicidal
    "death": ["dying", "mortality", "end of life"],
    "suicidal ideation": ["thoughts of suicide", "thinking about suicide", "suicidal thoughts"],
    "suicide attempt": ["trying to end life", "self-harm attempt", "attempted suicide"],
}

# Template patterns for rule-based paraphrasing
TEMPLATE_PATTERNS = [
    # Simplification
    (r"most of the day, nearly every day", "frequently"),
    (r"most of the day, nearly every day", "on most days"),
    (r"nearly every day", "almost daily"),
    (r"nearly every day", "on a regular basis"),

    # Perspective shifts
    (r"as indicated by either subjective report.*?or observation.*?\)", ""),
    (r"\(as indicated by.*?\)", ""),
    (r"\(observable by others.*?\)", ""),
    (r"\(either by subjective.*?\)", ""),
    (r"\(which may be delusional\)", ""),
    (r"\(not merely.*?\)", ""),
    (r"\(not just.*?\)", ""),
    (r"\(e\.g\.,.*?\)", ""),
]


def clean_text(text: str) -> str:
    """Clean and normalize text."""
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)
    # Remove leading/trailing whitespace
    text = text.strip()
    # Ensure proper sentence ending
    if text and text[-1] not in '.!?':
        text += '.'
    return text


def apply_synonym_replacement(text: str, n_variants: int = 3) -> List[str]:
    """Generate variants by replacing words with synonyms."""
    variants = []
    text_lower = text.lower()

    for word, synonyms in SYNONYM_MAP.items():
        if word in text_lower:
            for syn in synonyms[:n_variants]:
                # Case-insensitive replacement
                pattern = re.compile(re.escape(word), re.IGNORECASE)
                variant = pattern.sub(syn, text)
                variant = clean_text(variant)
                if variant != text and variant not in variants:
                    variants.append(variant)

    return variants


def apply_template_patterns(text: str) -> List[str]:
    """Generate variants by applying template patterns."""
    variants = []

    for pattern, replacement in TEMPLATE_PATTERNS:
        if re.search(pattern, text):
            variant = re.sub(pattern, replacement, text)
            variant = clean_text(variant)
            if variant != text and variant not in variants:
                variants.append(variant)

    return variants


def generate_question_forms(text: str) -> List[str]:
    """Generate question-form paraphrases."""
    variants = []

    # Simple conversion to question
    text_clean = text.rstrip('.')

    # "Does the person experience..."
    variants.append(f"Does the person experience {text_clean.lower()}?")

    # "Is there evidence of..."
    variants.append(f"Is there evidence of {text_clean.lower()}?")

    # "Are there signs of..."
    key_symptoms = extract_key_symptoms(text)
    if key_symptoms:
        variants.append(f"Are there signs of {key_symptoms}?")

    return variants


def extract_key_symptoms(text: str) -> str:
    """Extract key symptom phrases from criterion text."""
    # Common symptom patterns
    patterns = [
        r"(depressed mood)",
        r"(diminished interest or pleasure)",
        r"(weight loss|weight gain|appetite)",
        r"(insomnia|hypersomnia)",
        r"(psychomotor agitation|retardation)",
        r"(fatigue|loss of energy)",
        r"(worthlessness|guilt)",
        r"(diminished ability to think|concentrate|indecisiveness)",
        r"(thoughts of death|suicidal ideation|suicide)",
    ]

    for pattern in patterns:
        match = re.search(pattern, text.lower())
        if match:
            return match.group(1)

    return ""


def generate_paraphrases(
    criterion_id: str,
    criterion_text: str,
    n_paraphrases: int = 12,
) -> List[str]:
    """Generate paraphrases for a single criterion."""
    paraphrases = [criterion_text]  # Original always first

    # Apply template simplifications
    template_variants = apply_template_patterns(criterion_text)
    for v in template_variants:
        if len(paraphrases) < n_paraphrases and v not in paraphrases:
            paraphrases.append(v)

    # Apply synonym replacements
    synonym_variants = apply_synonym_replacement(criterion_text)
    for v in synonym_variants:
        if len(paraphrases) < n_paraphrases and v not in paraphrases:
            paraphrases.append(v)

    # Apply synonyms to simplified versions
    for simplified in template_variants[:3]:
        syn_variants = apply_synonym_replacement(simplified)
        for v in syn_variants:
            if len(paraphrases) < n_paraphrases and v not in paraphrases:
                paraphrases.append(v)

    # Add question forms
    question_variants = generate_question_forms(criterion_text)
    for v in question_variants:
        if len(paraphrases) < n_paraphrases and v not in paraphrases:
            paraphrases.append(v)

    # Deduplicate while preserving order
    seen: Set[str] = set()
    unique_paraphrases = []
    for p in paraphrases:
        p_normalized = p.lower().strip()
        if p_normalized not in seen:
            seen.add(p_normalized)
            unique_paraphrases.append(p)

    return unique_paraphrases[:n_paraphrases]


def main():
    parser = argparse.ArgumentParser(description="Build multi-query templates")
    parser.add_argument("--criteria_path", default="data/DSM5/MDD_Criteira.json")
    parser.add_argument("--output", default="outputs/maxout/multiquery/criteria_paraphrases.json")
    parser.add_argument("--n_paraphrases", type=int, default=12, help="Target paraphrases per criterion")
    parser.add_argument("--allow_external_api", action="store_true", help="Allow LLM-based paraphrasing")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load criteria
    criteria = load_criteria(Path(args.criteria_path))
    print(f"Loaded {len(criteria)} criteria")

    # Generate paraphrases
    results = {
        "timestamp": datetime.now().isoformat(),
        "n_criteria": len(criteria),
        "n_paraphrases_target": args.n_paraphrases,
        "methods": ["rule_based", "synonym_replacement", "question_form"],
        "allow_external_api": args.allow_external_api,
        "criteria_paraphrases": {},
    }

    total_paraphrases = 0

    for crit in criteria:
        print(f"\n{crit.criterion_id}: {crit.text[:60]}...")

        paraphrases = generate_paraphrases(
            crit.criterion_id,
            crit.text,
            n_paraphrases=args.n_paraphrases,
        )

        results["criteria_paraphrases"][crit.criterion_id] = {
            "original": crit.text,
            "paraphrases": paraphrases,
            "count": len(paraphrases),
        }

        total_paraphrases += len(paraphrases)
        print(f"  Generated {len(paraphrases)} paraphrases")
        for i, p in enumerate(paraphrases[:3]):
            print(f"    {i+1}. {p[:80]}...")

    results["total_paraphrases"] = total_paraphrases
    results["avg_paraphrases_per_criterion"] = total_paraphrases / len(criteria)

    # Save results
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Generated {total_paraphrases} total paraphrases")
    print(f"Average per criterion: {total_paraphrases / len(criteria):.1f}")
    print(f"Saved to: {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
