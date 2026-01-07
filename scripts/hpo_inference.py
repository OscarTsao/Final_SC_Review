#!/usr/bin/env python3
"""Run inference-stage HPO using cached scores."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from final_sc_review.hpo.cache_builder import build_cache, validate_cache_split
from final_sc_review.hpo.objective_inference import InferenceObjective, ObjectiveConfig, load_cache
from final_sc_review.hpo.reporting import TrialLogWriter, build_run_manifest, write_json
from final_sc_review.hpo.search_space import sample_inference_params
from final_sc_review.hpo.storage import get_or_create_study
from final_sc_review.utils.logging import get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/hpo_inference.yaml")
    parser.add_argument("--study_name", type=str, default="sc_inference")
    parser.add_argument("--storage", type=str, default="sqlite:///outputs/hpo/optuna.db")
    parser.add_argument("--n_trials", type=int, default=100)
    parser.add_argument("--timeout", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pruner", type=str, default="hyperband")
    args = parser.parse_args()

    cfg = _load_cfg(Path(args.config))
    cache_dir = build_cache(Path(args.config), force_rebuild=False)
    validate_cache_split(cache_dir, cfg["split"].get("dev_split", "val"))
    cache_path = cache_dir / "dev_cache.npz"
    cache = load_cache(cache_path)

    output_dir = Path(cfg["paths"]["hpo_output_dir"]) / args.study_name
    output_dir.mkdir(parents=True, exist_ok=True)

    trial_log = TrialLogWriter(output_dir / "trials.csv")
    objective_cfg = ObjectiveConfig(
        ks=cfg["evaluation"]["ks"],
        skip_no_positives=cfg["evaluation"]["skip_no_positives"],
        objective_metric=cfg["hpo"]["objective_metric"],
        prune_chunk_frac=cfg["hpo"]["prune_chunk_frac"],
        prune_min_queries=cfg["hpo"]["prune_min_queries"],
    )

    objective = InferenceObjective(cache, objective_cfg, trial_log, seed=args.seed)

    study = get_or_create_study(
        storage=args.storage,
        study_name=args.study_name,
        direction="maximize",
        seed=args.seed,
        pruner_name=args.pruner,
    )

    def _objective(trial):
        params = sample_inference_params(trial, cfg)
        return objective(trial, params)

    study.optimize(_objective, n_trials=args.n_trials, timeout=args.timeout)

    best_params = study.best_trial.params if study.best_trial else {}
    write_json(output_dir / "best_params.json", best_params)

    manifest = build_run_manifest(
        repo_dir=Path("."),
        config_snapshot=cfg,
        dataset_checksums=_dataset_checksums(cfg),
        cache_dir=cache_dir,
    )
    write_json(output_dir / "manifest.json", manifest)

    logger.info("Best value: %s", study.best_value if study.best_trial else "n/a")


def _load_cfg(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _dataset_checksums(cfg: dict) -> dict:
    from final_sc_review.hpo.reporting import compute_sha256

    return {
        "groundtruth": compute_sha256(Path(cfg["paths"]["groundtruth"])),
        "sentence_corpus": compute_sha256(Path(cfg["paths"]["sentence_corpus"])),
        "criteria": compute_sha256(Path(cfg["paths"]["criteria"])),
    }


if __name__ == "__main__":
    main()
