#!/usr/bin/env python3
"""MAXOUT++ Research Driver for S-C evidence retrieval.

Extends the research pipeline with MAXOUT++ phases for pushing beyond
the current best deployable system with SOTA-style upgrades.

Usage:
    python scripts/research_driver.py --phase maxout --budget maxout --strict
    python scripts/research_driver.py --phase maxout --budget maxout --allow_external_api --teacher_mode gemini
    python scripts/research_driver.py --phase maxout --budget maxout --resume
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
DATA_DIR = PROJECT_ROOT / "data"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
CONFIGS_DIR = PROJECT_ROOT / "configs"
MAXOUT_DIR = OUTPUTS_DIR / "maxout"


@dataclass
class HardwareConfig:
    """Detected hardware configuration."""
    gpu_name: str = "Unknown"
    gpu_vram_gb: float = 0.0
    gpu_count: int = 0
    cuda_available: bool = False
    cpu_cores: int = 1
    ram_gb: float = 0.0


@dataclass
class MaxoutBudgets:
    """Budgets for MAXOUT++ phases with auto-scaling support."""
    # Inference HPO
    inference_hpo_trials: int = 50000
    inference_hpo_early_stop_frac: float = 0.02  # Top 2% get full eval
    inference_hpo_fidelity_levels: List[float] = field(default_factory=lambda: [0.2, 0.5, 1.0])

    # Reranker training HPO
    reranker_hpo_trials: int = 500
    reranker_hpo_top_full_eval: int = 10
    reranker_hpo_top_seeds: int = 3
    reranker_seeds: int = 5

    # Retriever finetune HPO
    retriever_hpo_trials: int = 200
    retriever_hpo_top_full_eval: int = 5
    retriever_hpo_top_seeds: int = 2
    retriever_seeds: int = 3

    # Multi-query
    multiquery_paraphrases_per_criterion: int = 12
    multiquery_min_paraphrases: int = 8
    multiquery_max_paraphrases: int = 20

    # Ensemble stacking
    ensemble_max_base_models: int = 30
    ensemble_cv_folds: int = 5

    # GNN
    gnn_sweep_configs: int = 300
    gnn_top_full_eval: int = 10
    gnn_top_seeds: int = 3

    # Postprocessing threshold HPO
    postprocess_hpo_trials: int = 10000

    # Retriever zoo
    retriever_zoo_top_k_max: int = 800
    retriever_zoo_oracle_ks: List[int] = field(default_factory=lambda: [50, 100, 200, 400, 800])

    # Scaling factors for VRAM limitations
    vram_scale_factor: float = 1.0
    batch_size_scale: float = 1.0


def detect_hardware() -> HardwareConfig:
    """Detect hardware configuration."""
    config = HardwareConfig()

    try:
        import torch
        config.cuda_available = torch.cuda.is_available()
        if config.cuda_available:
            config.gpu_count = torch.cuda.device_count()
            config.gpu_name = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            config.gpu_vram_gb = props.total_memory / (1024**3)
    except Exception as e:
        logger.warning(f"Could not detect GPU: {e}")

    try:
        import psutil
        config.cpu_cores = psutil.cpu_count(logical=False) or 1
        config.ram_gb = psutil.virtual_memory().total / (1024**3)
    except Exception:
        import multiprocessing
        config.cpu_cores = multiprocessing.cpu_count()

    return config


def load_budgets(budget_type: str, hw_config: HardwareConfig) -> MaxoutBudgets:
    """Load and auto-scale budgets based on hardware."""
    budgets = MaxoutBudgets()

    # Try to load from config file
    budget_config_path = CONFIGS_DIR / "budgets_maxout.yaml"
    if budget_config_path.exists():
        with open(budget_config_path) as f:
            config = yaml.safe_load(f)
            if budget_type in config:
                for key, value in config[budget_type].items():
                    if hasattr(budgets, key):
                        setattr(budgets, key, value)

    # Auto-scale based on VRAM
    if hw_config.gpu_vram_gb > 0:
        if hw_config.gpu_vram_gb >= 30:  # RTX 5090 / A100 class
            budgets.vram_scale_factor = 1.0
            budgets.batch_size_scale = 1.0
        elif hw_config.gpu_vram_gb >= 20:  # RTX 4090 class
            budgets.vram_scale_factor = 0.8
            budgets.batch_size_scale = 0.75
        elif hw_config.gpu_vram_gb >= 12:  # RTX 3080/4080 class
            budgets.vram_scale_factor = 0.5
            budgets.batch_size_scale = 0.5
        else:  # Smaller GPUs
            budgets.vram_scale_factor = 0.25
            budgets.batch_size_scale = 0.25

        # Apply scaling
        budgets.inference_hpo_trials = int(budgets.inference_hpo_trials * budgets.vram_scale_factor)
        budgets.reranker_hpo_trials = int(budgets.reranker_hpo_trials * budgets.vram_scale_factor)
        budgets.retriever_hpo_trials = int(budgets.retriever_hpo_trials * budgets.vram_scale_factor)
        budgets.gnn_sweep_configs = int(budgets.gnn_sweep_configs * budgets.vram_scale_factor)
        budgets.retriever_zoo_top_k_max = int(budgets.retriever_zoo_top_k_max * budgets.batch_size_scale)

    return budgets


def run_command(cmd: List[str], description: str, timeout: int = 3600,
                cwd: Optional[Path] = None, env: Optional[Dict] = None) -> Tuple[bool, str]:
    """Run a command and capture output."""
    logger.info(f"{'='*60}")
    logger.info(f"RUNNING: {description}")
    logger.info(f"Command: {' '.join(cmd)}")
    logger.info(f"{'='*60}")

    try:
        run_env = os.environ.copy()
        if env:
            run_env.update(env)

        result = subprocess.run(
            cmd,
            cwd=cwd or PROJECT_ROOT,
            timeout=timeout,
            capture_output=True,
            text=True,
            env=run_env,
        )

        output = result.stdout + result.stderr

        if result.returncode != 0:
            logger.error(f"Command failed with return code {result.returncode}")
            logger.error(f"Output: {output[:2000]}")
            return False, output

        return True, output

    except subprocess.TimeoutExpired:
        logger.error(f"Command timed out after {timeout}s")
        return False, "TIMEOUT"
    except Exception as e:
        logger.error(f"Command error: {e}")
        return False, str(e)


def verify_preconditions() -> bool:
    """Verify all MAXOUT++ preconditions."""
    logger.info("Verifying MAXOUT++ preconditions...")

    checks = []

    # 1. Git status
    success, output = run_command(["git", "rev-parse", "HEAD"], "Get HEAD hash")
    if success:
        head_hash = output.strip()
        logger.info(f"HEAD hash: {head_hash}")
        checks.append(True)
    else:
        logger.error("Failed to get git HEAD")
        checks.append(False)

    # 2. Invariants
    success, _ = run_command(
        ["python", "scripts/verify_invariants.py"],
        "Verify invariants",
        timeout=300
    )
    checks.append(success)

    # 3. Pytest
    success, _ = run_command(
        ["python", "-m", "pytest", "-q"],
        "Run pytest",
        timeout=300
    )
    checks.append(success)

    # 4. Validate runs
    success, _ = run_command(
        ["python", "scripts/validate_runs.py"],
        "Validate paper artifacts",
        timeout=300
    )
    checks.append(success)

    # 5. Locked baseline config
    locked_config = CONFIGS_DIR / "locked_best_stageE_deploy.yaml"
    if locked_config.exists():
        logger.info(f"Locked baseline config exists: {locked_config}")
        checks.append(True)
    else:
        logger.error(f"Missing locked baseline config: {locked_config}")
        checks.append(False)

    # 6. Run registry
    registry = OUTPUTS_DIR / "run_registry.csv"
    if registry.exists():
        logger.info(f"Run registry exists: {registry}")
        checks.append(True)
    else:
        logger.error(f"Missing run registry: {registry}")
        checks.append(False)

    all_passed = all(checks)
    if all_passed:
        logger.info("All preconditions PASSED")
    else:
        logger.error("Some preconditions FAILED - fix before continuing")

    return all_passed


def phase_maxout_retriever_zoo(budgets: MaxoutBudgets, hw_config: HardwareConfig) -> bool:
    """Phase 2: Build retriever zoo and evaluate candidates."""
    logger.info("="*70)
    logger.info("MAXOUT++ PHASE 2: RETRIEVER ZOO")
    logger.info("="*70)

    # Create zoo directory
    zoo_dir = MAXOUT_DIR / "retriever_zoo"
    zoo_dir.mkdir(parents=True, exist_ok=True)

    # Define retriever candidates
    retrievers = [
        {"name": "bge-m3", "model_id": "BAAI/bge-m3", "type": "hybrid"},
        {"name": "bge-large-en-v1.5", "model_id": "BAAI/bge-large-en-v1.5", "type": "dense"},
        {"name": "e5-large-v2", "model_id": "intfloat/e5-large-v2", "type": "dense"},
        {"name": "gte-large-en-v1.5", "model_id": "Alibaba-NLP/gte-large-en-v1.5", "type": "dense"},
        {"name": "stella-en-1.5B-v5", "model_id": "dunzhang/stella_en_1.5B_v5", "type": "dense"},
        {"name": "bm25", "model_id": "bm25", "type": "lexical"},
    ]

    # Save config
    config_path = zoo_dir / "retriever_zoo_config.json"
    with open(config_path, "w") as f:
        json.dump({
            "retrievers": retrievers,
            "top_k_max": budgets.retriever_zoo_top_k_max,
            "oracle_ks": budgets.retriever_zoo_oracle_ks,
            "hw_config": {
                "gpu": hw_config.gpu_name,
                "vram_gb": hw_config.gpu_vram_gb,
            }
        }, f, indent=2)

    logger.info(f"Retriever zoo config saved to: {config_path}")
    logger.info("Retriever zoo evaluation requires implementation of zoo.py")
    logger.info("Skipping for now - will be implemented in Phase 2 of MAXOUT++")

    return True


def phase_maxout_multiquery(budgets: MaxoutBudgets, allow_external_api: bool) -> bool:
    """Phase 3: Multi-query retrieval with criterion paraphrases."""
    logger.info("="*70)
    logger.info("MAXOUT++ PHASE 3: MULTI-QUERY RETRIEVAL")
    logger.info("="*70)

    multiquery_dir = MAXOUT_DIR / "multiquery"
    multiquery_dir.mkdir(parents=True, exist_ok=True)

    # Config for paraphrase generation
    config = {
        "paraphrases_per_criterion": budgets.multiquery_paraphrases_per_criterion,
        "min_paraphrases": budgets.multiquery_min_paraphrases,
        "max_paraphrases": budgets.multiquery_max_paraphrases,
        "allow_external_api": allow_external_api,
        "methods": ["rule_based", "synonym_replacement"],
    }

    if allow_external_api:
        config["methods"].append("llm_paraphrase")

    config_path = multiquery_dir / "multiquery_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    logger.info(f"Multi-query config saved to: {config_path}")
    logger.info("Multi-query template generation requires implementation")

    return True


def phase_maxout_reranker(budgets: MaxoutBudgets, teacher_mode: str) -> bool:
    """Phase 4: Reranker MAXOUT with listwise + multi-reranker + distillation."""
    logger.info("="*70)
    logger.info("MAXOUT++ PHASE 4: RERANKER MAXOUT")
    logger.info("="*70)

    reranker_dir = MAXOUT_DIR / "reranker"
    reranker_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "hpo_trials": budgets.reranker_hpo_trials,
        "top_full_eval": budgets.reranker_hpo_top_full_eval,
        "top_seeds": budgets.reranker_hpo_top_seeds,
        "seeds": budgets.reranker_seeds,
        "teacher_mode": teacher_mode,
        "reranker_bases": [
            {"name": "jina-reranker-v3", "model_id": "jinaai/jina-reranker-v3", "lora": True},
            {"name": "bge-reranker-v2-m3", "model_id": "BAAI/bge-reranker-v2-m3", "lora": True},
        ],
        "training_config": {
            "hard_neg_mining": True,
            "include_empty_groups": True,
            "listwise_loss": True,
            "lora_rank": [4, 8, 16],
            "lora_alpha": [16, 32],
            "lora_dropout": [0.0, 0.1],
        },
    }

    config_path = reranker_dir / "reranker_maxout_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    logger.info(f"Reranker MAXOUT config saved to: {config_path}")

    if teacher_mode != "none":
        logger.info(f"Teacher mode: {teacher_mode}")
        logger.info("Teacher distillation requires API access")

    return True


def phase_maxout_ensemble(budgets: MaxoutBudgets) -> bool:
    """Phase 6: Ensemble + stacking meta-learner."""
    logger.info("="*70)
    logger.info("MAXOUT++ PHASE 6: ENSEMBLE STACKING")
    logger.info("="*70)

    ensemble_dir = MAXOUT_DIR / "ensemble"
    ensemble_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "max_base_models": budgets.ensemble_max_base_models,
        "cv_folds": budgets.ensemble_cv_folds,
        "feature_sources": [
            "retriever_dense_score",
            "retriever_sparse_score",
            "retriever_colbert_score",
            "reranker_score",
            "rrf_rank",
            "sentence_position",
            "sentence_length",
            "criterion_overlap_tfidf",
        ],
        "meta_learners": [
            "logistic_regression",
            "xgboost",
            "lightgbm",
        ],
        "nested_cv": True,
    }

    config_path = ensemble_dir / "ensemble_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    logger.info(f"Ensemble config saved to: {config_path}")

    return True


def phase_maxout_abstention(budgets: MaxoutBudgets) -> bool:
    """Phase 8: Risk-controlled abstention."""
    logger.info("="*70)
    logger.info("MAXOUT++ PHASE 8: RISK-CONTROLLED ABSTENTION")
    logger.info("="*70)

    abstention_dir = MAXOUT_DIR / "abstention"
    abstention_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "hpo_trials": budgets.postprocess_hpo_trials,
        "methods": [
            "conformal_risk_control",
            "calibrated_threshold",
            "selective_prediction",
        ],
        "epsilon_grid": [0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
        "target_metrics": [
            "false_evidence_rate",
            "empty_precision",
            "risk_coverage_auc",
        ],
    }

    config_path = abstention_dir / "abstention_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    logger.info(f"Abstention config saved to: {config_path}")

    return True


def run_maxout_pipeline(args) -> int:
    """Run the complete MAXOUT++ pipeline."""
    logger.info("="*70)
    logger.info("MAXOUT++ RESEARCH DRIVER")
    logger.info(f"Started: {datetime.now().isoformat()}")
    logger.info("="*70)

    # Create maxout output directory
    MAXOUT_DIR.mkdir(parents=True, exist_ok=True)

    # Detect hardware
    hw_config = detect_hardware()
    logger.info(f"Hardware: {hw_config.gpu_name} ({hw_config.gpu_vram_gb:.1f}GB VRAM)")
    logger.info(f"CPU cores: {hw_config.cpu_cores}, RAM: {hw_config.ram_gb:.1f}GB")

    # Load budgets
    budgets = load_budgets(args.budget, hw_config)
    logger.info(f"Budget type: {args.budget}")
    logger.info(f"VRAM scale factor: {budgets.vram_scale_factor}")
    logger.info(f"Inference HPO trials: {budgets.inference_hpo_trials}")

    # Save run manifest
    manifest = {
        "start_time": datetime.now().isoformat(),
        "args": vars(args),
        "hardware": {
            "gpu": hw_config.gpu_name,
            "vram_gb": hw_config.gpu_vram_gb,
            "cpu_cores": hw_config.cpu_cores,
            "ram_gb": hw_config.ram_gb,
        },
        "budgets": {
            "inference_hpo_trials": budgets.inference_hpo_trials,
            "reranker_hpo_trials": budgets.reranker_hpo_trials,
            "vram_scale_factor": budgets.vram_scale_factor,
        },
    }

    manifest_path = MAXOUT_DIR / "run_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    # Verify preconditions if strict mode
    if args.strict:
        if not verify_preconditions():
            logger.error("Preconditions failed - aborting")
            return 1

    # Run phases
    phases_to_run = [
        ("Phase 2: Retriever Zoo", lambda: phase_maxout_retriever_zoo(budgets, hw_config)),
        ("Phase 3: Multi-Query", lambda: phase_maxout_multiquery(budgets, args.allow_external_api)),
        ("Phase 4: Reranker MAXOUT", lambda: phase_maxout_reranker(budgets, args.teacher_mode)),
        ("Phase 6: Ensemble", lambda: phase_maxout_ensemble(budgets)),
        ("Phase 8: Abstention", lambda: phase_maxout_abstention(budgets)),
    ]

    for phase_name, phase_fn in phases_to_run:
        logger.info(f"\nStarting {phase_name}...")
        try:
            if not phase_fn():
                logger.error(f"{phase_name} failed")
                if args.strict:
                    return 1
        except Exception as e:
            logger.error(f"{phase_name} error: {e}")
            if args.strict:
                return 1

    # Update manifest with completion
    manifest["end_time"] = datetime.now().isoformat()
    manifest["status"] = "COMPLETED"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info("="*70)
    logger.info("MAXOUT++ PIPELINE SETUP COMPLETE")
    logger.info("="*70)
    logger.info(f"Outputs: {MAXOUT_DIR}")
    logger.info("Run individual phase scripts to execute experiments")

    return 0


def main():
    parser = argparse.ArgumentParser(
        description="MAXOUT++ Research Driver for S-C Evidence Retrieval",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--phase",
        choices=["maxout", "retriever_zoo", "multiquery", "reranker", "ensemble", "abstention", "final"],
        default="maxout",
        help="Which phase to run (default: maxout runs all)",
    )
    parser.add_argument(
        "--budget",
        choices=["quick", "standard", "maxout", "exhaustive"],
        default="maxout",
        help="Compute budget level",
    )
    parser.add_argument(
        "--allow_external_api",
        action="store_true",
        help="Allow external API calls (for teacher distillation/LLM paraphrases)",
    )
    parser.add_argument(
        "--teacher_mode",
        choices=["none", "gemini", "rankgpt"],
        default="none",
        help="Teacher model for distillation (requires --allow_external_api)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last checkpoint",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Strict mode: verify all preconditions and fail fast",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Dry run: plan but don't execute",
    )

    args = parser.parse_args()

    # Validate args
    if args.teacher_mode != "none" and not args.allow_external_api:
        logger.error("--teacher_mode requires --allow_external_api")
        return 1

    if args.phase == "maxout":
        return run_maxout_pipeline(args)
    else:
        logger.info(f"Running single phase: {args.phase}")
        # Individual phase execution would go here
        return 0


if __name__ == "__main__":
    sys.exit(main())
