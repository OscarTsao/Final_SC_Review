#!/usr/bin/env python3
"""
Full Optimization Pipeline Runner

This script:
1. Monitors ongoing training and waits for completion
2. Stage C: Assesses the retrained NO_EVIDENCE model
3. Launches HPO runs for all retriever+reranker combinations
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from final_sc_review.utils.logging import get_logger

logger = get_logger(__name__)


def wait_for_training(model_dir: str, check_interval: int = 60, max_wait: int = 14400):
    """
    Wait for training to complete by checking for model files.

    Args:
        model_dir: Path to expected model output directory
        check_interval: Seconds between checks
        max_wait: Maximum seconds to wait

    Returns:
        True if training completed, False if timeout
    """
    model_path = Path(model_dir)
    start_time = time.time()

    logger.info(f"Waiting for training to complete at {model_dir}")
    logger.info(f"Checking every {check_interval}s, max wait {max_wait}s")

    while time.time() - start_time < max_wait:
        # Check if model files exist
        config_file = model_path / "config.json"
        model_file = model_path / "model.safetensors"

        if config_file.exists() and model_file.exists():
            logger.info(f"Training complete! Model saved at {model_dir}")
            return True

        # Check if training process is still running
        result = subprocess.run(
            ["pgrep", "-f", "train_reranker_with_no_evidence"],
            capture_output=True
        )

        if result.returncode != 0:
            # Process not running - check if model exists
            if config_file.exists():
                logger.info("Training process ended, model exists")
                return True
            else:
                logger.warning("Training process ended but no model found!")
                return False

        elapsed = int(time.time() - start_time)
        logger.info(f"Training still running... ({elapsed}s elapsed)")
        time.sleep(check_interval)

    logger.error(f"Timeout waiting for training after {max_wait}s")
    return False


def run_assessment(model_dir: str, output_dir: str):
    """Run Stage C: Assess the retrained model."""
    logger.info("="*60)
    logger.info("STAGE C: Assessing NO_EVIDENCE Model")
    logger.info("="*60)

    cmd = [
        "python", "scripts/assess_no_evidence_model.py",
        "--model", model_dir,
        "--config", "configs/reranker_with_no_evidence.yaml",
        "--split", "val",
        "--outdir", output_dir
    ]

    logger.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)

    return result.returncode == 0


def run_combo_hpo(output_dir: str):
    """Run HPO for all retriever+reranker combinations."""
    logger.info("="*60)
    logger.info("STAGE D: Running HPO for All Combinations")
    logger.info("="*60)

    # Check if the script exists
    script_path = Path("scripts/run_inference_hpo_all_combos.py")
    if not script_path.exists():
        logger.warning(f"Combo HPO script not found: {script_path}")
        # Try alternative script
        script_path = Path("scripts/hpo_retriever_reranker_combinations.py")

    if not script_path.exists():
        logger.error("No combo HPO script found")
        return False

    cmd = [
        "python", str(script_path),
        "--output_dir", output_dir,
    ]

    logger.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)

    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Run full optimization pipeline")
    parser.add_argument(
        "--model_dir",
        default="outputs/reranker_with_no_evidence",
        help="Path to training model output"
    )
    parser.add_argument(
        "--assessment_dir",
        default="outputs/assessment_no_evidence_model",
        help="Path to assessment output"
    )
    parser.add_argument(
        "--combo_hpo_dir",
        default="outputs/combination_hpo",
        help="Path to combo HPO output"
    )
    parser.add_argument(
        "--skip_training_wait",
        action="store_true",
        help="Skip waiting for training (if already complete)"
    )
    parser.add_argument(
        "--skip_assessment",
        action="store_true",
        help="Skip Stage C assessment"
    )
    parser.add_argument(
        "--skip_combo_hpo",
        action="store_true",
        help="Skip combo HPO"
    )
    args = parser.parse_args()

    logger.info("="*60)
    logger.info("FULL OPTIMIZATION PIPELINE")
    logger.info("="*60)
    logger.info(f"Started at: {datetime.now().isoformat()}")

    # Step 1: Wait for training
    if not args.skip_training_wait:
        success = wait_for_training(args.model_dir)
        if not success:
            logger.error("Training did not complete successfully")
            sys.exit(1)
    else:
        logger.info("Skipping training wait (--skip_training_wait)")

    # Step 2: Stage C - Assessment
    if not args.skip_assessment:
        success = run_assessment(args.model_dir, args.assessment_dir)
        if not success:
            logger.error("Assessment failed")
            # Continue anyway to run combo HPO
    else:
        logger.info("Skipping assessment (--skip_assessment)")

    # Step 3: Combo HPO
    if not args.skip_combo_hpo:
        success = run_combo_hpo(args.combo_hpo_dir)
        if not success:
            logger.error("Combo HPO failed")
    else:
        logger.info("Skipping combo HPO (--skip_combo_hpo)")

    logger.info("="*60)
    logger.info("PIPELINE COMPLETE")
    logger.info("="*60)
    logger.info(f"Finished at: {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
