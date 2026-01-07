"""Optuna storage helpers."""

from __future__ import annotations

from typing import Optional

import optuna


def make_sampler(seed: int) -> optuna.samplers.BaseSampler:
    return optuna.samplers.TPESampler(seed=seed)


def make_pruner(name: str) -> optuna.pruners.BasePruner:
    key = (name or "hyperband").lower()
    if key == "hyperband":
        return optuna.pruners.HyperbandPruner()
    if key == "successive_halving":
        return optuna.pruners.SuccessiveHalvingPruner()
    if key == "median":
        return optuna.pruners.MedianPruner()
    if key == "none":
        return optuna.pruners.NopPruner()
    raise ValueError(f"Unknown pruner: {name}")


def get_or_create_study(
    storage: str,
    study_name: str,
    direction: str,
    seed: int,
    pruner_name: str,
) -> optuna.Study:
    sampler = make_sampler(seed)
    pruner = make_pruner(pruner_name)
    return optuna.create_study(
        storage=storage,
        study_name=study_name,
        direction=direction,
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )
