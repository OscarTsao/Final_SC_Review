import json
from pathlib import Path

import pytest

from final_sc_review.hpo.cache_builder import validate_cache_split


def test_hpo_refuses_test_split(tmp_path: Path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    manifest = {"split": "test"}
    with open(cache_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f)

    with pytest.raises(ValueError):
        validate_cache_split(cache_dir, "val")


def test_hpo_accepts_dev_split(tmp_path: Path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    manifest = {"split": "val"}
    with open(cache_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f)

    validate_cache_split(cache_dir, "val")
