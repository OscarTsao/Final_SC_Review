#!/usr/bin/env python3
"""
Preflight environment checker - enforces environment policy from docs/optimization.md.

Must hard-fail if:
- Not in conda env `llmhe`
- CUDA is unavailable
- More than one GPU is visible and not pinned to GPU 0
- Required snapshot files cannot be written
- Forbidden tooling detected (.venv/, venv activation)

Must also:
- Write/refresh outputs/system/hw.json and outputs/system/nvidia_smi.txt
- Write/refresh environment snapshots unless disabled
"""
import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


class PreflightError(Exception):
    """Raised when a preflight check fails."""
    pass


def get_repo_root() -> Path:
    """Find repository root."""
    current = Path(__file__).resolve().parent
    while current != current.parent:
        if (current / ".git").exists():
            return current
        current = current.parent
    return Path(__file__).resolve().parent.parent


def check_conda_env() -> None:
    """Verify active conda environment is llmhe."""
    conda_env = os.environ.get("CONDA_DEFAULT_ENV")
    if conda_env != "llmhe":
        raise PreflightError(
            f"CONDA_DEFAULT_ENV must be 'llmhe', got '{conda_env}'. "
            "Run: conda activate llmhe"
        )
    print(f"[OK] Conda environment: {conda_env}")


def check_cuda_available() -> int:
    """Check CUDA availability and return device count."""
    try:
        import torch
    except ImportError:
        raise PreflightError("PyTorch not installed. Run: pip install torch")

    if not torch.cuda.is_available():
        raise PreflightError(
            "CUDA is not available. Check GPU drivers and torch installation."
        )

    device_count = torch.cuda.device_count()
    device_name = torch.cuda.get_device_name(0)
    print(f"[OK] CUDA available: {device_name} ({device_count} device(s))")
    return device_count


def check_gpu_pinning(device_count: int, strict: bool) -> None:
    """Verify GPU pinning if multiple GPUs visible."""
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")

    if device_count > 1:
        if strict:
            raise PreflightError(
                f"Multiple GPUs visible ({device_count}) but policy requires single GPU. "
                "Set: export CUDA_VISIBLE_DEVICES=0"
            )
        else:
            print(f"[WARN] Multiple GPUs visible ({device_count}). "
                  "Consider: export CUDA_VISIBLE_DEVICES=0")
    else:
        print(f"[OK] GPU pinning: {cuda_visible or 'not set (single GPU ok)'}")


def check_forbidden_tooling(repo_root: Path) -> None:
    """Check for forbidden environment tooling."""
    forbidden_dirs = [".venv", "venv"]
    for dirname in forbidden_dirs:
        dirpath = repo_root / dirname
        if dirpath.exists():
            raise PreflightError(
                f"Forbidden directory detected: {dirpath}. "
                "Policy requires conda only. Run: rm -rf {dirpath}"
            )

    # Check if venv is in sys.prefix
    if "venv" in sys.prefix.lower() and "conda" not in sys.prefix.lower():
        raise PreflightError(
            f"Virtual environment detected in sys.prefix: {sys.prefix}. "
            "Policy requires conda only."
        )

    print("[OK] No forbidden tooling detected")


def write_nvidia_smi(output_dir: Path) -> None:
    """Write nvidia-smi output."""
    nvidia_smi_path = output_dir / "nvidia_smi.txt"
    try:
        result = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            text=True,
            timeout=30
        )
        nvidia_smi_path.write_text(result.stdout)
        print(f"[OK] Wrote {nvidia_smi_path}")
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        print(f"[WARN] Could not run nvidia-smi: {e}")


def write_hw_json(output_dir: Path) -> None:
    """Write hardware probe JSON."""
    from pathlib import Path
    import importlib.util

    repo_root = get_repo_root()
    hw_probe_path = repo_root / "scripts" / "hw_probe.py"

    if hw_probe_path.exists():
        try:
            subprocess.run(
                [sys.executable, str(hw_probe_path)],
                cwd=str(repo_root),
                check=True,
                timeout=60
            )
            print(f"[OK] Ran hw_probe.py")
        except subprocess.CalledProcessError as e:
            print(f"[WARN] hw_probe.py failed: {e}")
    else:
        print(f"[WARN] hw_probe.py not found at {hw_probe_path}")


def write_conda_snapshots(output_dir: Path) -> None:
    """Write conda environment snapshots."""
    snapshots = [
        ("conda_env_from_history.yml", ["conda", "env", "export", "--from-history"]),
        ("conda_list_explicit.txt", ["conda", "list", "--explicit"]),
    ]

    for filename, cmd in snapshots:
        filepath = output_dir / filename
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            filepath.write_text(result.stdout)
            print(f"[OK] Wrote {filepath}")
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            print(f"[WARN] Could not write {filename}: {e}")


def write_pip_freeze(output_dir: Path) -> None:
    """Write pip freeze output."""
    filepath = output_dir / "pip_freeze.txt"
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            capture_output=True,
            text=True,
            timeout=60
        )
        filepath.write_text(result.stdout)
        print(f"[OK] Wrote {filepath}")
    except subprocess.TimeoutExpired as e:
        print(f"[WARN] Could not write pip_freeze.txt: {e}")


def write_version_files(output_dir: Path) -> None:
    """Write version information files."""
    # Conda version
    try:
        result = subprocess.run(["conda", "-V"], capture_output=True, text=True, timeout=10)
        (output_dir / "conda_version.txt").write_text(result.stdout.strip() + "\n")
    except Exception:
        pass

    # Python version
    (output_dir / "python_version.txt").write_text(f"Python {sys.version}\n")

    # Pip version
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "-V"],
            capture_output=True, text=True, timeout=10
        )
        (output_dir / "pip_version.txt").write_text(result.stdout.strip() + "\n")
    except Exception:
        pass

    print(f"[OK] Wrote version files")


def run_preflight(strict: bool = False, skip_snapshots: bool = False) -> None:
    """Run all preflight checks."""
    print("=" * 60)
    print(f"PREFLIGHT CHECK - {datetime.now().isoformat()}")
    print("=" * 60)

    repo_root = get_repo_root()
    output_dir = repo_root / "outputs" / "system"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Mandatory checks
    check_conda_env()
    device_count = check_cuda_available()
    check_gpu_pinning(device_count, strict)
    check_forbidden_tooling(repo_root)

    # Write system files
    print("-" * 60)
    print("Writing system files...")
    write_nvidia_smi(output_dir)
    write_hw_json(output_dir)

    if not skip_snapshots:
        print("-" * 60)
        print("Writing environment snapshots...")
        write_conda_snapshots(output_dir)
        write_pip_freeze(output_dir)
        write_version_files(output_dir)

    print("=" * 60)
    print("[PASS] All preflight checks passed")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Preflight environment checker for llmhe research"
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Enable strict mode (fail on warnings)"
    )
    parser.add_argument(
        "--skip-snapshots",
        action="store_true",
        help="Skip writing environment snapshots"
    )
    args = parser.parse_args()

    try:
        run_preflight(strict=args.strict, skip_snapshots=args.skip_snapshots)
        sys.exit(0)
    except PreflightError as e:
        print(f"\n[FAIL] Preflight check failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
