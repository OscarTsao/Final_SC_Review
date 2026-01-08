#!/usr/bin/env python3
"""Hardware probe script for paper-grade reproducibility.

Outputs hardware info to outputs/hw.json for experiment tracking.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def get_cpu_info() -> dict:
    """Get CPU information."""
    info = {
        "processor": platform.processor(),
        "machine": platform.machine(),
        "physical_cores": os.cpu_count(),
    }

    # Try to get more detailed info on Linux
    try:
        with open("/proc/cpuinfo", "r") as f:
            cpuinfo = f.read()
            for line in cpuinfo.split("\n"):
                if "model name" in line:
                    info["model_name"] = line.split(":")[1].strip()
                    break
    except Exception:
        pass

    return info


def get_memory_info() -> dict:
    """Get memory information."""
    info = {}

    try:
        with open("/proc/meminfo", "r") as f:
            meminfo = f.read()
            for line in meminfo.split("\n"):
                if "MemTotal" in line:
                    # Convert from kB to GB
                    kb = int(line.split()[1])
                    info["total_gb"] = round(kb / (1024 * 1024), 2)
                    break
    except Exception:
        pass

    return info


def get_gpu_info() -> dict:
    """Get GPU information using nvidia-smi."""
    info = {"available": False, "devices": []}

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version,cuda_version",
             "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            info["available"] = True
            for i, line in enumerate(result.stdout.strip().split("\n")):
                if line.strip():
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) >= 4:
                        info["devices"].append({
                            "index": i,
                            "name": parts[0],
                            "memory_mb": int(parts[1]) if parts[1].isdigit() else parts[1],
                            "driver_version": parts[2],
                            "cuda_version": parts[3],
                        })
    except Exception as e:
        info["error"] = str(e)

    return info


def get_torch_info() -> dict:
    """Get PyTorch and CUDA information."""
    info = {}

    try:
        import torch
        info["torch_version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["cuda_version"] = torch.version.cuda
            info["cudnn_version"] = str(torch.backends.cudnn.version())
            info["device_count"] = torch.cuda.device_count()
            info["current_device"] = torch.cuda.current_device()
            info["device_name"] = torch.cuda.get_device_name(0)
            info["device_capability"] = list(torch.cuda.get_device_capability(0))
    except ImportError:
        info["error"] = "PyTorch not installed"
    except Exception as e:
        info["error"] = str(e)

    return info


def get_python_info() -> dict:
    """Get Python environment information."""
    return {
        "version": platform.python_version(),
        "implementation": platform.python_implementation(),
        "executable": sys.executable,
        "prefix": sys.prefix,
    }


def get_package_versions() -> dict:
    """Get versions of key packages."""
    packages = [
        "transformers",
        "sentence-transformers",
        "FlagEmbedding",
        "numpy",
        "pandas",
        "optuna",
        "tqdm",
        "pyyaml",
    ]

    versions = {}
    for pkg in packages:
        try:
            import importlib.metadata
            versions[pkg] = importlib.metadata.version(pkg.replace("-", "_"))
        except Exception:
            try:
                mod = __import__(pkg.replace("-", "_"))
                versions[pkg] = getattr(mod, "__version__", "unknown")
            except Exception:
                versions[pkg] = "not installed"

    return versions


def probe_hardware() -> dict:
    """Probe all hardware and software information."""
    return {
        "timestamp": datetime.now().isoformat(),
        "hostname": platform.node(),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
        },
        "cpu": get_cpu_info(),
        "memory": get_memory_info(),
        "gpu": get_gpu_info(),
        "python": get_python_info(),
        "torch": get_torch_info(),
        "packages": get_package_versions(),
    }


def get_nvidia_smi_full() -> str:
    """Get full nvidia-smi output."""
    try:
        result = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        return result.stdout
    except Exception as e:
        return f"Error: {e}"


def compute_recommended_settings(hw_info: dict) -> dict:
    """Compute recommended batch sizes based on VRAM."""
    vram_gb = 0
    # First try nvidia-smi detection
    if hw_info.get("gpu", {}).get("available"):
        devices = hw_info["gpu"].get("devices", [])
        if devices:
            vram_mb = devices[0].get("memory_mb", 0)
            if isinstance(vram_mb, int):
                vram_gb = vram_mb / 1024

    # Fallback to torch detection
    if vram_gb == 0 and hw_info.get("torch", {}).get("cuda_available"):
        try:
            import torch
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        except Exception:
            pass

    if vram_gb >= 28:
        return {
            "vram_detected_gb": vram_gb,
            "top_k_retriever_max": 800,
            "top_k_rerank_max": 400,
            "batch_size_embed": 128,
            "batch_size_rerank": 64,
            "budget_tier": "high",
        }
    elif vram_gb >= 20:
        return {
            "vram_detected_gb": vram_gb,
            "top_k_retriever_max": 400,
            "top_k_rerank_max": 200,
            "batch_size_embed": 64,
            "batch_size_rerank": 32,
            "budget_tier": "medium",
        }
    elif vram_gb >= 12:
        return {
            "vram_detected_gb": vram_gb,
            "top_k_retriever_max": 200,
            "top_k_rerank_max": 100,
            "batch_size_embed": 32,
            "batch_size_rerank": 16,
            "budget_tier": "low",
        }
    else:
        return {
            "vram_detected_gb": vram_gb,
            "top_k_retriever_max": 100,
            "top_k_rerank_max": 50,
            "batch_size_embed": 16,
            "batch_size_rerank": 8,
            "budget_tier": "minimal",
        }


def main():
    """Main entry point."""
    # Probe hardware
    hw_info = probe_hardware()

    # Add recommended settings
    hw_info["recommended"] = compute_recommended_settings(hw_info)

    # Print to console
    print("=" * 60)
    print("Hardware Probe Results")
    print("=" * 60)
    print(json.dumps(hw_info, indent=2))
    print("=" * 60)

    # Ensure outputs/system directory exists
    output_dir = Path(__file__).parent.parent / "outputs" / "system"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write hw.json
    output_path = output_dir / "hw.json"
    with open(output_path, "w") as f:
        json.dump(hw_info, f, indent=2)

    # Write nvidia_smi.txt
    nvidia_smi_output = get_nvidia_smi_full()
    nvidia_smi_path = output_dir / "nvidia_smi.txt"
    with open(nvidia_smi_path, "w") as f:
        f.write(nvidia_smi_output)

    print(f"\nHardware info saved to: {output_path}")
    print(f"nvidia-smi output saved to: {nvidia_smi_path}")

    # Summary for quick reference
    print("\n--- Quick Summary ---")
    print(f"CPU: {hw_info['cpu'].get('model_name', hw_info['cpu'].get('processor', 'Unknown'))}")
    print(f"Memory: {hw_info['memory'].get('total_gb', 'Unknown')} GB")

    if hw_info["gpu"]["available"]:
        for dev in hw_info["gpu"]["devices"]:
            print(f"GPU {dev['index']}: {dev['name']} ({dev['memory_mb']} MB)")
    else:
        print("GPU: Not available")

    if hw_info["torch"].get("cuda_available"):
        print(f"PyTorch: {hw_info['torch']['torch_version']} (CUDA {hw_info['torch']['cuda_version']})")
    else:
        print(f"PyTorch: {hw_info['torch'].get('torch_version', 'Not installed')} (CPU only)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
