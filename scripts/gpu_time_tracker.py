#!/usr/bin/env python3
"""GPU time tracking for research experiments.

Tracks GPU-active time across training runs to ensure minimum compute budget.
Target: 12 hours of GPU-active time for paper experiments.

Usage:
    from gpu_time_tracker import GPUTimeTracker
    tracker = GPUTimeTracker()
    tracker.start()
    # ... training code ...
    tracker.stop()
    print(tracker.summary())
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

# Minimum GPU hours for paper experiments
MIN_GPU_HOURS = 12.0


@dataclass
class GPUSession:
    """Single GPU session record."""
    session_id: str
    start_time: float
    end_time: Optional[float] = None
    phase: str = ""
    description: str = ""
    peak_memory_gb: float = 0.0
    avg_utilization: float = 0.0

    @property
    def duration_seconds(self) -> float:
        if self.end_time is None:
            return time.time() - self.start_time
        return self.end_time - self.start_time

    @property
    def duration_hours(self) -> float:
        return self.duration_seconds / 3600

    def to_dict(self) -> Dict:
        return {
            "session_id": self.session_id,
            "start_time": datetime.fromtimestamp(self.start_time).isoformat(),
            "end_time": datetime.fromtimestamp(self.end_time).isoformat() if self.end_time else None,
            "duration_seconds": self.duration_seconds,
            "duration_hours": self.duration_hours,
            "phase": self.phase,
            "description": self.description,
            "peak_memory_gb": self.peak_memory_gb,
            "avg_utilization": self.avg_utilization,
        }


class GPUTimeTracker:
    """Track GPU time across research sessions."""

    def __init__(self, output_dir: str = "outputs/system"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.state_file = self.output_dir / "gpu_time_total.json"
        self.sessions: List[GPUSession] = []
        self.current_session: Optional[GPUSession] = None
        self._monitor_thread: Optional[threading.Thread] = None
        self._stop_monitor = False
        self._utilization_samples: List[float] = []
        self._memory_samples: List[float] = []

        # Load existing state
        self._load_state()

    def _load_state(self) -> None:
        """Load previous tracking state."""
        if self.state_file.exists():
            try:
                with open(self.state_file) as f:
                    data = json.load(f)
                # Load previous sessions
                for s in data.get("sessions", []):
                    session = GPUSession(
                        session_id=s["session_id"],
                        start_time=datetime.fromisoformat(s["start_time"]).timestamp(),
                        end_time=datetime.fromisoformat(s["end_time"]).timestamp() if s.get("end_time") else None,
                        phase=s.get("phase", ""),
                        description=s.get("description", ""),
                        peak_memory_gb=s.get("peak_memory_gb", 0),
                        avg_utilization=s.get("avg_utilization", 0),
                    )
                    self.sessions.append(session)
            except Exception as e:
                print(f"Warning: Could not load GPU tracking state: {e}")

    def _save_state(self) -> None:
        """Save tracking state to disk."""
        data = {
            "total_gpu_hours": self.total_gpu_hours,
            "total_gpu_seconds": self.total_gpu_seconds,
            "n_sessions": len(self.sessions),
            "min_target_hours": MIN_GPU_HOURS,
            "target_reached": self.total_gpu_hours >= MIN_GPU_HOURS,
            "sessions": [s.to_dict() for s in self.sessions],
            "last_updated": datetime.now().isoformat(),
        }
        with open(self.state_file, "w") as f:
            json.dump(data, f, indent=2)

    @property
    def total_gpu_seconds(self) -> float:
        """Total GPU-active seconds across all sessions."""
        total = sum(s.duration_seconds for s in self.sessions if s.end_time is not None)
        if self.current_session:
            total += self.current_session.duration_seconds
        return total

    @property
    def total_gpu_hours(self) -> float:
        """Total GPU-active hours across all sessions."""
        return self.total_gpu_seconds / 3600

    @property
    def remaining_hours(self) -> float:
        """Hours remaining to reach minimum target."""
        return max(0, MIN_GPU_HOURS - self.total_gpu_hours)

    @property
    def target_reached(self) -> bool:
        """Whether minimum GPU hours target has been reached."""
        return self.total_gpu_hours >= MIN_GPU_HOURS

    def start(self, phase: str = "", description: str = "") -> str:
        """Start a new GPU tracking session.

        Args:
            phase: Current research phase (e.g., "reranker_train")
            description: Description of this session

        Returns:
            Session ID
        """
        session_id = f"session_{int(time.time())}_{len(self.sessions)}"
        self.current_session = GPUSession(
            session_id=session_id,
            start_time=time.time(),
            phase=phase,
            description=description,
        )

        # Start monitoring thread
        self._stop_monitor = False
        self._utilization_samples = []
        self._memory_samples = []
        self._monitor_thread = threading.Thread(target=self._monitor_gpu, daemon=True)
        self._monitor_thread.start()

        print(f"GPU tracking started: {session_id}")
        print(f"  Phase: {phase}")
        print(f"  Total GPU hours so far: {self.total_gpu_hours:.2f}h")
        print(f"  Remaining to target: {self.remaining_hours:.2f}h")

        return session_id

    def stop(self) -> GPUSession:
        """Stop current tracking session.

        Returns:
            Completed session object
        """
        if self.current_session is None:
            raise RuntimeError("No active session to stop")

        # Stop monitoring
        self._stop_monitor = True
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2.0)

        # Record final stats
        self.current_session.end_time = time.time()
        if self._utilization_samples:
            self.current_session.avg_utilization = sum(self._utilization_samples) / len(self._utilization_samples)
        if self._memory_samples:
            self.current_session.peak_memory_gb = max(self._memory_samples)

        # Save session
        session = self.current_session
        self.sessions.append(session)
        self.current_session = None

        # Save state
        self._save_state()

        print(f"GPU tracking stopped: {session.session_id}")
        print(f"  Duration: {session.duration_hours:.2f}h ({session.duration_seconds:.0f}s)")
        print(f"  Avg utilization: {session.avg_utilization:.1f}%")
        print(f"  Peak memory: {session.peak_memory_gb:.1f} GB")
        print(f"  Total GPU hours: {self.total_gpu_hours:.2f}h")

        return session

    def _monitor_gpu(self) -> None:
        """Background thread to monitor GPU utilization."""
        while not self._stop_monitor:
            try:
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                     "--format=csv,noheader,nounits"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if result.returncode == 0:
                    for line in result.stdout.strip().split("\n"):
                        parts = line.split(",")
                        if len(parts) >= 2:
                            util = float(parts[0].strip())
                            mem_mb = float(parts[1].strip())
                            self._utilization_samples.append(util)
                            self._memory_samples.append(mem_mb / 1024)  # Convert to GB
            except Exception:
                pass
            time.sleep(1.0)  # Sample every second

    def summary(self) -> Dict[str, Any]:
        """Get tracking summary."""
        return {
            "total_gpu_hours": round(self.total_gpu_hours, 2),
            "total_gpu_seconds": round(self.total_gpu_seconds, 1),
            "n_sessions": len(self.sessions),
            "min_target_hours": MIN_GPU_HOURS,
            "target_reached": self.target_reached,
            "remaining_hours": round(self.remaining_hours, 2),
            "current_session": self.current_session.session_id if self.current_session else None,
        }

    def format_summary(self) -> str:
        """Format summary for display."""
        s = self.summary()
        status = "REACHED" if s["target_reached"] else f"{s['remaining_hours']:.1f}h remaining"
        lines = [
            "=" * 50,
            "GPU TIME TRACKING SUMMARY",
            "=" * 50,
            f"Total GPU hours: {s['total_gpu_hours']:.2f}h",
            f"Target (minimum): {s['min_target_hours']:.0f}h",
            f"Status: {status}",
            f"Sessions completed: {s['n_sessions']}",
            "=" * 50,
        ]
        return "\n".join(lines)


# Global tracker instance
_global_tracker: Optional[GPUTimeTracker] = None


def get_tracker() -> GPUTimeTracker:
    """Get or create global GPU time tracker."""
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = GPUTimeTracker()
    return _global_tracker


def start_session(phase: str = "", description: str = "") -> str:
    """Start GPU tracking session (convenience function)."""
    return get_tracker().start(phase, description)


def stop_session() -> GPUSession:
    """Stop GPU tracking session (convenience function)."""
    return get_tracker().stop()


def get_summary() -> Dict[str, Any]:
    """Get tracking summary (convenience function)."""
    return get_tracker().summary()


def main():
    """CLI for checking GPU time tracking status."""
    import argparse
    parser = argparse.ArgumentParser(description="GPU time tracking utility")
    parser.add_argument("--status", action="store_true", help="Show current status")
    parser.add_argument("--reset", action="store_true", help="Reset tracking state")
    args = parser.parse_args()

    tracker = GPUTimeTracker()

    if args.reset:
        if tracker.state_file.exists():
            tracker.state_file.unlink()
        print("GPU tracking state reset")
        return

    print(tracker.format_summary())


if __name__ == "__main__":
    main()
