"""
Device selection and repo-path helpers.

All path resolution flows through here so the rest of the package
never hard-codes repo-relative strings.
"""

from __future__ import annotations

import os
from pathlib import Path


# ── repo root ─────────────────────────────────────────────────────────────────

def get_repo_root() -> Path:
    """
    Return the repository root as a Path.

    Resolution order:
      1. OCCLUSYNTH_REPO_ROOT env var  (set this in CI / containers)
      2. Walk upward from this file until we find pyproject.toml
      3. Fallback: cwd
    """
    if env := os.environ.get("OCCLUSYNTH_REPO_ROOT"):
        return Path(env).resolve()

    # Walk up from src/occlusynth/utils/device.py
    candidate = Path(__file__).resolve()
    for _ in range(6):
        candidate = candidate.parent
        if (candidate / "pyproject.toml").exists():
            return candidate

    return Path.cwd()


# ── config (reads [tool.occlusynth] from pyproject.toml) ─────────────────────

def get_config() -> dict:
    """
    Return the [tool.occlusynth] table from pyproject.toml, with defaults.
    Falls back gracefully if tomllib / tomli is unavailable.
    """
    defaults = {
        "vggt_omega_src":     "vggt/vggt-omega",
        "default_checkpoint": "vggt/vggt-omega/checkpoints/vggt_omega_1b_512.pt",
        "scannet_frames_25k": "data/scannet/tasks/scannet_frames_25k",
        "cache_dir":          "demo_outputs/pred_cache",
    }

    root = get_repo_root()
    toml_path = root / "pyproject.toml"
    if not toml_path.exists():
        return defaults

    try:
        try:
            import tomllib                  # Python 3.11+
        except ImportError:
            import tomli as tomllib         # type: ignore[no-redef]

        with open(toml_path, "rb") as f:
            data = tomllib.load(f)
        tool_cfg = data.get("tool", {}).get("occlusynth", {})
        return {**defaults, **tool_cfg}
    except Exception:
        return defaults


# ── torch device ──────────────────────────────────────────────────────────────

def get_device() -> str:
    """
    Return the best available torch device string: 'cuda' > 'mps' > 'cpu'.

    Prefer CUDA for production training.  MPS for Apple Silicon dev.
    CPU as last resort (slow but always works).
    """
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"
