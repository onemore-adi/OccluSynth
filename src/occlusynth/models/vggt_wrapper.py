"""
VGGTWrapper — thin, importable interface to VGGT-Omega inference.

Design rules
------------
* No side-effects at import time (no sys.path mutation at module level).
* Lazy model loading — ``load()`` must be called explicitly, or use the
  context manager: ``with VGGTWrapper(...) as w: ...``
* Prediction results are plain dicts of numpy arrays — no torch tensors leak
  out of this module.
* Caching is opt-in via ``predict_cached()``.

Moved from: scripts/depth_scale_fit.py :: run_vggt_cached()
            scripts/test-vggt.py        (original test script logic)
"""

from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from occlusynth.utils.device import get_device, get_repo_root, get_config


class VGGTWrapper:
    """
    Wraps VGGT-Omega for feed-forward depth + pose prediction.

    Args:
        ckpt_path:        path to ``vggt_omega_1b_512.pt``
        vggt_src:         path to the ``vggt-omega`` source directory
                          (the one containing ``vggt_omega/``)
        device:           'cuda' | 'mps' | 'cpu'  (None → auto-detect)
        image_resolution: longest-side resolution fed to VGGT (default 512)

    Example::

        wrapper = VGGTWrapper()
        wrapper.load()
        result = wrapper.predict(["frame0.jpg", "frame1.jpg", "frame2.jpg"])
        depth = result["depth"]   # (N, H, W) float32
        conf  = result["conf"]    # (N, H, W) float32
        extr  = result["extrinsics"]   # (N, 3, 4) float32
        intr  = result["intrinsics"]   # (N, 3, 3) float32
    """

    def __init__(
        self,
        ckpt_path:        Optional[str | Path] = None,
        vggt_src:         Optional[str | Path] = None,
        device:           Optional[str]        = None,
        image_resolution: int                  = 512,
    ) -> None:
        cfg  = get_config()
        root = get_repo_root()

        self.ckpt_path        = Path(ckpt_path  or (root / cfg["default_checkpoint"]))
        self.vggt_src         = Path(vggt_src   or (root / cfg["vggt_omega_src"]))
        self.device           = device or get_device()
        self.image_resolution = image_resolution
        self._model           = None

    # ── lazy loading ──────────────────────────────────────────────────────────

    def load(self) -> "VGGTWrapper":
        """Load the model and checkpoint into memory. Idempotent."""
        if self._model is not None:
            return self

        self._ensure_vggt_on_path()

        import torch
        from vggt_omega.models import VGGTOmega

        if not self.ckpt_path.exists():
            raise FileNotFoundError(
                f"VGGT-Omega checkpoint not found: {self.ckpt_path}\n"
                "Request access at https://huggingface.co/facebook/VGGT-Omega"
            )

        t0 = time.time()
        model = VGGTOmega()
        model.load_state_dict(
            torch.load(str(self.ckpt_path), map_location="cpu", weights_only=True)
        )
        self._model = model.to(self.device).eval()
        self._load_time = time.time() - t0
        return self

    def _ensure_vggt_on_path(self) -> None:
        """Add vggt-omega source to sys.path if not already importable."""
        try:
            import vggt_omega  # noqa: F401
            return
        except ImportError:
            pass
        import sys
        src = str(self.vggt_src)
        if src not in sys.path:
            sys.path.insert(0, src)

    # ── context manager ───────────────────────────────────────────────────────

    def __enter__(self) -> "VGGTWrapper":
        return self.load()

    def __exit__(self, *_) -> None:
        self.unload()

    def unload(self) -> None:
        """Release GPU/MPS memory."""
        self._model = None

    # ── inference ─────────────────────────────────────────────────────────────

    def predict(self, image_paths: List[str | Path]) -> Dict[str, np.ndarray]:
        """
        Run VGGT-Omega on a list of RGB images.

        Args:
            image_paths: N paths to JPEG / PNG colour frames

        Returns dict with numpy arrays (no torch tensors):
            depth       (N, H, W)    float32 — raw predicted depth (arbitrary scale)
            conf        (N, H, W)    float32 — per-pixel confidence
            pose_enc    (N, 9)       float32 — raw pose encoding (T + quat + 2×FOV)
            extrinsics  (N, 3, 4)   float32 — world-to-camera [R|t]
            intrinsics  (N, 3, 3)   float32 — predicted K
            image_hw    (2,)         int     — (H, W) of the VGGT input resolution
        """
        if self._model is None:
            raise RuntimeError("Call .load() before .predict()")

        import torch
        from vggt_omega.utils.load_fn  import load_and_preprocess_images
        from vggt_omega.utils.pose_enc import encoding_to_camera

        images   = load_and_preprocess_images(
            [str(p) for p in image_paths],
            image_resolution=self.image_resolution,
        )
        H, W     = images.shape[-2:]
        images_b = images.unsqueeze(0).to(self.device)  # (1, N, 3, H, W)

        with torch.inference_mode():
            preds = self._model(images_b)

        pose_enc   = preds["pose_enc"].float().cpu()            # (1, N, 9)
        extr, intr = encoding_to_camera(pose_enc, (H, W))

        return {
            "depth":       preds["depth"].float().cpu()[0, :, :, :, 0].numpy(),
            "conf":        preds["depth_conf"].float().cpu()[0].numpy(),
            "pose_enc":    pose_enc[0].numpy(),
            "extrinsics":  extr[0].numpy(),
            "intrinsics":  intr[0].numpy(),
            "image_hw":    np.array([H, W], dtype=np.int32),
        }

    # ── cached inference ──────────────────────────────────────────────────────

    def predict_cached(
        self,
        scene_id:  str,
        stems:     List[str],
        scene_dir: str | Path,
        cache_dir: Optional[str | Path] = None,
    ) -> Dict[str, np.ndarray]:
        """
        predict() with an on-disk .npy cache keyed on (scene_id, stems, resolution).

        The cache avoids re-running the ~14-minute MPS inference on every run.
        Cache files are stored in ``cache_dir`` (default: demo_outputs/pred_cache/).

        Returns the same dict as predict(), loaded from cache if available.
        """
        cfg = get_config()
        root = get_repo_root()
        cache_dir = Path(cache_dir or (root / cfg["cache_dir"]))
        cache_dir.mkdir(parents=True, exist_ok=True)

        stem_key  = hashlib.md5("_".join(stems).encode()).hexdigest()[:8]
        cache_key = f"{scene_id}_{stem_key}_res{self.image_resolution}"
        cache_d   = cache_dir / (cache_key + "_depth.npy")
        cache_c   = cache_dir / (cache_key + "_conf.npy")
        cache_p   = cache_dir / (cache_key + "_pose_enc.npy")
        cache_e   = cache_dir / (cache_key + "_extrinsics.npy")
        cache_i   = cache_dir / (cache_key + "_intrinsics.npy")
        cache_hw  = cache_dir / (cache_key + "_hw.npy")

        all_cache_files = [cache_d, cache_c, cache_p, cache_e, cache_i, cache_hw]
        if all(f.exists() for f in all_cache_files):
            hw = np.load(cache_hw)
            return {
                "depth":      np.load(cache_d),
                "conf":       np.load(cache_c),
                "pose_enc":   np.load(cache_p),
                "extrinsics": np.load(cache_e),
                "intrinsics": np.load(cache_i),
                "image_hw":   hw,
            }

        # Cache miss — run inference
        self.load()
        color_paths = [
            str(Path(scene_dir) / "color" / (s + ".jpg")) for s in stems
        ]
        result = self.predict(color_paths)

        np.save(cache_d,  result["depth"])
        np.save(cache_c,  result["conf"])
        np.save(cache_p,  result["pose_enc"])
        np.save(cache_e,  result["extrinsics"])
        np.save(cache_i,  result["intrinsics"])
        np.save(cache_hw, result["image_hw"])

        return result

    # ── repr ──────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        loaded = "loaded" if self._model is not None else "not loaded"
        return (
            f"VGGTWrapper(device='{self.device}', "
            f"res={self.image_resolution}, {loaded})"
        )
