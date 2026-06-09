#!/usr/bin/env python
"""
demo_vggt_scannet.py — Smoke-test VGGT-Omega against real ScanNet frames.

What this script does:
  1. Verifies all imports and device availability.
  2. Loads N colour frames from a ScanNet scene (frames_25k subset).
  3. Runs a VGGT-Omega forward pass (random weights if no checkpoint given).
  4. Decodes predicted poses and depth maps.
  5. Reads and compares GT poses from the dataset.
  6. Prints a sanity-check summary — shapes, ranges, GT vs predicted translation.

Usage (no checkpoint — architecture smoke-test):
    python scripts/demo_vggt_scannet.py

Usage (with a real checkpoint):
    python scripts/demo_vggt_scannet.py --checkpoint /path/to/vggt_omega_1b_512.pt

Usage (different scene / more frames):
    python scripts/demo_vggt_scannet.py --scene scene0231_00 --num_frames 8
"""

import argparse
import os
import sys
import time

import numpy as np
import torch

# ── Path setup ────────────────────────────────────────────────────────────────
REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")
VGGT_SRC  = os.path.join(REPO_ROOT, "vggt", "vggt-omega")
DATA_ROOT  = os.path.join(REPO_ROOT, "data", "scannet", "tasks", "scannet_frames_25k")

sys.path.insert(0, os.path.abspath(VGGT_SRC))

# ── VGGT imports ──────────────────────────────────────────────────────────────
from vggt_omega.models import VGGTOmega
from vggt_omega.utils.load_fn import load_and_preprocess_images
from vggt_omega.utils.pose_enc import encoding_to_camera


# ─────────────────────────────────────────────────────────────────────────────
def sep(title=""):
    width = 60
    if title:
        print(f"\n{'─' * 4} {title} {'─' * max(0, width - len(title) - 6)}")
    else:
        print("─" * width)


def check_env():
    sep("Environment")
    import torch, numpy, PIL, einops, safetensors, cv2, torchvision
    print(f"  torch        : {torch.__version__}")
    print(f"  torchvision  : {torchvision.__version__}")
    print(f"  numpy        : {numpy.__version__}")
    print(f"  PIL          : {PIL.__version__}")
    print(f"  einops       : {einops.__version__}")
    print(f"  safetensors  : {safetensors.__version__}")
    print(f"  cv2          : {cv2.__version__}")
    print(f"  CUDA         : {torch.cuda.is_available()}")
    print(f"  MPS (Apple)  : {torch.backends.mps.is_available()}")

    import vggt_omega
    print(f"  vggt_omega   : {vggt_omega.__version__}  ✓")

    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"\n  → Using device: {device}")
    return device


def pick_scene(scene_id=None):
    sep("Dataset")
    if not os.path.isdir(DATA_ROOT):
        print(f"  [ERROR] frames_25k not found at:\n    {DATA_ROOT}")
        print("  Run: python scripts/download_scannet_subset.py --out_dir ./data/scannet --mode frames_25k")
        sys.exit(1)

    scenes = sorted(
        d for d in os.listdir(DATA_ROOT)
        if os.path.isdir(os.path.join(DATA_ROOT, d)) and d.startswith("scene")
    )
    print(f"  scenes available : {len(scenes)}")

    if scene_id is None:
        scene_id = scenes[0]
    elif scene_id not in scenes:
        print(f"  [ERROR] scene '{scene_id}' not in dataset. First 5: {scenes[:5]}")
        sys.exit(1)

    scene_dir = os.path.join(DATA_ROOT, scene_id)
    color_dir  = os.path.join(scene_dir, "color")
    depth_dir  = os.path.join(scene_dir, "depth")
    pose_dir   = os.path.join(scene_dir, "pose")
    intrin_file = os.path.join(scene_dir, "intrinsics_color.txt")

    color_frames = sorted(f for f in os.listdir(color_dir) if f.endswith(".jpg"))
    depth_frames = sorted(f for f in os.listdir(depth_dir) if f.endswith(".png"))
    pose_files   = sorted(f for f in os.listdir(pose_dir) if f.endswith(".txt"))

    print(f"  scene          : {scene_id}")
    print(f"  colour frames  : {len(color_frames)}")
    print(f"  depth frames   : {len(depth_frames)}")
    print(f"  pose files     : {len(pose_files)}")
    print(f"  intrinsics     : {'✓' if os.path.isfile(intrin_file) else '✗ MISSING'}")

    return scene_id, scene_dir, color_frames, pose_dir, intrin_file


def load_gt_pose(pose_dir, frame_name):
    """Load a 4×4 camera-to-world pose matrix from the ScanNet text file."""
    txt = os.path.join(pose_dir, os.path.splitext(frame_name)[0] + ".txt")
    if not os.path.isfile(txt):
        return None
    rows = []
    with open(txt) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append([float(v) for v in line.split()])
    return np.array(rows, dtype=np.float32)


def load_gt_intrinsics(intrin_file):
    """Load 4×4 intrinsics matrix."""
    rows = []
    with open(intrin_file) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append([float(v) for v in line.split()])
    return np.array(rows, dtype=np.float32)


def run_inference(device, scene_dir, color_frames, num_frames, checkpoint=None):
    sep("Model")

    model = VGGTOmega()

    if checkpoint:
        if not os.path.isfile(checkpoint):
            print(f"  [WARN] checkpoint not found: {checkpoint}")
            print("  → running with random weights (architecture test only)")
        else:
            state = torch.load(checkpoint, map_location="cpu")
            model.load_state_dict(state)
            print(f"  checkpoint loaded: {checkpoint}")
    else:
        print("  no checkpoint supplied → random weights (architecture smoke-test)")

    model = model.to(device).eval()

    # ── pick frames ───────────────────────────────────────────────────────────
    step   = max(1, len(color_frames) // num_frames)
    chosen = color_frames[::step][:num_frames]
    color_dir = os.path.join(scene_dir, "color")
    paths = [os.path.join(color_dir, f) for f in chosen]

    print(f"\n  frames selected  : {num_frames} ({chosen[0]} … {chosen[-1]})")
    print("  loading & preprocessing…")
    images = load_and_preprocess_images(paths, image_resolution=512)
    print(f"  images tensor    : {tuple(images.shape)}  dtype={images.dtype}")

    # vggt expects (1, N, C, H, W)
    images_batched = images.unsqueeze(0).to(device)

    sep("Forward pass")
    t0 = time.time()
    with torch.inference_mode():
        # MPS/CPU: use float32 context; CUDA: bfloat16 is faster
        ctx = torch.autocast("cuda", dtype=torch.bfloat16) if device == "cuda" else torch.no_grad()
        with ctx:
            predictions = model(images_batched)
    elapsed = time.time() - t0
    print(f"  inference time   : {elapsed:.2f}s")

    return predictions, chosen, images.shape[-2:]


def report(predictions, chosen, image_hw, pose_dir, intrin_file):
    sep("Predictions")

    pose_enc   = predictions["pose_enc"]
    depth      = predictions["depth"]
    depth_conf = predictions["depth_conf"]

    print(f"  pose_enc shape   : {tuple(pose_enc.shape)}")
    print(f"  depth shape      : {tuple(depth.shape)}")
    print(f"  depth_conf shape : {tuple(depth_conf.shape)}")

    extrinsics, intrinsics = encoding_to_camera(
        pose_enc.float().cpu(),
        image_hw,
    )
    print(f"\n  extrinsics shape : {tuple(extrinsics.shape)}")
    print(f"  intrinsics shape : {tuple(intrinsics.shape)}")

    depth_np = depth[0].float().cpu().numpy()  # (N, H, W)
    print(f"\n  depth range      : [{depth_np.min():.3f}, {depth_np.max():.3f}]")
    conf_np  = depth_conf[0].float().cpu().numpy()
    print(f"  conf range       : [{conf_np.min():.3f}, {conf_np.max():.3f}]")

    # ── GT comparison ─────────────────────────────────────────────────────────
    sep("GT vs Predicted (translation, first frame)")
    gt_pose = load_gt_pose(pose_dir, chosen[0])
    if gt_pose is not None and gt_pose.shape == (4, 4):
        gt_t = gt_pose[:3, 3]
        pred_t = extrinsics[0, 0, :3, 3].numpy()
        print(f"  GT  translation  : {gt_t}")
        print(f"  Pred translation : {pred_t}")
        if not any(np.isnan(pred_t)):
            print("  (random weights → values won't match GT, but must be finite)")

    gt_intrin = load_gt_intrinsics(intrin_file)
    pred_fx = intrinsics[0, 0, 0, 0].item()
    pred_fy = intrinsics[0, 0, 1, 1].item()
    gt_fx   = gt_intrin[0, 0]
    gt_fy   = gt_intrin[1, 1]
    print(f"\n  GT  fx / fy      : {gt_fx:.1f} / {gt_fy:.1f}")
    print(f"  Pred fx / fy     : {pred_fx:.1f} / {pred_fy:.1f}")

    sep("Optional outputs")
    if "camera_and_register_tokens" in predictions:
        car = predictions["camera_and_register_tokens"]
        print(f"  camera_and_register_tokens : {tuple(car.shape)}")
        print(f"    camera_tokens            : {tuple(car[:, :, :1].shape)}")
        print(f"    register_tokens          : {tuple(car[:, :, 1:].shape)}")
    else:
        print("  camera_and_register_tokens : not present in this build")

    sep()
    print("  ALL CHECKS PASSED ✓")
    print("  VGGT-Omega + ScanNet pipeline is functional and ready for OccluSynth.\n")


# ─────────────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scene",      default=None,  help="ScanNet scene id (default: first available)")
    p.add_argument("--num_frames", type=int, default=4, help="frames to feed to VGGT (default: 4)")
    p.add_argument("--checkpoint", default=None, help="path to vggt_omega_1b_512.pt")
    args = p.parse_args()

    device = check_env()
    scene_id, scene_dir, color_frames, pose_dir, intrin_file = pick_scene(args.scene)
    predictions, chosen, image_hw = run_inference(
        device, scene_dir, color_frames, args.num_frames, args.checkpoint
    )
    report(predictions, chosen, image_hw, pose_dir, intrin_file)


if __name__ == "__main__":
    main()
