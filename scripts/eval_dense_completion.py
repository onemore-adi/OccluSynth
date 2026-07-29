#!/usr/bin/env python
"""Score completed geometry against ScanNet GT in the DENSE (n40) regime.

The val-crop frontier only covers n6 (sparse). The deliverable renders at n40,
so compare checkpoints where it matters: how close is the geometry each one
invents to the real scene, and how much of the room does it recover.

  accuracy_cm   mean distance from COMPLETED (amber) vertices -> GT mesh.
                Low = what it invents actually exists.
  completeness  fraction of GT surface within 5 cm of the completed mesh
                (measured over GT points NOT already covered by the measured
                mesh, i.e. the part only completion can win).
"""
import sys
from pathlib import Path

import numpy as np
import open3d as o3d

ROOT = Path(__file__).resolve().parent.parent
GT = ROOT / "data/scannet/scans/scene0000_00/scene0000_00_vh_clean_2.ply"
BEFORE = ROOT / "demo_outputs/before_after/scene0000_00_before_v3.ply"
VARIANTS = {
    "interim (shipping)": "scene0000_00_completed_only_v3.ply",
    "md_resume (new)":    "scene0000_00_completed_only_md.ply",
}
N_SAMPLE = 200_000


def dist_to_mesh(points, mesh):
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    q = o3d.core.Tensor(points.astype(np.float32))
    return scene.compute_distance(q).numpy()


def main():
    gt = o3d.io.read_triangle_mesh(str(GT))
    gt_pts = np.asarray(gt.sample_points_uniformly(N_SAMPLE).points)
    before = o3d.io.read_triangle_mesh(str(BEFORE))

    # GT surface the measured mesh already explains -> completion can't add there
    d_before = dist_to_mesh(gt_pts, before)
    hidden = d_before > 0.05
    print(f"GT points: {len(gt_pts):,} | not covered by measured mesh: "
          f"{hidden.sum():,} ({100*hidden.mean():.1f}%)\n")

    print(f"{'checkpoint':<22}{'accuracy_cm':>13}{'hidden_recovered':>19}{'verts':>9}")
    print("-" * 63)
    for name, fn in VARIANTS.items():
        m = o3d.io.read_triangle_mesh(str(ROOT / "demo_outputs/before_after" / fn))
        v = np.asarray(m.vertices)
        if len(v) == 0:
            print(f"{name:<22}{'(empty)':>13}")
            continue
        # accuracy: completed geometry -> GT
        acc = dist_to_mesh(v, gt).mean() * 100
        # completeness: hidden GT -> completed geometry
        d_hidden = dist_to_mesh(gt_pts[hidden], m)
        rec = (d_hidden < 0.05).mean() * 100
        print(f"{name:<22}{acc:>13.2f}{rec:>18.1f}%{len(v):>9,}")


if __name__ == "__main__":
    main()
