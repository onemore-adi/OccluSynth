"""
astar_planner.py — risk-graded A* path planner on OccluSynth voxel grids.

Collapses the 3D state grid to a 2D floor-plan cost map at robot height, then
runs 8-connected A* with Euclidean heuristic.

Voxel state convention (matches fusion/scene_grid.py):
    UNOBSERVABLE = 0  FREE = 1  SURFACE = 2  OCCLUDED = 3

Grid axes: (nx, ny, nz), axis-2 is vertical (z / gravity-aligned).
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

# Voxel state constants (mirrored from fusion layer — do not import to avoid coupling)
_UNOBSERVABLE = 0
_FREE = 1
_SURFACE = 2
_OCCLUDED = 3


@dataclass
class PlannerConfig:
    lambda_risk: float = 4.0        # risk weight applied to p_occupied for occluded cells
    robot_height_lo: float = 0.10   # lower bound of robot body height above floor (m)
    robot_height_hi: float = 0.50   # upper bound of robot body height above floor (m)


# ---------------------------------------------------------------------------
# Cost map construction
# ---------------------------------------------------------------------------

def build_cost_map(
    state: np.ndarray,
    completed_sdf: np.ndarray,
    config: PlannerConfig,
    voxel_size: float = 0.05,
) -> np.ndarray:
    """Build a 2D (nx, ny) cost map by collapsing the z-axis.

    Args:
        state:          uint8 (nx, ny, nz) — voxel state labels
        completed_sdf:  float32 (nx, ny, nz) — completed SDF in metres (sdf<0 = occupied)
        config:         PlannerConfig
        voxel_size:     metres per voxel (default 0.05 m = 5 cm)

    Returns:
        cost_map: float64 (nx, ny); inf = impassable, 6.0 = unobservable,
                  1.0..1+lambda_risk = traversable.

    Column cost rules (per (x,y) in the robot height band):
        - any SURFACE        → inf
        - any OCCLUDED       → 1.0 + lambda_risk * p_occupied
                               where p_occupied = frac(occluded cells with sdf < 0)
        - any FREE (no occ.) → 1.0
        - all UNOBSERVABLE   → 6.0
    """
    nx, ny, nz = state.shape

    # Locate the floor: minimum z index of any FREE voxel in the entire grid
    free_zs = np.argwhere(state == _FREE)
    if len(free_zs) == 0:
        raise ValueError("No FREE voxels in state grid — cannot determine floor z.")
    z_floor = int(free_zs[:, 2].min())

    vs = voxel_size

    z_lo = z_floor + int(round(config.robot_height_lo / vs))
    z_hi = z_floor + int(round(config.robot_height_hi / vs))
    z_lo = max(z_lo, 0)
    z_hi = min(z_hi, nz - 1)
    if z_lo > z_hi:
        raise ValueError(f"Empty height band after clamping: z_lo={z_lo} z_hi={z_hi} nz={nz}")

    # Extract height-band slices (nx, ny, band)
    band_state = state[:, :, z_lo : z_hi + 1]            # uint8
    band_sdf   = completed_sdf[:, :, z_lo : z_hi + 1].astype(np.float64)

    cost_map = np.full((nx, ny), np.inf, dtype=np.float64)

    has_surface = np.any(band_state == _SURFACE, axis=2)  # (nx, ny) bool
    has_occluded = np.any(band_state == _OCCLUDED, axis=2)
    has_free = np.any(band_state == _FREE, axis=2)

    # Traversable: no SURFACE
    traversable = ~has_surface

    # All-UNOBSERVABLE columns (no FREE, no OCCLUDED, no SURFACE)
    all_unobs = traversable & ~has_occluded & ~has_free
    cost_map[all_unobs] = 6.0

    # Pure FREE columns (traversable, no OCCLUDED, has FREE)
    pure_free = traversable & has_free & ~has_occluded
    cost_map[pure_free] = 1.0

    # Columns with OCCLUDED voxels (may also have FREE; not SURFACE)
    occ_cols = traversable & has_occluded
    if occ_cols.any():
        # p_occupied per column: fraction of OCCLUDED voxels with sdf < 0
        occ_mask_3d = (band_state == _OCCLUDED)          # (nx, ny, band)
        occ_sdf_neg = (band_sdf < 0) & occ_mask_3d       # occupied prediction

        n_occ = occ_mask_3d.sum(axis=2).astype(np.float64)     # (nx, ny)
        n_neg = occ_sdf_neg.sum(axis=2).astype(np.float64)

        # np.where evaluates both branches before masking, so guard the division
        with np.errstate(divide="ignore", invalid="ignore"):
            p_occ = np.where(n_occ > 0, n_neg / n_occ, 0.0)
        cost_map[occ_cols] = 1.0 + config.lambda_risk * p_occ[occ_cols]

    return cost_map


# ---------------------------------------------------------------------------
# A* search
# ---------------------------------------------------------------------------

_MOVES = [
    (-1, -1), (-1, 0), (-1, 1),
    ( 0, -1),          ( 0, 1),
    ( 1, -1), ( 1, 0), ( 1, 1),
]
_MOVE_DISTS = [np.sqrt(di * di + dj * dj) for di, dj in _MOVES]


def astar(
    cost_map: np.ndarray,
    start: Tuple[int, int],
    goal: Tuple[int, int],
) -> Optional[List[Tuple[int, int]]]:
    """8-connected A* on a 2D cost map.

    Edge cost = move_distance × destination cell cost (diagonal = √2 × cost).
    Heuristic = Euclidean distance (admissible: all finite costs ≥ 1.0).

    Returns path as list of (i, j) tuples from start to goal (inclusive),
    or None if goal is unreachable.
    """
    nx, ny = cost_map.shape
    si, sj = start
    gi, gj = goal

    if cost_map[si, sj] == np.inf:
        return None
    if cost_map[gi, gj] == np.inf:
        return None
    if start == goal:
        return [start]

    dist = np.full((nx, ny), np.inf, dtype=np.float64)
    dist[si, sj] = 0.0
    prev: dict[Tuple[int, int], Tuple[int, int]] = {}

    def _h(i: int, j: int) -> float:
        return np.sqrt((i - gi) ** 2 + (j - gj) ** 2)

    # heap entries: (f_score, g_score, i, j)
    heap: list = [(0.0 + _h(si, sj), 0.0, si, sj)]

    while heap:
        f, g, i, j = heapq.heappop(heap)
        if g > dist[i, j] + 1e-12:
            continue
        if (i, j) == goal:
            path = []
            cur: Tuple[int, int] = goal
            while cur != start:
                path.append(cur)
                cur = prev[cur]
            path.append(start)
            path.reverse()
            return path

        for (di, dj), move_d in zip(_MOVES, _MOVE_DISTS):
            ni, nj = i + di, j + dj
            if not (0 <= ni < nx and 0 <= nj < ny):
                continue
            c = cost_map[ni, nj]
            if c == np.inf:
                continue
            new_g = g + move_d * c
            if new_g < dist[ni, nj] - 1e-12:
                dist[ni, nj] = new_g
                prev[(ni, nj)] = (i, j)
                heapq.heappush(heap, (new_g + _h(ni, nj), new_g, ni, nj))

    return None


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def farthest_free_pair(
    cost_map: np.ndarray,
) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """Return the two mutually-reachable free cells that are farthest apart.

    Uses double-BFS (hop-count) approximation of graph diameter:
        1. BFS from any free seed → farthest cell A
        2. BFS from A            → farthest cell B (= goal)
    Returns (A, B).
    """
    nx, ny = cost_map.shape
    free = cost_map < np.inf   # (nx, ny) bool

    if not free.any():
        raise ValueError("cost map has no traversable cells")

    seed_ij = tuple(np.argwhere(free)[0])

    def _bfs_farthest(src: Tuple[int, int]) -> Tuple[int, int]:
        visited = np.zeros((nx, ny), dtype=bool)
        visited[src] = True
        frontier = [src]
        farthest = src
        while frontier:
            nxt = []
            for (i, j) in frontier:
                for di, dj in _MOVES:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < nx and 0 <= nj < ny and not visited[ni, nj] and free[ni, nj]:
                        visited[ni, nj] = True
                        nxt.append((ni, nj))
                        farthest = (ni, nj)
            frontier = nxt
        return farthest

    cell_a = _bfs_farthest(seed_ij)
    cell_b = _bfs_farthest(cell_a)
    # Convert numpy ints → plain Python ints for clean printing/hashing
    return (int(cell_a[0]), int(cell_a[1])), (int(cell_b[0]), int(cell_b[1]))


def path_geom_length(path: List[Tuple[int, int]], voxel_size: float) -> float:
    """Geometric length of a grid path in metres."""
    total = 0.0
    for (i0, j0), (i1, j1) in zip(path[:-1], path[1:]):
        total += np.sqrt((i1 - i0) ** 2 + (j1 - j0) ** 2) * voxel_size
    return total


def _count_occluded_on_path(
    path: List[Tuple[int, int]],
    band_state: np.ndarray,
) -> int:
    """Count path cells that have at least one OCCLUDED voxel in their height band."""
    count = 0
    for i, j in path:
        if np.any(band_state[i, j] == _OCCLUDED):
            count += 1
    return count
