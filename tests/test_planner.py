"""
tests/test_planner.py — 4 mandatory tests for the risk-graded A* planner.

(a) cost_monotonicity  — build_cost_map produces the right cost ordering
(b) no_blocked_on_path — A* path never crosses an inf-cost cell
(c) connected_path     — path[0]==start, path[-1]==goal
(d) detour_around_wall — A* detours around an injected wall; detour > straight line
"""

import numpy as np
import pytest

from occlusynth.planning import (
    PlannerConfig,
    astar,
    build_cost_map,
    farthest_free_pair,
    path_geom_length,
)

# Voxel state constants
UNOBSERVABLE = 0
FREE = 1
SURFACE = 2
OCCLUDED = 3

VOXEL_SIZE = 0.05  # 5 cm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_grid(nx: int, ny: int, nz: int):
    """Return (state, sdf) arrays initialised to all-UNOBSERVABLE / SDF=+1."""
    state = np.zeros((nx, ny, nz), dtype=np.uint8)
    sdf   = np.ones((nx, ny, nz),  dtype=np.float32)
    return state, sdf


def _set_floor(state: np.ndarray, z_floor: int = 0):
    """Mark the entire z=z_floor slice FREE so build_cost_map can find the floor."""
    state[:, :, z_floor] = FREE


# ---------------------------------------------------------------------------
# (a) Cost monotonicity
# ---------------------------------------------------------------------------

class TestCostMonotonicity:
    """build_cost_map assigns costs in the expected partial order:
        FREE==1.0  ≤  OCC(sdf>0)==1.0  ≤  OCC(sdf<0)==1+λ  <  UNOBS==6.0  <  SURF==inf
    """

    def _build(self):
        # 5 columns × 1 × 12 z-voxels; floor at z=0; band z=2..10
        nx, ny, nz = 5, 1, 12
        state, sdf = _make_grid(nx, ny, nz)
        _set_floor(state, z_floor=0)

        cfg = PlannerConfig()
        # band: z_lo = 0 + round(0.10/0.05)=2,  z_hi = 0 + round(0.50/0.05)=10
        band = slice(2, 11)

        # col 0: all FREE in band
        state[0, 0, band] = FREE

        # col 1: OCCLUDED, completed SDF > 0  (not physically occupied)
        state[1, 0, band] = OCCLUDED
        sdf[1, 0, band]   = 0.10

        # col 2: OCCLUDED, completed SDF < 0  (physically occupied)
        state[2, 0, band] = OCCLUDED
        sdf[2, 0, band]   = -0.10

        # col 3: all UNOBSERVABLE in band  (state stays 0)

        # col 4: SURFACE in band  (impassable wall)
        state[4, 0, band] = SURFACE

        cost = build_cost_map(state, sdf, cfg, voxel_size=VOXEL_SIZE)
        return cost, cfg

    def test_free_column_cost_is_one(self):
        cost, _ = self._build()
        assert cost[0, 0] == pytest.approx(1.0)

    def test_occluded_free_sdf_cost_is_one(self):
        """Occluded voxels whose completed SDF > 0 → p_occ=0 → cost 1.0."""
        cost, _ = self._build()
        assert cost[1, 0] == pytest.approx(1.0)

    def test_occluded_occupied_sdf_is_max_risk(self):
        """Occluded voxels where all completed SDF < 0 → p_occ=1 → cost 1+λ."""
        cost, cfg = self._build()
        assert cost[2, 0] == pytest.approx(1.0 + cfg.lambda_risk)

    def test_unobservable_column_cost_is_six(self):
        cost, _ = self._build()
        assert cost[3, 0] == pytest.approx(6.0)

    def test_surface_column_is_impassable(self):
        cost, _ = self._build()
        assert cost[4, 0] == np.inf

    def test_partial_order(self):
        """Strict monotonicity chain: 1.0 ≤ 1.0 ≤ 1+λ < 6.0 < inf."""
        cost, cfg = self._build()
        assert cost[0, 0] <= cost[1, 0]
        assert cost[1, 0] <= cost[2, 0]
        assert cost[2, 0] < cost[3, 0]    # 1+λ=5 < 6
        assert cost[3, 0] < cost[4, 0]    # 6 < inf


# ---------------------------------------------------------------------------
# (b) No blocked cells on path
# ---------------------------------------------------------------------------

class TestNoBlockedCellsOnPath:
    def _flat_cost_map(self, n: int = 20) -> np.ndarray:
        """All-free n×n cost map (cost 1.0 everywhere)."""
        return np.ones((n, n), dtype=np.float64)

    def test_path_avoids_inf_cells(self):
        cost = self._flat_cost_map(20)
        # Inject a wall column in the middle with a gap
        cost[10, :15] = np.inf
        start, goal = (0, 10), (19, 10)
        path = astar(cost, start, goal)
        assert path is not None, "path should exist via the gap"
        for cell in path:
            assert cost[cell] < np.inf, f"blocked cell {cell} found on path"

    def test_no_inf_cells_on_straight_path(self):
        cost = self._flat_cost_map(10)
        path = astar(cost, (0, 0), (9, 9))
        assert path is not None
        for cell in path:
            assert cost[cell] < np.inf


# ---------------------------------------------------------------------------
# (c) Connected: path[0]==start, path[-1]==goal
# ---------------------------------------------------------------------------

class TestConnectedPath:
    def test_start_and_goal_in_path(self):
        cost = np.ones((15, 15), dtype=np.float64)
        start, goal = (0, 0), (14, 14)
        path = astar(cost, start, goal)
        assert path is not None
        assert path[0] == start
        assert path[-1] == goal

    def test_path_is_8_connected(self):
        """Each step in the path must move at most 1 cell in each axis."""
        cost = np.ones((20, 20), dtype=np.float64)
        path = astar(cost, (0, 0), (19, 19))
        assert path is not None
        for (i0, j0), (i1, j1) in zip(path[:-1], path[1:]):
            assert abs(i1 - i0) <= 1 and abs(j1 - j0) <= 1

    def test_none_when_goal_blocked(self):
        cost = np.ones((10, 10), dtype=np.float64)
        cost[9, 9] = np.inf
        assert astar(cost, (0, 0), (9, 9)) is None

    def test_trivial_same_start_goal(self):
        cost = np.ones((5, 5), dtype=np.float64)
        path = astar(cost, (2, 2), (2, 2))
        assert path == [(2, 2)]

    def test_farthest_free_pair_are_reachable(self):
        """farthest_free_pair returns cells that A* can actually connect."""
        cost = np.ones((30, 30), dtype=np.float64)
        cost[15, :] = np.inf   # wall across the middle with a gap
        cost[15, 25] = 1.0     # gap
        start, goal = farthest_free_pair(cost)
        path = astar(cost, start, goal)
        assert path is not None, "farthest pair must be mutually reachable"
        assert path[0] == start
        assert path[-1] == goal


# ---------------------------------------------------------------------------
# (d) Regression: detour around an injected hazard
# ---------------------------------------------------------------------------

class TestDetourAroundWall:
    """
    Layout (25×25 grid):

        columns  0..24, rows  0..24
        SURFACE wall: col x=12, rows j=0..17  (gap at j=18..24)
        start=(0,12), goal=(24,12)

    Straight-line distance = 24 voxels.
    The direct path (x: 0→24, j stays ~12) is blocked by the wall at x=12.
    A* must route through the gap (j≥18), producing a longer path.
    """

    def _setup(self):
        nx, ny, nz = 25, 25, 12
        state, sdf = _make_grid(nx, ny, nz)
        _set_floor(state, z_floor=0)
        # band z=2..10 (floor=0, lo=2, hi=10)
        band = slice(2, 11)

        # Free the whole floor-plan
        state[:, :, band] = FREE

        # Inject SURFACE wall at x=12, all j in [0,17], across the height band
        state[12, :18, band] = SURFACE

        cfg = PlannerConfig()
        cost = build_cost_map(state, sdf, cfg, voxel_size=VOXEL_SIZE)
        return cost, cfg

    def test_wall_blocks_direct_path(self):
        cost, _ = self._setup()
        # The direct horizontal corridor x=0→24 at y=12 goes through the wall at x=12
        assert cost[12, 12] == np.inf, "wall cell must be impassable"

    def test_path_exists_via_gap(self):
        cost, _ = self._setup()
        start, goal = (0, 12), (24, 12)
        path = astar(cost, start, goal)
        assert path is not None, "path must exist via the gap at j≥18"

    def test_detour_longer_than_straight_line(self):
        cost, _ = self._setup()
        start, goal = (0, 12), (24, 12)
        path = astar(cost, start, goal)
        assert path is not None

        straight_line = path_geom_length([(0, 12), (24, 12)], VOXEL_SIZE)
        detour_length  = path_geom_length(path, VOXEL_SIZE)
        assert detour_length > straight_line, (
            f"detour ({detour_length:.2f} m) should exceed straight line ({straight_line:.2f} m)"
        )

    def test_path_crosses_gap_region(self):
        """The path must pass through j≥18 to get around the wall."""
        cost, _ = self._setup()
        path = astar(cost, (0, 12), (24, 12))
        assert path is not None
        max_j = max(j for _, j in path)
        assert max_j >= 18, f"path must swing through the gap (max j={max_j})"

    def test_no_wall_cell_on_path(self):
        cost, _ = self._setup()
        path = astar(cost, (0, 12), (24, 12))
        assert path is not None
        for cell in path:
            assert cost[cell] < np.inf, f"wall cell {cell} on path"
