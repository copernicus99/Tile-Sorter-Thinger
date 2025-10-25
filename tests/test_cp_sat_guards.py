import importlib
import sys
import types

import pytest


def _install_ortools_stub():
    """Provide a minimal ortools stub so ``solver.cp_sat`` can import."""

    ortools_mod = types.ModuleType("ortools")
    sat_mod = types.ModuleType("ortools.sat")
    python_mod = types.ModuleType("ortools.sat.python")
    cp_model_mod = types.ModuleType("ortools.sat.python.cp_model")

    class _MissingCp:
        OPTIMAL = 4
        FEASIBLE = 3
        INFEASIBLE = 2
        MODEL_INVALID = 1

        class CpModel:  # pragma: no cover - stub for import only
            pass

        class CpSolver:  # pragma: no cover - stub for import only
            pass

    cp_model_mod.CpModel = _MissingCp.CpModel
    cp_model_mod.CpSolver = _MissingCp.CpSolver
    cp_model_mod.OPTIMAL = _MissingCp.OPTIMAL
    cp_model_mod.FEASIBLE = _MissingCp.FEASIBLE
    cp_model_mod.INFEASIBLE = _MissingCp.INFEASIBLE
    cp_model_mod.MODEL_INVALID = _MissingCp.MODEL_INVALID

    ortools_mod.sat = sat_mod
    sat_mod.python = python_mod
    python_mod.cp_model = cp_model_mod

    sys.modules["ortools"] = ortools_mod
    sys.modules["ortools.sat"] = sat_mod
    sys.modules["ortools.sat.python"] = python_mod
    sys.modules["ortools.sat.python.cp_model"] = cp_model_mod


@pytest.fixture
def cp_sat_module():
    for mod in [
        "solver.cp_sat",
        "ortools.sat.python.cp_model",
        "ortools.sat.python",
        "ortools.sat",
        "ortools",
    ]:
        sys.modules.pop(mod, None)

    _install_ortools_stub()
    module = importlib.import_module("solver.cp_sat")
    try:
        yield module
    finally:
        for mod in [
            "solver.cp_sat",
            "ortools.sat.python.cp_model",
            "ortools.sat.python",
            "ortools.sat",
            "ortools",
        ]:
            sys.modules.pop(mod, None)


def test_edge_guard_disabled_in_test_mode(cp_sat_module):
    helper = cp_sat_module._compute_edge_guard_cells
    assert helper(6, cell_size=cp_sat_module.CELL, board_max=20, test_mode=True) is None


def test_edge_guard_uses_cell_size(cp_sat_module):
    helper = cp_sat_module._compute_edge_guard_cells
    assert helper(6, cell_size=0.5, board_max=20, test_mode=False) == 12
    assert helper(6, cell_size=1.0, board_max=20, test_mode=False) == 6


def test_edge_guard_skips_small_boards(cp_sat_module):
    helper = cp_sat_module._compute_edge_guard_cells
    # Guard would be 12 cells, but a 10-cell board should disable it entirely
    assert helper(6, cell_size=0.5, board_max=10, test_mode=False) is None


def test_no_plus_guard_respects_test_mode(cp_sat_module):
    cfg = cp_sat_module.CFG
    original_no_plus = getattr(cfg, "NO_PLUS", None)
    original_test_mode = getattr(cfg, "TEST_MODE", None)
    try:
        setattr(cfg, "NO_PLUS", True)
        setattr(cfg, "TEST_MODE", True)
        assert not cp_sat_module._no_plus_guard_enabled(cfg)

        setattr(cfg, "TEST_MODE", False)
        assert cp_sat_module._no_plus_guard_enabled(cfg)
    finally:
        if original_no_plus is not None:
            setattr(cfg, "NO_PLUS", original_no_plus)
        if original_test_mode is not None:
            setattr(cfg, "TEST_MODE", original_test_mode)


def test_guard_backoff_skips_edge_guard_first(cp_sat_module):
    attempts = []

    def fake_attempt(*, edge_guard_cells, plus_guard_enabled):
        attempts.append((edge_guard_cells, plus_guard_enabled))
        if edge_guard_cells is None:
            return True, ["ok"], None
        return False, [], "Proven infeasible under current constraints"

    ok, placed, reason, edge_used, plus_used = cp_sat_module._resolve_guard_backoffs(
        fake_attempt,
        edge_guard_cells=3,
        plus_guard_enabled=True,
    )

    assert ok
    assert placed == ["ok"]
    assert reason is None
    assert edge_used is None
    assert plus_used is True
    assert attempts == [(3, True), (None, True)]


def test_guard_backoff_drops_plus_guard_when_needed(cp_sat_module):
    attempts = []

    def fake_attempt(*, edge_guard_cells, plus_guard_enabled):
        attempts.append((edge_guard_cells, plus_guard_enabled))
        if plus_guard_enabled:
            return False, [], "Proven infeasible under current constraints"
        return True, ["ok"], None

    ok, placed, reason, edge_used, plus_used = cp_sat_module._resolve_guard_backoffs(
        fake_attempt,
        edge_guard_cells=None,
        plus_guard_enabled=True,
    )

    assert ok
    assert placed == ["ok"]
    assert reason is None
    assert edge_used is None
    assert plus_used is False
    assert attempts == [(None, True), (None, False)]


def test_guard_backoff_preserves_non_infeasible_reason(cp_sat_module):
    def fake_attempt(*, edge_guard_cells, plus_guard_enabled):
        return False, [], "Model invalid (configuration error)"

    ok, placed, reason, edge_used, plus_used = cp_sat_module._resolve_guard_backoffs(
        fake_attempt,
        edge_guard_cells=5,
        plus_guard_enabled=True,
    )

    assert not ok
    assert placed == []
    assert reason == "Model invalid (configuration error)"
    assert edge_used == 5
    assert plus_used is True


def test_crash_demand_guards_disable_backtracking(monkeypatch, cp_sat_module):
    from models import ft_to_cells
    from tests.data.crash_demand import CRASH_DEMAND_BAG_FT

    Rect = cp_sat_module.Rect

    monkeypatch.setattr(cp_sat_module.CFG, "SAME_SHAPE_LIMIT", 1, raising=False)
    monkeypatch.setattr(cp_sat_module.CFG, "MAX_EDGE_FT", 6.0, raising=False)
    monkeypatch.setattr(cp_sat_module.CFG, "NO_PLUS", False, raising=False)
    monkeypatch.setattr(cp_sat_module.CFG, "TEST_MODE", False, raising=False)
    monkeypatch.setattr(cp_sat_module.CFG, "BACKTRACK_PROBE_FIRST", True, raising=False)

    def fake_build_options(W, H, tiles, stride, *, rng=None, randomize=False):
        return [((0, 0, 0, min(W, t.w), min(H, t.h)),) for t in tiles]

    monkeypatch.setattr(cp_sat_module, "build_options", fake_build_options)

    def forbid_backtracking(*args, **kwargs):  # pragma: no cover - sanity guard
        raise AssertionError("backtracking should not run when guards are active")

    monkeypatch.setattr(cp_sat_module, "_backtracking_exact_cover", forbid_backtracking)

    calls = {}

    def fake_resolver(attempt, *, edge_guard_cells, plus_guard_enabled):
        calls["edge_guard"] = edge_guard_cells
        calls["plus_guard"] = plus_guard_enabled
        return False, [], "Simulated CP-SAT unavailable", edge_guard_cells, plus_guard_enabled

    monkeypatch.setattr(cp_sat_module, "_resolve_guard_backoffs", fake_resolver)

    tiles = []
    for (w_ft, h_ft), count in CRASH_DEMAND_BAG_FT.items():
        w = ft_to_cells(w_ft)
        h = ft_to_cells(h_ft)
        for idx in range(count):
            tiles.append(Rect(w, h, f"{w_ft}x{h_ft}_{idx}"))

    board = ft_to_cells(10.5)

    ok, placed, reason = cp_sat_module.try_pack_exact_cover(board, board, tiles, allow_discard=False, max_seconds=1)

    assert not ok
    assert placed == []
    assert reason == "Simulated CP-SAT unavailable"
    assert calls["edge_guard"] == ft_to_cells(6.0)
    assert calls["plus_guard"] is False

    meta = getattr(cp_sat_module.try_pack_exact_cover, "last_meta", {})
    assert meta.get("tiles_requested") == sum(CRASH_DEMAND_BAG_FT.values())
    assert meta.get("backtracking_prefilter") is False
    assert meta.get("backtracking_prefilter_blocked") == "guards"

def test_backtracking_exact_cover_solves_small_board(cp_sat_module):
    Rect = cp_sat_module.Rect
    tiles = [
        Rect(2, 2, "A"),
        Rect(2, 2, "B"),
        Rect(2, 2, "C"),
        Rect(2, 2, "D"),
    ]
    options = cp_sat_module.build_options(4, 4, tiles, stride=1, randomize=False)

    placements = cp_sat_module._backtracking_exact_cover(4, 4, tiles, options)

    assert placements is not None
    assert len(placements) == len(tiles)
    covered = {(p.x + dx, p.y + dy) for p in placements for dx in range(p.rect.w) for dy in range(p.rect.h)}
    assert len(covered) == 16


def test_backtracking_handles_medium_board(cp_sat_module):
    Rect = cp_sat_module.Rect
    tiles = [
        Rect(20, 10, "A"),
        Rect(20, 10, "B"),
    ]
    options = cp_sat_module.build_options(20, 20, tiles, stride=1, randomize=False)

    placements = cp_sat_module._backtracking_exact_cover(20, 20, tiles, options)

    assert placements is not None
    assert len(placements) == len(tiles)
    covered = {
        (p.x + dx, p.y + dy)
        for p in placements
        for dx in range(p.rect.w)
        for dy in range(p.rect.h)
    }
    assert len(covered) == 400


def test_guard_blocks_backtracking_prefilter(monkeypatch, cp_sat_module):
    Rect = cp_sat_module.Rect
    tiles = [Rect(1, 1, "A")]

    monkeypatch.setattr(cp_sat_module.CFG, "SAME_SHAPE_LIMIT", 1, raising=False)
    monkeypatch.setattr(cp_sat_module.CFG, "MAX_EDGE_FT", 1.0, raising=False)
    monkeypatch.setattr(cp_sat_module.CFG, "NO_PLUS", True, raising=False)
    monkeypatch.setattr(cp_sat_module.CFG, "TEST_MODE", False, raising=False)
    monkeypatch.setattr(cp_sat_module.CFG, "BACKTRACK_PROBE_FIRST", True, raising=False)

    def fake_backtracking(*args, **kwargs):  # pragma: no cover - ensures guard skip
        raise AssertionError("backtracking should not run when guards are active")

    placements = [cp_sat_module.Placed(0, 0, Rect(1, 1, "A"))]

    def fake_resolver(attempt, *, edge_guard_cells, plus_guard_enabled):
        return True, placements, None, edge_guard_cells, plus_guard_enabled

    monkeypatch.setattr(cp_sat_module, "_backtracking_exact_cover", fake_backtracking)
    monkeypatch.setattr(cp_sat_module, "_resolve_guard_backoffs", fake_resolver)

    ok, placed, reason = cp_sat_module.try_pack_exact_cover(3, 3, tiles)

    assert ok
    assert reason is None
    assert placed == placements

    meta = getattr(cp_sat_module.try_pack_exact_cover, "last_meta", {})
    assert meta.get("backtracking_prefilter") is False
    assert meta.get("backtracking_prefilter_blocked") == "guards"

def test_force_backtracking_solves_board(monkeypatch, cp_sat_module):
    Rect = cp_sat_module.Rect
    tiles = [
        Rect(2, 2, "A"),
        Rect(2, 2, "B"),
        Rect(2, 2, "C"),
        Rect(2, 2, "D"),
    ]

    monkeypatch.setattr(cp_sat_module.CFG, "SAME_SHAPE_LIMIT", -1, raising=False)
    monkeypatch.setattr(cp_sat_module.CFG, "MAX_EDGE_FT", None, raising=False)
    monkeypatch.setattr(cp_sat_module.CFG, "NO_PLUS", False, raising=False)

    ok, placed, reason = cp_sat_module.try_pack_exact_cover(
        4,
        4,
        tiles,
        force_backtracking=True,
    )

    assert ok
    assert reason is None
    assert len(placed) == len(tiles)
    meta = getattr(cp_sat_module.try_pack_exact_cover, "last_meta", {})
    solved_via = meta.get("solved_via")
    assert solved_via in {"backtracking_forced", "backtracking_prefilter"}
    cp_sat_meta = meta.get("cp_sat") if isinstance(meta, dict) else None
    assert isinstance(cp_sat_meta, dict) and cp_sat_meta.get("skipped") is True


def test_force_backtracking_requires_strict_mode(cp_sat_module):
    Rect = cp_sat_module.Rect
    tiles = [Rect(2, 2, "A"), Rect(2, 2, "B")]

    ok, placed, reason = cp_sat_module.try_pack_exact_cover(
        4,
        4,
        tiles,
        allow_discard=True,
        force_backtracking=True,
    )

    assert not ok
    assert placed == []
    assert "Backtracking rescue" in (reason or "")


def test_force_backtracking_handles_crash_demand(monkeypatch, cp_sat_module):
    from tests.data import CRASH_DEMAND_BAG_FT
    from models import ft_to_cells

    monkeypatch.setattr(cp_sat_module.CFG, "SAME_SHAPE_LIMIT", -1, raising=False)
    monkeypatch.setattr(cp_sat_module.CFG, "MAX_EDGE_FT", None, raising=False)
    monkeypatch.setattr(cp_sat_module.CFG, "NO_PLUS", False, raising=False)

    W = ft_to_cells(10.5)
    H = ft_to_cells(10.5)
    bag_cells = {
        (ft_to_cells(w_ft), ft_to_cells(h_ft)): count
        for (w_ft, h_ft), count in CRASH_DEMAND_BAG_FT.items()
    }

    ok, placed, reason = cp_sat_module.try_pack_exact_cover(
        W,
        H,
        bag_cells,
        force_backtracking=True,
    )

    assert ok, reason
    assert placed
    assert len(placed) == sum(bag_cells.values())


def test_backtracking_respects_limits(monkeypatch, cp_sat_module):
    Rect = cp_sat_module.Rect
    tiles = [Rect(1, 1, f"t{i}") for i in range(10)]
    options = cp_sat_module.build_options(4, 4, tiles, stride=1, randomize=False)

    monkeypatch.setattr(cp_sat_module.CFG, "BACKTRACK_MAX_TILES", 4, raising=False)

    assert cp_sat_module._backtracking_exact_cover(4, 4, tiles, options) is None
