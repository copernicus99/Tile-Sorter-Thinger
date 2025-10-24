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


def test_force_backtracking_solves_board(cp_sat_module):
    Rect = cp_sat_module.Rect
    tiles = [
        Rect(2, 2, "A"),
        Rect(2, 2, "B"),
        Rect(2, 2, "C"),
        Rect(2, 2, "D"),
    ]

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


def test_backtracking_respects_limits(monkeypatch, cp_sat_module):
    Rect = cp_sat_module.Rect
    tiles = [Rect(1, 1, f"t{i}") for i in range(10)]
    options = cp_sat_module.build_options(4, 4, tiles, stride=1, randomize=False)

    monkeypatch.setattr(cp_sat_module.CFG, "BACKTRACK_MAX_TILES", 4, raising=False)

    assert cp_sat_module._backtracking_exact_cover(4, 4, tiles, options) is None
