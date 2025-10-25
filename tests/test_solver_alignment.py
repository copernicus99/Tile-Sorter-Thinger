import pytest

cp_sat = pytest.importorskip("solver.cp_sat")
orchestrator = pytest.importorskip("solver.orchestrator")

try_pack_exact_cover = cp_sat.try_pack_exact_cover
_align_up_to_multiple = orchestrator._align_up_to_multiple
_grid_step_from_bag = orchestrator._grid_step_from_bag
_square_candidates = orchestrator._square_candidates


def test_try_pack_exact_cover_respects_tile_stride():
    # Require placements at 2-cell increments; coarse stride (10) would fail.
    multiset = {(2, 2): 8}
    ok, placed, reason = try_pack_exact_cover(
        W=8,
        H=4,
        multiset=multiset,
        allow_discard=False,
        max_seconds=5.0,
    )
    assert ok, reason
    assert len(placed) == 8


def test_square_candidates_align_to_tile_gcd():
    bag_cells = {
        (30, 30): 6,
        (30, 10): 6,
        (30, 6): 6,
        (30, 4): 6,
        (10, 4): 6,
        (4, 4): 6,
        (2, 4): 6,
    }
    step = _grid_step_from_bag(bag_cells)
    assert step == 2
    aligned = _align_up_to_multiple(97, step)
    assert aligned == 98
    candidates = _square_candidates(64, aligned, descending=False, multiple_of=step)
    widths = [cb.W for cb in candidates]
    assert aligned in widths
    assert all(w % step == 0 for w in widths)


def test_orchestrator_surfaces_solver_reason(monkeypatch):
    cp_sat = pytest.importorskip("solver.cp_sat")
    orchestrator = pytest.importorskip("solver.orchestrator")

    monkeypatch.setattr(cp_sat.CFG, "MAX_EDGE_FT", 0.5, raising=False)
    monkeypatch.setattr(orchestrator.CFG, "MAX_EDGE_FT", 0.5, raising=False)

    ok, placed, W_ft, H_ft, strategy, reason, meta = orchestrator.solve_orchestrator(
        bag_ft={(1.0, 1.0): 4}
    )

    assert not ok
    assert reason == "Proven infeasible under current constraints"
    assert meta.get("reason") == reason


def test_seam_guard_ignores_empty_runs(monkeypatch):
    cp_sat = pytest.importorskip("solver.cp_sat")

    monkeypatch.setattr(cp_sat.CFG, "MAX_EDGE_FT", 1.0, raising=False)

    ok, placed, reason = try_pack_exact_cover(
        W=4,
        H=6,
        multiset={(2, 2): 2},
        allow_discard=False,
        max_seconds=5.0,
    )

    assert ok, reason
    assert len(placed) == 2


def test_backtracking_rescue_disabled_under_guards(monkeypatch):
    orchestrator = pytest.importorskip("solver.orchestrator")

    monkeypatch.setattr(orchestrator.CFG, "TEST_MODE", False, raising=False)
    monkeypatch.setattr(orchestrator.CFG, "MAX_EDGE_FT", 6.0, raising=False)
    monkeypatch.setattr(orchestrator.CFG, "NO_PLUS", True, raising=False)
    monkeypatch.setattr(orchestrator.CFG, "SAME_SHAPE_LIMIT", 1, raising=False)

    ok, placed, reason, meta = orchestrator._run_backtracking_rescue(
        4,
        4,
        {(2, 2): 4},
        seconds=1.0,
    )

    assert not ok
    assert placed == []
    assert "CP-SAT" in reason
    assert isinstance(meta, dict)
    assert meta.get("error") == "backtracking_guard_blocked"
    assert meta.get("guards")
