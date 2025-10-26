import sys
import time
import types
from typing import List, Optional, Tuple

import pytest

# Provide a lightweight stub for ortools so importing the orchestrator works in
# environments where the optional dependency is unavailable (e.g. CI for unit
# tests that focus on parsing helpers).
if "ortools" not in sys.modules:
    ortools_mod = types.ModuleType("ortools")
    sat_mod = types.ModuleType("ortools.sat")
    python_mod = types.ModuleType("ortools.sat.python")
    cp_model_mod = types.ModuleType("ortools.sat.python.cp_model")

    class _MissingCp:
        OPTIMAL = 4
        FEASIBLE = 3
        INFEASIBLE = 2
        MODEL_INVALID = 1

        class CpModel:
            def __init__(self, *args, **kwargs):  # pragma: no cover - defensive
                raise RuntimeError("ortools is not installed")

        class CpSolver:
            def __init__(self, *args, **kwargs):  # pragma: no cover - defensive
                raise RuntimeError("ortools is not installed")

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

from solver.orchestrator import (
    CandidateBoard,
    _PHASE_D_MAX_CANDIDATES,
    _align_up_to_multiple,
    _bag_ft_to_cells,
    _coerce_bag_ft,
    _fmt_ft,
    _grid_step_from_bag,
    _phase_c_candidates,
    _phase_d_candidates,
    _mirrored_probe_order,
    _should_retry_phase,
)
from models import ft_to_cells
from tests.data import CRASH_DEMAND_BAG_FT

# Remove the temporary stubs so other tests that rely on pytest.importorskip
# still see the absence of the optional dependency and skip accordingly.
sys.modules.pop("solver.cp_sat", None)
sys.modules.pop("ortools.sat.python.cp_model", None)
sys.modules.pop("ortools.sat.python", None)
sys.modules.pop("ortools.sat", None)
sys.modules.pop("ortools", None)


def test_coerce_bag_ft_from_tiles_form_list():
    form_payload = {
        "tiles": [
            {"w": 1, "h": 2, "count": 4},
            {"w": "1", "h": "2", "count": "3"},
            {"w": 2, "h": 2, "count": 1},
        ]
    }

    bag = _coerce_bag_ft(form_payload)

    assert bag == {(1.0, 2.0): 7, (2.0, 2.0): 1}


def test_coerce_bag_ft_empty_when_parse_fails():
    # No tiles -> parse_demand would report nothing parsed
    assert _coerce_bag_ft({}) == {}


def test_phase_c_candidates_returns_only_base_board():
    base_side = 26
    grid_step = 2

    candidates = _phase_c_candidates(base_side, grid_step=grid_step)

    assert len(candidates) == 1
    cb = candidates[0]
    expected_side = _align_up_to_multiple(base_side, grid_step)
    expected_side = max(6, expected_side)
    assert cb.W == cb.H == expected_side
    expected_label = f"{_fmt_ft(expected_side)} × {_fmt_ft(expected_side)} ft"
    assert cb.label == expected_label


def test_phase_d_candidates_walk_nearby_grids():
    base_side = ft_to_cells(10.0)
    shrink_floor = max(6, base_side - 4)
    area_cells = (base_side + 2) * (base_side + 1)

    candidates = _phase_d_candidates(
        shrink_floor=shrink_floor,
        base_side=base_side,
        grid_step=1,
        area_cells=area_cells,
    )

    dims = [(cb.W, cb.H) for cb in candidates]

    assert dims
    assert dims[0][0] * dims[0][1] >= area_cells
    assert len(dims) <= _PHASE_D_MAX_CANDIDATES
    assert any(w > base_side or h > base_side for w, h in dims)
    assert all(w * h >= area_cells for w, h in dims)
    assert all(shrink_floor <= min(w, h) for w, h in dims)
    assert all(max(w, h) <= base_side + 4 for w, h in dims)


def test_mirrored_probe_order_handles_duplicates_gracefully():
    order = _mirrored_probe_order([6, 8, 8, 10])
    # duplicates should not break the alternating pattern
    assert order == [10, 6, 8, 8]


def test_should_retry_phase_reason_tokens():
    assert _should_retry_phase("Stopped before solution (timebox)")
    assert _should_retry_phase("Model capped: placements limit hit")
    assert not _should_retry_phase("Proven infeasible under current constraints")
    assert not _should_retry_phase("no solution")
    assert not _should_retry_phase(None)


def test_orchestrator_phase_d_sticks_to_10x10(monkeypatch):
    import solver.orchestrator as orchestrator

    calls = []

    def fake_run_cp_sat_isolated(W, H, bag, seconds, allow_discard, *, hint=None):
        calls.append((W, H, allow_discard, seconds))
        return False, [], "no solution", {}

    monkeypatch.setattr(orchestrator, "_run_cp_sat_isolated", fake_run_cp_sat_isolated)

    monkeypatch.setattr(orchestrator.CFG, "TIME_C", 1)
    monkeypatch.setattr(orchestrator.CFG, "TIME_D", 1)
    monkeypatch.setattr(orchestrator.CFG, "TIME_E", 1)
    monkeypatch.setattr(orchestrator.CFG, "TIME_F", 0)

    bag_ft = {(1.0, 1.0): 100}

    orchestrator.solve_orchestrator(bag_ft=bag_ft)

    ten_cells = ft_to_cells(10.0)
    assert calls
    first_W, first_H, first_allow, _first_seconds = calls[0]
    assert first_W == first_H == ten_cells
    assert not first_allow
    # The Phase D attempt should use the entire configured budget on a 10×10 grid
    d_calls = [(W, H, seconds) for (W, H, allow, seconds) in calls if allow]
    assert d_calls
    _first_d_W, _first_d_H, first_d_seconds = d_calls[0]
    assert first_d_seconds == pytest.approx(orchestrator.CFG.TIME_D, rel=0.05)

    bag_cells = _bag_ft_to_cells(bag_ft)
    grid_step = _grid_step_from_bag(bag_cells)
    max_tile_side = max(max(abs(int(w)), abs(int(h))) for (w, h) in bag_cells.keys())
    shrink_floor = _align_up_to_multiple(max(6, max_tile_side), grid_step)
    if shrink_floor > ten_cells:
        shrink_floor = ten_cells
    area_cells = sum((w * h) * c for (w, h), c in bag_cells.items())
    expected_dims = {
        (cb.W, cb.H)
        for cb in _phase_d_candidates(
            shrink_floor=shrink_floor,
            base_side=ten_cells,
            grid_step=grid_step,
            area_cells=area_cells,
        )
    }
    assert expected_dims

    for W, H, seconds in d_calls:
        assert (W, H) in expected_dims
        assert seconds <= orchestrator.CFG.TIME_D + 1e-6


def test_orchestrator_phase_d_does_not_requeue_without_progress(monkeypatch):
    import solver.orchestrator as orchestrator

    calls = []

    def fake_run_cp_sat_isolated(W, H, bag, seconds, allow_discard, *, hint=None):
        calls.append((W, H, allow_discard, seconds))
        return False, [], "Stopped before solution (timebox)", {}

    phase_d_candidates = [
        CandidateBoard(20, 20, "20 × 20 ft"),
        CandidateBoard(20, 18, "20 × 18 ft"),
        CandidateBoard(18, 20, "18 × 20 ft"),
    ]

    def fake_phase_d_candidates(*args, **kwargs):
        return list(phase_d_candidates)

    monkeypatch.setattr(orchestrator, "_run_cp_sat_isolated", fake_run_cp_sat_isolated)
    monkeypatch.setattr(orchestrator, "_phase_d_candidates", fake_phase_d_candidates)
    monkeypatch.setattr(orchestrator.CFG, "TIME_C", 1)
    monkeypatch.setattr(orchestrator.CFG, "TIME_D", 60)
    monkeypatch.setattr(orchestrator.CFG, "TIME_E", 0)
    monkeypatch.setattr(orchestrator.CFG, "TIME_F", 0)

    bag_ft = {(2.0, 2.0): 30, (2.0, 3.0): 10}

    orchestrator.solve_orchestrator(bag_ft=bag_ft)

    d_calls = [(W, H, seconds) for (W, H, allow, seconds) in calls if allow]
    assert len(d_calls) == len(phase_d_candidates)
    assert all((W, H) in {(cb.W, cb.H) for cb in phase_d_candidates} for (W, H, _s) in d_calls)


def test_orchestrator_phase_c_consumes_full_timebox(monkeypatch):
    import solver.orchestrator as orchestrator

    budgets = []

    def fake_run_cp_sat_isolated(W, H, bag, seconds, allow_discard, *, hint=None):
        budgets.append((W, H, seconds))
        return False, [], "no solution", {}

    monkeypatch.setattr(orchestrator, "_run_cp_sat_isolated", fake_run_cp_sat_isolated)

    monkeypatch.setattr(orchestrator.CFG, "TIME_C", 1)
    monkeypatch.setattr(orchestrator.CFG, "TIME_D", 0)
    monkeypatch.setattr(orchestrator.CFG, "TIME_E", 1)
    monkeypatch.setattr(orchestrator.CFG, "TIME_F", 0)

    bag_ft = {(1.0, 1.0): 100}

    start = time.time()
    orchestrator.solve_orchestrator(bag_ft=bag_ft)
    elapsed = time.time() - start

    assert budgets
    ten_cells = ft_to_cells(10.0)
    W, H, seconds = budgets[0]
    assert W == H == ten_cells
    assert seconds == pytest.approx(orchestrator.CFG.TIME_C, rel=0.01)
    assert elapsed >= orchestrator.CFG.TIME_C

def test_orchestrator_expands_board_to_cover_tall_tiles(monkeypatch):
    import solver.orchestrator as orchestrator
    from models import ft_to_cells

    calls = []

    def fake_run_cp_sat_isolated(W, H, bag, seconds, allow_discard, *, hint=None):
        calls.append((W, H, allow_discard))
        max_h = max(h for _w, h in bag.keys())
        if H < max_h:
            return False, [], "grid too small", {}

        from models import Rect, Placed

        placed = []
        for idx, ((w, h), cnt) in enumerate(bag.items()):
            rect = Rect(w, h, f"tile-{idx}")
            for _ in range(cnt):
                placed.append(Placed(0, 0, rect))

        return True, placed, None, {}

    monkeypatch.setattr(orchestrator, "_run_cp_sat_isolated", fake_run_cp_sat_isolated)

    monkeypatch.setattr(orchestrator.CFG, "TIME_C", 1)
    monkeypatch.setattr(orchestrator.CFG, "TIME_D", 1)
    monkeypatch.setattr(orchestrator.CFG, "TIME_E", 1)
    monkeypatch.setattr(orchestrator.CFG, "TIME_F", 0)

    bag_ft = {
        (1.0, 25.0): 1,
        (1.0, 15.0): 1,
        (2.0, 5.0): 2,
        (1.0, 2.0): 20,
    }

    ok, placed, W_ft, H_ft, strategy, reason, meta = orchestrator.solve_orchestrator(bag_ft=bag_ft)

    assert ok, reason
    assert placed

    required_height = max(ft_to_cells(h_ft) for _w_ft, h_ft in bag_ft.keys())
    assert any(H >= required_height for _W, H, _discard in calls)
    assert H_ft >= max(h_ft for _w_ft, h_ft in bag_ft.keys())

def test_phase_e_pipe_crash_triggers_rescue(monkeypatch):
    import solver.orchestrator as orchestrator
    from models import Rect, Placed

    candidate = CandidateBoard(22, 22, "22 × 22 ft")

    def fake_square_candidates(min_side, max_side, *, descending, multiple_of=1):
        return [candidate]

    call_state = {"count": 0}

    def fake_run_cp_sat_isolated(W, H, bag, seconds, allow_discard, *, hint=None):
        call_state["count"] += 1
        return (
            False,
            [],
            "Subprocess ended early (pipe closed)",
            {
                "isolation": {
                    "exitcode": -11,
                    "payload_received": False,
                    "payload_kind": None,
                    "payload_raw_type": None,
                    "placement_count": 0,
                    "stderr_tail": "Segmentation fault",
                }
            },
        )

    rescue_calls = []

    def fake_run_backtracking_rescue(W, H, bag, *, hint=None, seconds=None):
        rescue_calls.append((W, H, seconds))
        rect = Rect(W, H, "rescue")
        placed = [Placed(0, 0, rect)]
        return True, placed, "Deterministic rescue", {"solved_via": "rescue"}

    monkeypatch.setattr(orchestrator, "_square_candidates", fake_square_candidates)
    monkeypatch.setattr(orchestrator, "_run_cp_sat_isolated", fake_run_cp_sat_isolated)
    monkeypatch.setattr(orchestrator, "_run_backtracking_rescue", fake_run_backtracking_rescue)

    monkeypatch.setattr(orchestrator.CFG, "TIME_C", 0)
    monkeypatch.setattr(orchestrator.CFG, "TIME_D", 0)
    monkeypatch.setattr(orchestrator.CFG, "TIME_F", 0)
    monkeypatch.setattr(orchestrator.CFG, "TIME_E", 12)
    monkeypatch.setattr(orchestrator.CFG, "PHASE_RETRY_LIMIT", 1, raising=False)

    ok, placed, W_ft, H_ft, strategy, reason, meta = orchestrator.solve_orchestrator(
        bag_ft=CRASH_DEMAND_BAG_FT
    )

    assert call_state["count"] == 1
    assert rescue_calls, "Deterministic rescue was not invoked"
    assert ok
    assert strategy == "E"
    assert placed


def test_phase_e_crash_with_partial_results_discards_and_rescues(monkeypatch):
    import solver.orchestrator as orchestrator
    from models import Rect, Placed

    candidate = CandidateBoard(24, 24, "24 × 24 ft")

    def fake_square_candidates(min_side, max_side, *, descending, multiple_of=1):
        return [candidate]

    call_state = {"count": 0}

    def fake_run_cp_sat_isolated(W, H, bag, seconds, allow_discard, *, hint=None):
        call_state["count"] += 1
        rect = Rect(2, 2, "partial")
        placed = [Placed(0, 0, rect)]
        return (
            False,
            placed,
            "Subprocess ended early (pipe closed)",
            {
                "isolation": {
                    "exitcode": -11,
                    "payload_received": True,
                    "payload_kind": "result",
                    "payload_raw_type": "tuple",
                    "placement_count": len(placed),
                    "stderr_tail": "Segmentation fault",
                }
            },
        )

    rescue_calls = []

    def fake_run_backtracking_rescue(W, H, bag, *, hint=None, seconds=None):
        rescue_calls.append((W, H, seconds, hint))
        rect = Rect(W, H, "rescue")
        placed = [Placed(0, 0, rect)]
        return True, placed, "Deterministic rescue", {"solved_via": "rescue"}

    monkeypatch.setattr(orchestrator, "_square_candidates", fake_square_candidates)
    monkeypatch.setattr(orchestrator, "_run_cp_sat_isolated", fake_run_cp_sat_isolated)
    monkeypatch.setattr(orchestrator, "_run_backtracking_rescue", fake_run_backtracking_rescue)

    monkeypatch.setattr(orchestrator.CFG, "TIME_C", 0)
    monkeypatch.setattr(orchestrator.CFG, "TIME_D", 0)
    monkeypatch.setattr(orchestrator.CFG, "TIME_F", 0)
    monkeypatch.setattr(orchestrator.CFG, "TIME_E", 12)
    monkeypatch.setattr(orchestrator.CFG, "PHASE_RETRY_LIMIT", 3, raising=False)

    ok, placed, W_ft, H_ft, strategy, reason, meta = orchestrator.solve_orchestrator(
        bag_ft=CRASH_DEMAND_BAG_FT
    )

    assert call_state["count"] == 1
    assert rescue_calls, "Deterministic rescue was not invoked"
    W, H, seconds, hint = rescue_calls[0]
    assert hint is None
    assert ok
    assert strategy == "E"
    assert placed


def test_phase_f_crash_triggers_rescue_even_with_allow_discard(monkeypatch):
    import solver.orchestrator as orchestrator
    from models import Rect, Placed

    candidate = CandidateBoard(26, 26, "26 × 26 ft")

    def fake_square_candidates(min_side, max_side, *, descending, multiple_of=1):
        return [candidate]

    call_state = {"count": 0, "allow_flags": []}

    def fake_run_cp_sat_isolated(W, H, bag, seconds, allow_discard, *, hint=None):
        call_state["count"] += 1
        call_state["allow_flags"].append(allow_discard)
        return (
            False,
            [],
            "Subprocess ended early (pipe closed)",
            {
                "isolation": {
                    "exitcode": -11,
                    "payload_received": False,
                    "payload_kind": None,
                    "payload_raw_type": None,
                    "placement_count": 0,
                    "stderr_tail": "Segmentation fault",
                }
            },
        )

    rescue_calls = []

    def fake_run_backtracking_rescue(W, H, bag, *, hint=None, seconds=None):
        rescue_calls.append((W, H, seconds, hint))
        rect = Rect(W, H, "rescue")
        placed = [Placed(0, 0, rect)]
        return True, placed, "Deterministic rescue", {"solved_via": "rescue"}

    monkeypatch.setattr(orchestrator, "_square_candidates", fake_square_candidates)
    monkeypatch.setattr(orchestrator, "_run_cp_sat_isolated", fake_run_cp_sat_isolated)
    monkeypatch.setattr(orchestrator, "_run_backtracking_rescue", fake_run_backtracking_rescue)

    monkeypatch.setattr(orchestrator.CFG, "TIME_C", 0)
    monkeypatch.setattr(orchestrator.CFG, "TIME_D", 0)
    monkeypatch.setattr(orchestrator.CFG, "TIME_E", 0)
    monkeypatch.setattr(orchestrator.CFG, "TIME_F", 12)
    monkeypatch.setattr(orchestrator.CFG, "PHASE_RETRY_LIMIT", 1, raising=False)

    ok, placed, W_ft, H_ft, strategy, reason, meta = orchestrator.solve_orchestrator(
        bag_ft=CRASH_DEMAND_BAG_FT
    )

    assert call_state["count"] == 1
    assert call_state["allow_flags"] == [True]
    assert rescue_calls, "Deterministic rescue was not invoked"
    W, H, seconds, hint = rescue_calls[0]
    assert W == candidate.W and H == candidate.H
    assert hint is None
    assert ok
    assert strategy == "F"
    assert placed


def test_phase_d_crash_skips_rescue_when_board_too_small(monkeypatch):
    import solver.orchestrator as orchestrator

    candidate = CandidateBoard(10, 10, "10 × 10 ft")

    def fake_phase_d_candidates(*args, **kwargs):
        return [candidate]

    call_state = {"count": 0}

    def fake_run_cp_sat_isolated(W, H, bag, seconds, allow_discard, *, hint=None):
        call_state["count"] += 1
        return (
            False,
            [],
            "Subprocess ended early (pipe closed)",
            {
                "isolation": {
                    "exitcode": -11,
                    "payload_received": False,
                    "payload_kind": None,
                    "payload_raw_type": None,
                    "placement_count": 0,
                    "stderr_tail": "Segmentation fault",
                }
            },
        )

    rescue_calls: List[Tuple[int, int, Optional[float], Optional[object]]] = []

    def fake_run_backtracking_rescue(W, H, bag, *, hint=None, seconds=None):
        rescue_calls.append((W, H, seconds, hint))
        return False, [], "Deterministic rescue skipped", {"solved_via": "rescue"}

    monkeypatch.setattr(orchestrator, "_phase_d_candidates", fake_phase_d_candidates)
    monkeypatch.setattr(orchestrator, "_run_cp_sat_isolated", fake_run_cp_sat_isolated)
    monkeypatch.setattr(orchestrator, "_run_backtracking_rescue", fake_run_backtracking_rescue)

    monkeypatch.setattr(orchestrator.CFG, "TIME_C", 0)
    monkeypatch.setattr(orchestrator.CFG, "TIME_D", 0.1)
    monkeypatch.setattr(orchestrator.CFG, "TIME_E", 0)
    monkeypatch.setattr(orchestrator.CFG, "TIME_F", 0)
    monkeypatch.setattr(orchestrator.CFG, "BASE_GRID_AREA_SQFT", 1, raising=False)
    monkeypatch.setattr(orchestrator.CFG, "PHASE_RETRY_LIMIT", 1, raising=False)

    bag_ft = {
        (5.0, 5.0): 3,  # total demand area is 300 cells; board area is only 100 cells
    }

    ok, placed, W_ft, H_ft, strategy, reason, meta = orchestrator.solve_orchestrator(
        bag_ft=bag_ft
    )

    assert call_state["count"] == 1
    assert not rescue_calls, "Deterministic rescue should be skipped when board is undersized"
    assert not ok
    assert strategy in (None, "error")
    assert not placed
