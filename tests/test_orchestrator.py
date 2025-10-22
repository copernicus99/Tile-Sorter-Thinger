import sys
import types

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
    _coerce_bag_ft,
    _phase_c_candidates,
    _phase_d_candidates,
    _mirrored_probe_order,
    _should_retry_phase,
)

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
    base_side = 20
    grid_step = 2

    candidates = _phase_c_candidates(base_side, grid_step=grid_step)

    assert len(candidates) == 1
    cb = candidates[0]
    assert cb.W == cb.H == 20
    assert "10.0 × 10.0 ft" in cb.label


def test_phase_d_candidates_prioritize_base_and_neighbors():
    shrink_floor = 12
    base_side = 20
    grid_step = 2
    area_cells = 24 * 24

    candidates = _phase_d_candidates(
        shrink_floor,
        base_side,
        grid_step=grid_step,
        area_cells=area_cells,
    )

    sides = [cb.W for cb in candidates]
    assert sides[:5] == [20, 22, 18, 24, 16]
    assert sides[-1] == 12


def test_phase_d_candidates_honor_minimum_board_size():
    shrink_floor = 22
    base_side = 20
    grid_step = 2
    area_cells = 0

    candidates = _phase_d_candidates(
        shrink_floor,
        base_side,
        grid_step=grid_step,
        area_cells=area_cells,
    )

    sides = [cb.W for cb in candidates]
    assert sides[0] == 22
    assert all(side >= 22 for side in sides)


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


def test_orchestrator_phase_d_sticks_to_base_board(monkeypatch):
    import math
    import solver.orchestrator as orchestrator
    from models import ft_to_cells

    calls = []

    def fake_run_cp_sat_isolated(W, H, bag, seconds, allow_discard):
        calls.append((W, H, allow_discard))
        return False, [], "no solution"

    monkeypatch.setattr(orchestrator, "_run_cp_sat_isolated", fake_run_cp_sat_isolated)

    monkeypatch.setattr(orchestrator.CFG, "TIME_C", 1)
    monkeypatch.setattr(orchestrator.CFG, "TIME_D", 1)
    monkeypatch.setattr(orchestrator.CFG, "TIME_E", 0)
    monkeypatch.setattr(orchestrator.CFG, "TIME_F", 0)

    bag_ft = {(1.0, 1.0): 100}

    orchestrator.solve_orchestrator(bag_ft=bag_ft)

    bag_cells = orchestrator._bag_ft_to_cells(bag_ft)
    grid_step = orchestrator._grid_step_from_bag(bag_cells)
    max_tile_side = max((max(w, h) for (w, h) in bag_cells.keys()), default=6)
    base_side_cells = max(6, ft_to_cells(math.sqrt(orchestrator.CFG.BASE_GRID_AREA_SQFT)))
    base_side_cells = max(base_side_cells, max_tile_side)
    base_side_cells = orchestrator._align_up_to_multiple(base_side_cells, grid_step)
    min_board = orchestrator._align_up_to_multiple(max_tile_side, grid_step)
    if min_board > base_side_cells:
        base_side_cells = min_board
    area_cells = sum((w * h) * c for (w, h), c in bag_cells.items())
    needed_side = orchestrator._align_up_to_multiple(orchestrator._ceil_sqrt_cells(area_cells), grid_step)
    if area_cells <= 0:
        expand_ceiling = base_side_cells
    else:
        expand_ceiling = max(base_side_cells + grid_step, needed_side)

    d_calls = [(W, H) for (W, H, allow) in calls if allow]
    assert d_calls
    assert d_calls[0] == (base_side_cells, base_side_cells)
    assert all(min_board <= W <= expand_ceiling for (W, _H) in d_calls)


def test_orchestrator_retries_timebox_attempts(monkeypatch):
    import solver.orchestrator as orchestrator

    calls = []

    def fake_run_cp_sat_isolated(W, H, bag, seconds, allow_discard):
        calls.append(seconds)
        return False, [], "Stopped before solution (timebox)"

    monkeypatch.setattr(orchestrator, "_run_cp_sat_isolated", fake_run_cp_sat_isolated)

    monkeypatch.setattr(orchestrator.CFG, "TIME_C", 30)
    monkeypatch.setattr(orchestrator.CFG, "TIME_D", 0)
    monkeypatch.setattr(orchestrator.CFG, "TIME_E", 0)
    monkeypatch.setattr(orchestrator.CFG, "TIME_F", 0)
    monkeypatch.setattr(orchestrator.CFG, "PHASE_RETRY_LIMIT", 2, raising=False)

    bag_ft = {(1.0, 1.0): 100}

    orchestrator.solve_orchestrator(bag_ft=bag_ft)

    assert len(calls) == 3

def test_orchestrator_expands_board_to_cover_tall_tiles(monkeypatch):
    import solver.orchestrator as orchestrator
    from models import ft_to_cells

    calls = []

    def fake_run_cp_sat_isolated(W, H, bag, seconds, allow_discard):
        calls.append((W, H, allow_discard))
        max_h = max(h for _w, h in bag.keys())
        if H < max_h:
            return False, [], "grid too small"

        from models import Rect, Placed

        placed = []
        for idx, ((w, h), cnt) in enumerate(bag.items()):
            rect = Rect(w, h, f"tile-{idx}")
            for _ in range(cnt):
                placed.append(Placed(0, 0, rect))

        return True, placed, None

    monkeypatch.setattr(orchestrator, "_run_cp_sat_isolated", fake_run_cp_sat_isolated)

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
