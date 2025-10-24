import importlib
import sys
import types


def _ensure_ortools_stub() -> None:
    if "ortools" in sys.modules:
        return

    ortools_mod = types.ModuleType("ortools")
    sat_mod = types.ModuleType("ortools.sat")
    python_mod = types.ModuleType("ortools.sat.python")
    cp_model_mod = types.ModuleType("ortools.sat.python.cp_model")

    class _DummyModel:  # pragma: no cover - defensive fallback
        pass

    cp_model_mod.CpModel = _DummyModel
    cp_model_mod.CpSolver = _DummyModel
    cp_model_mod.OPTIMAL = 4
    cp_model_mod.FEASIBLE = 3
    cp_model_mod.INFEASIBLE = 2
    cp_model_mod.MODEL_INVALID = 1

    ortools_mod.sat = sat_mod
    sat_mod.python = python_mod
    python_mod.cp_model = cp_model_mod

    sys.modules["ortools"] = ortools_mod
    sys.modules["ortools.sat"] = sat_mod
    sys.modules["ortools.sat.python"] = python_mod
    sys.modules["ortools.sat.python.cp_model"] = cp_model_mod


_ensure_ortools_stub()
cp_sat = importlib.import_module("solver.cp_sat")


def test_edge_guard_relaxed_when_tile_exceeds_limit():
    guard = cp_sat._compute_edge_guard_cells(
        6.0,
        cell_size=0.5,
        board_max=24,
        test_mode=False,
        max_tile_side_ft=9.0,
    )
    assert guard is None


def test_edge_guard_kept_when_tiles_within_limit():
    guard = cp_sat._compute_edge_guard_cells(
        6.0,
        cell_size=0.5,
        board_max=24,
        test_mode=False,
        max_tile_side_ft=5.0,
    )
    assert guard == 12
