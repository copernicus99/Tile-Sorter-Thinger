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


def test_edge_guard_retry_succeeds_when_relaxed(cp_sat_module):
    calls = []

    def fake_solver(*, edge_guard_cells):
        calls.append(edge_guard_cells)
        if edge_guard_cells is not None:
            return False, [], "Stopped before solution (timebox)"
        return True, ["ok"], None

    ok, placed, reason = cp_sat_module._resolve_with_edge_guard(
        fake_solver,
        guard_cells=5,
        guard_applies=True,
    )

    assert ok is True
    assert placed == ["ok"]
    assert reason is None
    assert calls == [5, None]


def test_edge_guard_retry_reports_relaxed_reason(cp_sat_module):
    def fake_solver(*, edge_guard_cells):
        if edge_guard_cells is not None:
            return False, [], "Proven infeasible under current constraints"
        return False, [], "Stopped before solution (timebox)"

    ok, placed, reason = cp_sat_module._resolve_with_edge_guard(
        fake_solver,
        guard_cells=4,
        guard_applies=True,
    )

    assert ok is False
    assert placed == []
    assert reason == "Stopped before solution (timebox)"
