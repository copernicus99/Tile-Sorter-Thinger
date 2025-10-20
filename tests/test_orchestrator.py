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

from solver.orchestrator import _coerce_bag_ft

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
