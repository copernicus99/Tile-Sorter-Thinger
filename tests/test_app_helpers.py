import importlib
import sys
import types


def _install_flask_stub(monkeypatch):
    flask_mod = types.ModuleType("flask")

    class DummyFlask:
        def __init__(self, name, *args, **kwargs):
            self.name = name

        def route(self, *args, **kwargs):  # pragma: no cover - decorator passthrough
            def decorator(func):
                return func

            return decorator

        def after_request(self, func):  # pragma: no cover - decorator passthrough
            return func

    flask_mod.Flask = DummyFlask
    flask_mod.request = types.SimpleNamespace()
    flask_mod.render_template = lambda *args, **kwargs: ""
    flask_mod.send_from_directory = lambda *args, **kwargs: ""
    flask_mod.jsonify = lambda *args, **kwargs: ""
    flask_mod.url_for = lambda endpoint, **kwargs: f"/{endpoint}"

    monkeypatch.setitem(sys.modules, "flask", flask_mod)

    # Provide a minimal ortools stub so importing the orchestrator succeeds even
    # when the optional dependency is absent in test environments.
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
            pass

        class CpSolver:
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

    monkeypatch.setitem(sys.modules, "ortools", ortools_mod)
    monkeypatch.setitem(sys.modules, "ortools.sat", sat_mod)
    monkeypatch.setitem(sys.modules, "ortools.sat.python", python_mod)
    monkeypatch.setitem(sys.modules, "ortools.sat.python.cp_model", cp_model_mod)


def test_finalize_solver_progress_respects_ok_flag(monkeypatch):
    _install_flask_stub(monkeypatch)
    app = importlib.import_module("app")

    calls = {"status": [], "done": []}

    def fake_set_status(value):
        calls["status"].append(value)

    def fake_set_done(ok=None, *, reason=None, message=None):
        calls["done"].append((ok, reason, message))

    monkeypatch.setattr(app, "set_status", fake_set_status)
    monkeypatch.setattr(app, "set_done", fake_set_done)

    app._finalize_solver_progress(True, "All good")
    assert calls["status"][-1] == "Solved"
    assert calls["done"][-1] == (True, "All good", None)

    app._finalize_solver_progress(False, "error happened")
    assert calls["status"][-1] == "error"
    assert calls["done"][-1] == (False, "error happened", None)
