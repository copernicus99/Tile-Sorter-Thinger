import importlib
import json
import os
import time

from progress import reset, set_status, set_done, set_result_url, snapshot, _set_progress


def test_set_done_no_args_defaults_to_solved():
    reset()
    set_done()
    snap = snapshot()
    assert snap["status"] == "Solved"
    assert snap["percent"] == 100.0
    assert snap["done"] is True
    assert snap["ok"] is True
    assert snap["result_url"] == ""


def test_set_done_accepts_legacy_arguments():
    reset()
    set_status("error")
    set_done(False, reason="boom")
    snap = snapshot()
    assert snap["status"] == "Error"
    assert snap["percent"] == 100.0
    assert snap["message"] == "boom"
    assert snap["done"] is True
    assert snap["ok"] is False


def test_set_result_url_tracks_navigation_target():
    reset()
    set_result_url("/foo")
    snap = snapshot()
    assert snap["result_url"] == "/foo"
    assert snap["done"] is False


def test_reset_increments_run_identifier():
    reset()
    first = snapshot()["run_id"]
    reset()
    second = snapshot()["run_id"]
    assert isinstance(first, int)
    assert isinstance(second, int)
    assert second == first + 1


def test_strategy_field_populated_via_set_progress():
    reset()
    _set_progress(strategy="S0", phase="", phase_total=3)
    snap = snapshot()
    assert snap["strategy"] == "S0"
    # phase remains blank when only strategy is provided
    assert snap["phase"] == ""


def test_snapshot_reads_state_written_by_other_process(tmp_path, monkeypatch):
    import progress as progress_module

    state_path = tmp_path / "state.json"
    monkeypatch.setenv("PROGRESS_STATE_FILE", str(state_path))
    progress = importlib.reload(progress_module)

    progress.reset()
    progress.set_phase("S0")
    first = progress.snapshot()
    assert first["phase"] == "S0"

    data = dict(first)
    data["phase"] = "F"
    data["phase_total"] = "9"
    state_path.write_text(json.dumps(data))
    os.utime(state_path, None)

    with progress.PROGRESS_LOCK:
        progress.PROGRESS["phase"] = ""
        progress.PROGRESS["phase_total"] = ""
        progress._LAST_STATE_MTIME = 0.0

    time.sleep(0.01)
    updated = progress.snapshot()
    assert updated["phase"] == "F"
    assert updated["phase_total"] == "9"

    monkeypatch.delenv("PROGRESS_STATE_FILE", raising=False)
    importlib.reload(progress_module)
