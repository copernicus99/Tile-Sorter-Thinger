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


def test_strategy_field_populated_via_set_progress():
    reset()
    _set_progress(strategy="S0", phase="", phase_total=3)
    snap = snapshot()
    assert snap["strategy"] == "S0"
    # phase remains blank when only strategy is provided
    assert snap["phase"] == ""
