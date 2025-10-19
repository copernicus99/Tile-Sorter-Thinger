from progress import reset, set_status, set_done, snapshot


def test_set_done_no_args_defaults_to_solved():
    reset()
    set_done()
    snap = snapshot()
    assert snap["status"] == "Solved"
    assert snap["percent"] == 100.0


def test_set_done_accepts_legacy_arguments():
    reset()
    set_status("error")
    set_done(False, reason="boom")
    snap = snapshot()
    assert snap["status"] == "Error"
    assert snap["percent"] == 100.0
    assert snap["message"] == "boom"
