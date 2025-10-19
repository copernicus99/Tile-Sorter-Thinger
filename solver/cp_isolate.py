# solver/cp_isolate.py
import multiprocessing as mp
from typing import List, Tuple, Optional
import traceback

# Worker must be top-level (picklable on Windows spawn)
def _solve_worker(q, W: int, H: int, bag, allow_discard: bool, max_seconds: float):
    try:
        from solver.cp_sat import try_pack_exact_cover  # import inside child
        ok, placed, reason = try_pack_exact_cover(W, H, bag, allow_discard, max_seconds)
        q.put(("ok", ok, placed, reason))
    except MemoryError:
        q.put(("err", False, [], "Child ran out of memory"))
    except Exception as e:
        q.put(("exc", False, [], f"{e}\n{traceback.format_exc()}"))

def run_cp_sat_isolated(W: int, H: int, bag, allow_discard: bool, max_seconds: float) -> Tuple[bool, List, Optional[str], Optional[str]]:
    """
    Returns (ok, placed, reason, crash_note).
    crash_note is non-empty only if the child crashed/was killed/timed out.
    """
    ctx = mp.get_context("spawn")  # safest on Windows
    q: mp.Queue = ctx.Queue()
    p = ctx.Process(target=_solve_worker, args=(q, W, H, bag, allow_discard, float(max_seconds)))
    p.daemon = True
    p.start()

    # Allow a small buffer beyond model time for teardown
    timeout = float(max_seconds) + 5.0
    p.join(timeout=timeout)

    if p.is_alive():
        p.terminate()
        p.join(2.0)
        return False, [], "Stopped before solution (timebox)", "killed: timeout"

    # Non-zero exit code means native crash or hard error
    if p.exitcode not in (0, None):
        return False, [], f"Stopped before solution (child exit {p.exitcode})", "child crashed"

    try:
        tag, ok, placed, reason = q.get_nowait()
    except Exception:
        return False, [], "No result from child process", "no-result"

    if tag == "ok":
        return ok, placed, reason, None
    elif tag == "err":
        return False, [], reason, None
    else:  # "exc"
        return False, [], reason, None
