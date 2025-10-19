# config.py
import os

# ======= Core units / cell size =======
CELL = float(os.getenv("TS_CELL", "0.5"))

# ======= Worker / search caps =======
WORKERS              = int(os.getenv("TS_WORKERS", "1"))
MAX_PLACEMENTS       = int(os.getenv("TS_MAX_PLACEMENTS", "200000"))
MAX_MEMORY_MB        = int(os.getenv("TS_MAX_MEMORY_MB", "2048"))

# ======= Candidate generation knobs =======
CAND_WIDTHS          = os.getenv("TS_CAND_WIDTHS", "auto")
CAND_NEIGHBOR_DELTAS = os.getenv("TS_CAND_NEIGHBOR_DELTAS", "auto")
MAX_CANDIDATES       = int(os.getenv("TS_MAX_CANDIDATES", "4000"))

# ======= Rule / guard knobs =======
BLOCK_EXACT_10x10     = int(os.getenv("TS_BLOCK_EXACT_10x10", "0"))
MAX_INTERNAL_SEAM_FT  = float(os.getenv("TS_MAX_INTERNAL_SEAM_FT", "3"))
MAX_EDGE_FT           = float(os.getenv("TS_MAX_EDGE_FT", "6"))
NO_PLUS               = int(os.getenv("TS_NO_PLUS", "1")) != 0
SAME_SHAPE_LIMIT      = int(os.getenv("TS_SAME_SHAPE_LIMIT", "1"))
BASE_GRID_AREA_SQFT   = float(os.getenv("TS_BASE_GRID_AREA_SQFT", "1000"))

# ======= Pre-flight sizing / accounting =======
S_MAX_EST_PLACEMENTS  = int(os.getenv("TS_S_MAX_EST_PLACEMENTS", "150000"))
S1_MAX_HEIGHT_FT      = float(os.getenv("TS_S1_MAX_HEIGHT_FT", "16"))

# ======= Legacy A..F timeboxes (seconds) =======
TIME_A = int(os.getenv("TS_TIME_A", "600"))
TIME_B = int(os.getenv("TS_TIME_B", "600"))
TIME_C = int(os.getenv("TS_TIME_C", "300"))
TIME_D = int(os.getenv("TS_TIME_D", "300"))
TIME_E = int(os.getenv("TS_TIME_E", "900"))
TIME_F = int(os.getenv("TS_TIME_F", "900"))

# ======= New S0..S6 (pre-flight) timeboxes (seconds) =======
TIME_S0 = int(os.getenv("TS_TIME_S0", "30"))
TIME_S1 = int(os.getenv("TS_TIME_S1", "45"))
TIME_S2 = int(os.getenv("TS_TIME_S2", "60"))
TIME_S3 = int(os.getenv("TS_TIME_S3", "30"))
TIME_S4 = int(os.getenv("TS_TIME_S4", "45"))
TIME_S5 = int(os.getenv("TS_TIME_S5", "45"))
TIME_S6 = int(os.getenv("TS_TIME_S6", "30"))

# ======= Grid sweep (F-phase) knobs =======
GRID_STRIDE_BASE = int(os.getenv("TS_GRID_STRIDE_BASE", "10"))  # number of base probes
GRID_STRIDE_STEP = int(os.getenv("TS_GRID_STRIDE_STEP", "1"))   # grow/shrink delta (cells)
GRID_MIN_W_FT    = float(os.getenv("TS_GRID_MIN_W_FT", "8"))
GRID_MIN_H_FT    = float(os.getenv("TS_GRID_MIN_H_FT", "8"))
GRID_MAX_W_FT    = float(os.getenv("TS_GRID_MAX_W_FT", "24"))
GRID_MAX_H_FT    = float(os.getenv("TS_GRID_MAX_H_FT", "24"))

# ======= Output names =======
COORDS_OUT  = os.getenv("TS_COORDS_OUT", "coords.txt")
LAYOUT_HTML = os.getenv("TS_LAYOUT_HTML", "layout_view.html")

# ======= Test mode (edge-perimeter relaxation) =======
TEST_MODE = (os.getenv("TS_TEST_MODE", "1") == "1")  # on by default

class CFG:
    CELL = CELL

    WORKERS        = WORKERS
    MAX_PLACEMENTS = MAX_PLACEMENTS
    MAX_MEMORY_MB  = MAX_MEMORY_MB

    CAND_WIDTHS          = CAND_WIDTHS
    CAND_NEIGHBOR_DELTAS = CAND_NEIGHBOR_DELTAS
    MAX_CANDIDATES       = MAX_CANDIDATES

    BLOCK_EXACT_10x10    = BLOCK_EXACT_10x10
    MAX_INTERNAL_SEAM_FT = MAX_INTERNAL_SEAM_FT
    MAX_EDGE_FT          = MAX_EDGE_FT
    NO_PLUS              = NO_PLUS
    SAME_SHAPE_LIMIT     = SAME_SHAPE_LIMIT
    BASE_GRID_AREA_SQFT  = BASE_GRID_AREA_SQFT

    S_MAX_EST_PLACEMENTS = S_MAX_EST_PLACEMENTS
    S1_MAX_HEIGHT_FT     = S1_MAX_HEIGHT_FT

    TIME_A = TIME_A
    TIME_B = TIME_B
    TIME_C = TIME_C
    TIME_D = TIME_D
    TIME_E = TIME_E
    TIME_F = TIME_F

    TIME_S0 = TIME_S0
    TIME_S1 = TIME_S1
    TIME_S2 = TIME_S2
    TIME_S3 = TIME_S3
    TIME_S4 = TIME_S4
    TIME_S5 = TIME_S5
    TIME_S6 = TIME_S6

    GRID_STRIDE_BASE = GRID_STRIDE_BASE
    GRID_STRIDE_STEP = GRID_STRIDE_STEP
    GRID_MIN_W_FT    = GRID_MIN_W_FT
    GRID_MIN_H_FT    = GRID_MIN_H_FT
    GRID_MAX_W_FT    = GRID_MAX_W_FT
    GRID_MAX_H_FT    = GRID_MAX_H_FT

    COORDS_OUT  = COORDS_OUT
    LAYOUT_HTML = LAYOUT_HTML

    TEST_MODE = TEST_MODE

# legacy convenience
CELL      = CFG.CELL
TEST_MODE = CFG.TEST_MODE

__all__ = ["CFG", "CELL", "TEST_MODE"]
