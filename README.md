# Tile Sorter Thinger

Tile Sorter Thinger is a Flask web application that turns a bag of rectangular tile sizes into a packed layout using an OR-Tools CP-SAT solver. The project includes a client-side form for quickly describing tile demand, a server-side orchestrator that coordinates multiple solve phases, and rich progress reporting so users can track what the solver is doing in real time.【F:app.py†L8-L24】【F:tile_selection_form.html†L1-L133】

## Project tour

* **`tile_selection_form.html`** – Single-page form with a pre-rendered catalog of tile sizes, live estimates of solve difficulty, and a modal overlay that polls `/progress3` for updates while the solver runs.【F:tile_selection_form.html†L14-L200】
* **`app.py`** – Flask entry point that normalizes requests, calls the solver orchestrator, streams progress updates, and renders download links for the coordinate report and layout preview when a solution is found.【F:app.py†L64-L471】
* **`solver/`** – Algorithms that search for placements. `solver/cp_sat.py` contains the OR-Tools CP-SAT exact-cover model, while `solver/orchestrator.py` manages phase sequencing, grid candidates, and adaptive search limits.【F:solver/cp_sat.py†L1-L200】【F:solver/orchestrator.py†L1-L120】
* **`progress.py`** – Thread-safe service that persists status fields to disk, powers the polling overlay, and records attempt logs when enabled.【F:progress.py†L11-L187】
* **`io_files.py`** – Helpers that write coordinate reports and HTML previews to configurable locations for download.【F:io_files.py†L12-L82】
* **`tests/`** – Pytest suite covering the orchestrator, IO helpers, and progress service.

## Getting started

1. **Install dependencies**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install flask ortools pytest
   ```
   OR-Tools requires Python 3.8+; the test suite currently runs on Python 3.11.

2. **Run the web server**
   ```bash
   python app.py
   ```
   The app serves `tile_selection_form.html` at `http://localhost:5000/`. Submit tile counts and watch the overlay update as the solver progresses.【F:app.py†L64-L471】

3. **Inspect results**
   * Open `/result/latest` for the most recent run.
   * Download solver artifacts from `/download/coords` (text report) and `/download/html` (layout preview).【F:app.py†L459-L466】【F:io_files.py†L21-L82】

4. **Run the tests**
   ```bash
   pytest
   ```

## Configuration reference

All solver knobs are exposed via environment variables (prefix `TS_`). Defaults are shown below.【F:config.py†L4-L118】

| Category | Variable | Default | Description |
| --- | --- | --- | --- |
| Core units | `TS_CELL` | `0.5` | Side length, in feet, for one grid cell used throughout the solver.【F:config.py†L4-L6】 |
| Workers & limits | `TS_WORKERS` | `1` | Parallel worker count for multi-phase runs.【F:config.py†L7-L10】 |
|  | `TS_MAX_PLACEMENTS` | `200000` | Cap on placement combinations considered by CP-SAT before thinning.【F:config.py†L7-L15】 |
|  | `TS_MAX_MEMORY_MB` | `2048` | Soft memory budget passed down to solver phases.【F:config.py†L7-L10】 |
| Candidate generation | `TS_CAND_WIDTHS` | `auto` | Overrides width heuristics for candidate board sizes.【F:config.py†L12-L15】 |
|  | `TS_CAND_NEIGHBOR_DELTAS` | `auto` | Step sizes for neighborhood probes when expanding candidates.【F:config.py†L12-L15】 |
|  | `TS_MAX_CANDIDATES` | `4000` | Upper bound on candidate boards per phase.【F:config.py†L12-L15】 |
| Rule guards | `TS_BLOCK_EXACT_10x10` | `0` | Disable the exact 10×10 ft board phase when set to 1.【F:config.py†L17-L24】 |
|  | `TS_MAX_INTERNAL_SEAM_FT` | `3` | Maximum seam length allowed inside composite tiles.【F:config.py†L17-L20】 |
|  | `TS_MAX_EDGE_FT` | `6` | Maximum exposed edge length for placed tiles.【F:config.py†L17-L20】 |
|  | `TS_NO_PLUS` | `1` | When non-zero, disallows plus-shaped placements.【F:config.py†L17-L21】 |
|  | `TS_SAME_SHAPE_LIMIT` | `1` | Maximum same-shape neighbours allowed per tile (`-1` disables the guard).【F:config.py†L21-L25】 |
| Grid heuristics | `TS_BASE_GRID_AREA_SQFT` | `100` | Starting board area (ft²) used to split phases between small and large grids.【F:config.py†L25-L28】 |
| Pre-flight sizing | `TS_S_MAX_EST_PLACEMENTS` | `150000` | Estimate cap for preliminary passes.【F:config.py†L30-L33】 |
|  | `TS_S1_MAX_HEIGHT_FT` | `16` | Maximum board height (ft) evaluated in stage S1.【F:config.py†L30-L33】 |
| Legacy timeboxes | `TS_TIME_A`…`TS_TIME_F` | see defaults | Time limits (seconds) for legacy A–F solve phases.【F:config.py†L34-L40】 |
| Pre-flight timeboxes | `TS_TIME_S0`…`TS_TIME_S6` | see defaults | Time limits (seconds) for modern S0–S6 stages.【F:config.py†L42-L49】 |
| Grid sweep (Phase F) | `TS_GRID_STRIDE_BASE` | `10` | Base number of stride probes when sweeping grids.【F:config.py†L51-L58】 |
|  | `TS_GRID_STRIDE_STEP` | `1` | Increment/decrement applied when expanding the stride.【F:config.py†L51-L58】 |
|  | `TS_GRID_MIN_W_FT` / `TS_GRID_MIN_H_FT` | `8` | Minimum width/height (ft) explored during sweeps.【F:config.py†L51-L58】 |
|  | `TS_GRID_MAX_W_FT` / `TS_GRID_MAX_H_FT` | `24` | Maximum width/height (ft) explored during sweeps.【F:config.py†L51-L58】 |
| Outputs | `TS_COORDS_OUT` | `coords.txt` | Destination (path or filename) for the coordinate report.【F:config.py†L59-L62】【F:io_files.py†L21-L37】 |
|  | `TS_LAYOUT_HTML` | `layout_view.html` | Destination for the downloadable layout preview HTML.【F:config.py†L59-L62】【F:io_files.py†L39-L82】 |
| Testing | `TS_TEST_MODE` | `1` | Enables relaxed edge rules useful while iterating locally.【F:config.py†L63-L64】 |

### Progress persistence

The progress service stores the latest snapshot in `solver/logs/progress_state.json` and writes attempt logs to `solver/logs/solver_attempts.log`. Override the state file location with `PROGRESS_STATE_FILE` if you deploy to read-only disks.【F:progress.py†L15-L54】

## Solver internals

* Tile demand is parsed from flexible form or JSON payloads, supporting keys such as `count_2x3` or nested arrays.【F:tiles.py†L24-L102】
* The orchestrator converts tiles to grid cells, estimates viable board sizes, and calls CP-SAT with adaptive stride thinning to keep placement counts tractable.【F:solver/orchestrator.py†L29-L133】【F:solver/cp_sat.py†L84-L140】
* When the solver returns placements, the app renders an SVG layout, writes artifact files, and exposes them for download alongside a normalized summary payload.【F:app.py†L406-L456】【F:io_files.py†L21-L82】

## Development tips

* Use the `/progress3` endpoint to inspect solver status JSON while a run is active.【F:app.py†L469-L471】
* Tune environment variables in `.env` or via your process manager to match real-world job sizes.
* The pytest suite is a good starting point for understanding failure modes (`tests/test_orchestrator.py`, `tests/test_solver_alignment.py`, etc.).
