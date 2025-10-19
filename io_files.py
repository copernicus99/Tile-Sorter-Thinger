"""Helpers for writing solver outputs to disk."""

from __future__ import annotations

import os
from typing import List

from config import CFG
from models import Placed


def _resolve_output_path(base_dir: str, configured_name: str, fallback: str) -> str:
    """Return the absolute path where an output artifact should be written."""

    name = (configured_name or "").strip() or fallback
    if os.path.isabs(name):
        return name
    return os.path.join(base_dir, name)


def write_coords(placed: List[Placed], Wc: int, Hc: int, base_dir: str) -> str:
    """Write the placed tile coordinates to the configured text file."""

    path = _resolve_output_path(base_dir, CFG.COORDS_OUT, "coords.txt")
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        if not placed:
            f.write("No solution\n")
        else:
            for p in placed:
                xft, yft, wft, hft = p.to_ft_tuple()
                f.write(
                    f"{p.rect.name} @ ({xft:.2f},{yft:.2f}) size ({wft:.2f}×{hft:.2f})\n"
                )
    return path


def write_layout_view_html(svg: str, legend_html: str, base_dir: str) -> str:
    """Write the rendered SVG/legend preview to the configured HTML file."""

    path = _resolve_output_path(base_dir, CFG.LAYOUT_HTML, "layout_view.html")
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w", encoding="utf-8") as vf:
        vf.write(
            f"""<!doctype html>
<html><head><meta charset='utf-8'><title>Layout View</title>
<link rel='stylesheet' href='/styles.css'>
<link rel='stylesheet' href='/styles_add_rcgrid.css'></head>
<body class='container'>
<h1>Layout View</h1>
<section class='card'><div class='gridwrap'><div class='grid-bg'></div>{svg}</div></section>
<section class='card'><h3>Legend</h3><ul>{legend_html}</ul></section>
</body></html>"""
        )
    return path


__all__ = ["write_coords", "write_layout_view_html"]
