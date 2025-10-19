
import os
from typing import List, Tuple
from models import Placed, cells_to_ft

def write_coords(placed: List[Placed], Wc: int, Hc: int, base_dir: str) -> str:
    path = os.path.join(base_dir, "layout_coords.txt")
    with open(path, "w") as f:
        if not placed:
            f.write("No solution\n")
        else:
            for p in placed:
                xft, yft, wft, hft = p.to_ft_tuple()
                f.write(f"{p.rect.name} @ ({xft:.2f},{yft:.2f}) size ({wft:.2f}×{hft:.2f})\n")
    return path

def write_layout_view_html(svg: str, legend_html: str, base_dir: str) -> str:
    path = os.path.join(base_dir, "layout_view.html")
    with open(path, "w", encoding="utf-8") as vf:
        vf.write(f"""<!doctype html>
<html><head><meta charset='utf-8'><title>Layout View</title>
<link rel='stylesheet' href='/styles.css'>
<link rel='stylesheet' href='/styles_add_rcgrid.css'></head>
<body class='container'>
<h1>Layout View</h1>
<section class='card'><div class='gridwrap'><div class='grid-bg'></div>{svg}</div></section>
<section class='card'><h3>Legend</h3><ul>{legend_html}</ul></section>
</body></html>""" )
    return path
