
import random
from typing import List, Tuple, Dict
from models import Placed, cells_to_ft

def _color(name: str) -> str:
    random.seed(hash(name) & 0xFFFFFFFF)
    r = random.randint(40, 200)
    g = random.randint(40, 200)
    b = random.randint(40, 200)
    return f"rgb({r},{g},{b})"

def render_result(placed: List[Placed], Wc: int, Hc: int):
    palette: Dict[str, str] = {}
    for p in placed:
        palette.setdefault(p.rect.name, _color(p.rect.name))

    scale = 120
    Wft = cells_to_ft(Wc)
    Hft = cells_to_ft(Hc)
    svg_w = int(Wft * scale) + 2
    svg_h = int(Hft * scale) + 2

    rects = []
    for p in placed:
        xft, yft, wft, hft = p.to_ft_tuple()
        x = int(xft * scale)
        y = int(yft * scale)
        w = int(wft * scale)
        h = int(hft * scale)
        rects.append(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{palette[p.rect.name]}" stroke="black" stroke-width="1"/>'
            f'<text x="{x+4}" y="{y+14}" font-size="12" fill="black">{p.rect.name}</text>'
        )
    grid = f'<rect x="1" y="1" width="{svg_w-2}" height="{svg_h-2}" fill="none" stroke="black" stroke-width="2"/>'
    svg = (
        f'<svg class="layout-svg" xmlns="http://www.w3.org/2000/svg" '
        f'width="{svg_w}" height="{svg_h}" '
        f'viewBox="0 0 {svg_w} {svg_h}" preserveAspectRatio="xMinYMin meet">'
        f'{grid}{"".join(rects)}</svg>'
    )

    legend = "".join(f"<li><span class='swatch' style='background:{c}'></span>{n}</li>" for n, c in palette.items())
    return svg, legend
