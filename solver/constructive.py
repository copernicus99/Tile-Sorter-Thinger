# solver/constructive.py
from typing import Dict, List, Tuple, Optional, Set
from tiles import NAME_TO_RECT_DIMS
from models import Placed, Rect, ft_to_cells, cells_to_ft

def quick_banded_fill_width10(demand: Dict[str, int]) -> Tuple[Optional[List[Placed]], int, int]:
    """
    Full-width 10 ft banded constructive solver.

    If the given tile set can be arranged in horizontal bands that exactly tile a 10 ft width
    (i.e., each tile width divides 10 ft, possibly after rotation), and the total area is an
    exact multiple of 10, this constructs a placement instantly.

    Returns:
        (placed, Wcells, Hcells) on success, or (None, 0, 0) if not applicable.
    """
    W_ft = 10.0

    # Build a (width_ft, height_ft, count, name) list, preferring shorter height for banding.
    items: List[List[object]] = []
    for name, cnt in demand.items():
        w_c, h_c = NAME_TO_RECT_DIMS[name]
        w_ft, h_ft = cells_to_ft(w_c), cells_to_ft(h_c)
        # Prefer orientation with smaller side as band height.
        if h_ft > w_ft:
            w_ft, h_ft = h_ft, w_ft
        items.append([w_ft, h_ft, int(cnt), name])

    # Each band's tile width must divide 10 ft
    for w_ft, _, cnt, _ in items:
        if cnt > 0 and abs(W_ft % float(w_ft)) > 1e-9:
            return None, 0, 0

    # Total height must be total_area / 10
    total_area = 0.0
    for w_ft, h_ft, cnt, _ in items:
        total_area += float(w_ft) * float(h_ft) * int(cnt)
    if abs(total_area % W_ft) > 1e-9:
        return None, 0, 0
    target_H_ft = total_area / W_ft

    # Greedy tallest-bands-first
    items.sort(key=lambda t: float(t[1]), reverse=True)

    placed: List[Placed] = []
    y = 0.0
    remaining = target_H_ft

    for i, (w_ft, h_ft, cnt, name) in enumerate(items):
        w_ft = float(w_ft); h_ft = float(h_ft); cnt = int(cnt)
        across = int(round(W_ft / w_ft)) if w_ft > 0 else 0  # tiles per full row
        max_rows = (cnt // across) if across else 0
        if max_rows <= 0:
            continue

        usable_rows = min(max_rows, int((remaining + 1e-9) // h_ft))
        for _ in range(usable_rows):
            x = 0.0
            for __ in range(across):
                placed.append(
                    Placed(
                        ft_to_cells(x),
                        ft_to_cells(y),
                        Rect(ft_to_cells(w_ft), ft_to_cells(h_ft), name)  # keep label in chosen orientation
                    )
                )
                x += w_ft
            items[i][2] = int(items[i][2]) - across
            y += h_ft
            remaining = round(target_H_ft - y, 6)
            if remaining <= 1e-9:
                break
        if remaining <= 1e-9:
            break

    if remaining > 1e-6:
        # Couldn’t reach exact target height with full rows
        return None, 0, 0

    return placed, ft_to_cells(W_ft), ft_to_cells(target_H_ft)


def quick_strip_fill_common_dimension(demand: Dict[str, int]) -> Tuple[Optional[List[Placed]], int, int]:
    """Construct layouts when every tile shares a common strip dimension.

    The solver looks for a dimension (width or height) that every tile can
    adopt through rotation.  When such a dimension exists and the total area
    forms an integer number of strips, the layout can be assembled instantly
    without invoking CP-SAT.
    """

    if not demand:
        return None, 0, 0

    bag: List[Tuple[int, int, int, str]] = []
    candidate_dims: Optional[Set[int]] = None
    total_area = 0

    for name, cnt in demand.items():
        if cnt <= 0:
            continue
        w, h = NAME_TO_RECT_DIMS[name]
        bag.append((w, h, int(cnt), name))
        total_area += int(cnt) * w * h
        dims = {w, h}
        if candidate_dims is None:
            candidate_dims = set(dims)
        else:
            candidate_dims &= dims

    if not bag or not candidate_dims:
        return None, 0, 0

    def _attempt_strip(strip_width: int, align_width: bool) -> Tuple[Optional[List[Placed]], int, int]:
        if strip_width <= 0:
            return None, 0, 0
        if total_area % strip_width != 0:
            return None, 0, 0

        strip_height = total_area // strip_width
        placements: List[Placed] = []
        cursor = 0

        # Prepare oriented tiles (width aligned if align_width else height)
        oriented: List[Tuple[int, int, int, str]] = []
        for w, h, cnt, name in bag:
            if align_width:
                if w == strip_width:
                    oriented.append((w, h, cnt, name))
                elif h == strip_width:
                    oriented.append((h, w, cnt, name))
                else:
                    return None, 0, 0
            else:
                if h == strip_width:
                    oriented.append((w, h, cnt, name))
                elif w == strip_width:
                    oriented.append((h, w, cnt, name))
                else:
                    return None, 0, 0

        for width, height, cnt, name in sorted(oriented, key=lambda t: (-t[1], t[3])):
            for _ in range(cnt):
                if align_width:
                    if cursor + height > strip_height:
                        return None, 0, 0
                    placements.append(Placed(0, cursor, Rect(width, height, name)))
                    cursor += height
                else:
                    if cursor + width > strip_height:
                        return None, 0, 0
                    placements.append(Placed(cursor, 0, Rect(width, height, name)))
                    cursor += width

        if cursor != strip_height:
            return None, 0, 0

        if align_width:
            return placements, strip_width, strip_height
        return placements, strip_height, strip_width

    # Try aligning widths first, then heights
    for dim in sorted(candidate_dims, reverse=True):
        res = _attempt_strip(dim, align_width=True)
        if res[0]:
            return res
        res = _attempt_strip(dim, align_width=False)
        if res[0]:
            return res

    return None, 0, 0
