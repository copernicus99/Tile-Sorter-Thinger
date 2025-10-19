
from dataclasses import dataclass
from typing import Tuple
from config import CELL

def ft_to_cells(ft: float) -> int:
    return round(ft / CELL + 1e-9)

def cells_to_ft(c: int) -> float:
    return round(c * CELL + 1e-9, 6)

@dataclass(frozen=True)
class Rect:
    w: int
    h: int
    name: str

@dataclass
class Placed:
    x: int
    y: int
    rect: Rect

    def to_ft_tuple(self):
        from models import cells_to_ft
        return (
            cells_to_ft(self.x),
            cells_to_ft(self.y),
            cells_to_ft(self.rect.w),
            cells_to_ft(self.rect.h),
        )

@dataclass
class Meta:
    template: str
    elapsed_sec: float
    edge_label: str

    def template_vars(self, **kw):
        elapsed = f"{int(self.elapsed_sec//60)}m {int(self.elapsed_sec%60)}s"
        return dict(elapsed_str=elapsed, edge_label=self.edge_label, **kw)
