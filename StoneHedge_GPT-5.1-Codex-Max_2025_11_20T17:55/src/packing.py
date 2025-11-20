"""Simple irregular polygon packing prototype using Shapely."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np
from shapely import affinity
from shapely.geometry import Polygon
from shapely.ops import unary_union

Point = Tuple[float, float]


def make_target_polygon(kind: str, area_sqft: float) -> Polygon:
    """Generate a simple target polygon shape of approximately the requested area."""

    area_sq_in = area_sqft * 144.0
    if kind.lower() == "square":
        side = math.sqrt(area_sq_in)
        return Polygon([(0, 0), (side, 0), (side, side), (0, side)])
    if kind.lower() == "rectangle":
        width = math.sqrt(area_sq_in / 1.5)
        height = area_sq_in / width
        return Polygon([(0, 0), (width, 0), (width, height), (0, height)])
    raise ValueError(f"Unsupported target kind: {kind}")


def polygon_from_points(points: Sequence[Sequence[float]]) -> Polygon:
    return Polygon(points)


@dataclass
class Placement:
    stone_index: int
    shape: Polygon


@dataclass
class PackingResult:
    placements: List[Placement]
    coverage_ratio: float
    filled_area_sqft: float


def _random_pose(shape: Polygon, bounds: Tuple[float, float, float, float]) -> Polygon:
    minx, miny, maxx, maxy = bounds
    tx = random.uniform(minx, maxx)
    ty = random.uniform(miny, maxy)
    angle = random.uniform(0, 360)
    rotated = affinity.rotate(shape, angle, origin="centroid")
    return affinity.translate(rotated, xoff=tx, yoff=ty)


def pack_greedy(polygons: Iterable[Polygon], target: Polygon, max_iters: int = 500, seed: int = 13) -> PackingResult:
    """Greedy random packing: place shapes if they fit without intersection."""

    random.seed(seed)
    placements: List[Placement] = []
    placed_union = unary_union([])

    for idx, shape in enumerate(polygons):
        for _ in range(max_iters):
            candidate = _random_pose(shape, target.bounds)
            if not target.contains(candidate):
                continue
            if placed_union.is_empty:
                placements.append(Placement(stone_index=idx, shape=candidate))
                placed_union = candidate
                break
            if placed_union.disjoint(candidate):
                placements.append(Placement(stone_index=idx, shape=candidate))
                placed_union = unary_union([placed_union, candidate])
                break

    filled_area = placed_union.area if not placed_union.is_empty else 0.0
    coverage_ratio = filled_area / target.area
    return PackingResult(placements=placements, coverage_ratio=coverage_ratio, filled_area_sqft=filled_area / 144.0)
