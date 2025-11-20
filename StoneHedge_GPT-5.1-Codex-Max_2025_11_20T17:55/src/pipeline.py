"""End-to-end helper functions for the StoneHedge prototype."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import cv2
import numpy as np
from shapely.geometry import Polygon

from .calibration import CalibrationResult, calibrate_image, normalize_polygons
from .packing import PackingResult, make_target_polygon, pack_greedy, polygon_from_points
from .segmentation import SegmentResult, segment_stones

DATA_DIR = Path("data/stonehedge/aruco_stones")


@dataclass
class StoneShape:
    path: Path
    raw_image: np.ndarray
    calibration: CalibrationResult
    segmentation: SegmentResult
    polygon_inches: Polygon


@dataclass
class PipelineOutput:
    stones: List[StoneShape]
    target: Polygon
    packing: PackingResult


def load_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(path)
    return image


def build_stone(path: Path) -> StoneShape:
    image = load_image(path)
    calibration = calibrate_image(image)
    segmentation = segment_stones(image)
    scaled_polygons = normalize_polygons(segmentation.polygons, calibration.scale_pixels_per_unit)
    polygon_shapes = [polygon_from_points(poly) for poly in scaled_polygons]
    polygon_inches = polygon_shapes[0] if polygon_shapes else Polygon()
    return StoneShape(path=path, raw_image=image, calibration=calibration, segmentation=segmentation, polygon_inches=polygon_inches)


def run_pipeline(image_limit: int = 5, target_kind: str = "square", target_area_sqft: float = 100.0) -> PipelineOutput:
    paths = sorted(DATA_DIR.glob("IMG_34*.JPEG"))[:image_limit]
    stones: List[StoneShape] = [build_stone(path) for path in paths]

    stone_polygons = [stone.polygon_inches for stone in stones if not stone.polygon_inches.is_empty]
    target = make_target_polygon(target_kind, target_area_sqft)
    packing = pack_greedy(stone_polygons, target)

    return PipelineOutput(stones=stones, target=target, packing=packing)
