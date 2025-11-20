"""Stone segmentation helpers using OpenCV contour extraction.

This module keeps the implementation lightweight so it can run offline
on CPU-only environments. The goal is to provide a baseline that can be
swapped with SAM/SAM2/SAM3 or other segmentation backends later.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import cv2
import numpy as np

Contour = np.ndarray
Point = Tuple[float, float]


@dataclass
class SegmentResult:
    """Container for extracted stone polygons.

    Attributes:
        contours: Raw OpenCV contours for downstream processing.
        polygons: Simplified polygons as lists of (x, y) vertices.
        mask: Binary mask showing detected stone regions.
    """

    contours: List[Contour]
    polygons: List[List[Point]]
    mask: np.ndarray


def _remove_aruco_contours(contours: Sequence[Contour], min_area: float) -> List[Contour]:
    """Filter out likely ArUco marker contours based on area and squareness.

    The ArUco marker is 5x5 inches and appears as a near-perfect square.
    We remove quadrilateral contours that are much smaller than the stone
    surface to avoid polluting the polygon extraction.
    """

    cleaned: List[Contour] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        if len(approx) == 4:
            # Likely the marker or a rectangular artifact; drop it.
            continue
        cleaned.append(contour)
    return cleaned


def segment_stones(image: np.ndarray, min_area_ratio: float = 0.005) -> SegmentResult:
    """Extract stone polygons from an image.

    Args:
        image: BGR OpenCV image containing one stone and an ArUco marker.
        min_area_ratio: Minimum contour area as a fraction of the image area.

    Returns:
        SegmentResult containing contours, simplified polygons, and a mask.
    """

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Adaptive threshold to separate stone from background.
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 2
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_area = image.shape[0] * image.shape[1]
    min_area = image_area * min_area_ratio
    filtered = [c for c in contours if cv2.contourArea(c) >= min_area]
    filtered = _remove_aruco_contours(filtered, min_area=min_area)

    polygons: List[List[Point]] = []
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, filtered, -1, color=255, thickness=-1)

    for contour in filtered:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.005 * peri, True)
        polygon = [(float(x), float(y)) for [[x, y]] in approx]
        polygons.append(polygon)

    return SegmentResult(contours=filtered, polygons=polygons, mask=mask)


def polygon_area(points: Sequence[Point]) -> float:
    """Compute polygon area using the shoelace formula."""

    xs, ys = zip(*points)
    area = 0.5 * np.abs(np.dot(xs, np.roll(ys, 1)) - np.dot(ys, np.roll(xs, 1)))
    return float(area)
