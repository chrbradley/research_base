"""ArUco-driven calibration utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

Point = Tuple[float, float]


@dataclass
class CalibrationResult:
    scale_pixels_per_unit: float
    marker_corners: Optional[np.ndarray]
    warped_image: Optional[np.ndarray]


ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
DETECTOR_PARAMS = cv2.aruco.DetectorParameters()
DETECTOR = cv2.aruco.ArucoDetector(ARUCO_DICT, DETECTOR_PARAMS)


def detect_first_marker(image: np.ndarray) -> Tuple[Optional[np.ndarray], np.ndarray]:
    """Detect the first ArUco marker in the image."""

    corners, ids, rejected = DETECTOR.detectMarkers(image)
    if ids is None or len(corners) == 0:
        return None, rejected
    return corners[0], rejected


def compute_scale(marker_corners: np.ndarray, marker_size_in_inches: float = 5.0) -> float:
    """Compute pixels-per-inch scale from detected ArUco corners."""

    side_lengths = []
    for i in range(4):
        p1 = marker_corners[i][0]
        p2 = marker_corners[(i + 1) % 4][0]
        side_lengths.append(np.linalg.norm(p1 - p2))
    mean_side_pixels = float(np.mean(side_lengths))
    return mean_side_pixels / marker_size_in_inches


def warp_to_marker_square(image: np.ndarray, marker_corners: np.ndarray, output_size: int = 600) -> np.ndarray:
    """Perspective correct the image so the marker becomes a square."""

    src = marker_corners[:, 0, :].astype(np.float32)
    dst = np.array(
        [[0, 0], [output_size - 1, 0], [output_size - 1, output_size - 1], [0, output_size - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(image, matrix, (output_size, output_size))
    return warped


def normalize_polygons(polygons: Sequence[Sequence[Point]], scale_pixels_per_inch: float) -> Sequence[np.ndarray]:
    """Scale pixel polygons into real-world inch units."""

    scaled = []
    for polygon in polygons:
        scaled_points = np.array(polygon, dtype=np.float32) / scale_pixels_per_inch
        scaled.append(scaled_points)
    return scaled


def calibrate_image(image_bgr: np.ndarray) -> CalibrationResult:
    """Detect an ArUco marker, compute scale, and return deskewed imagery."""

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    marker, _ = detect_first_marker(gray)
    if marker is None:
        return CalibrationResult(scale_pixels_per_unit=1.0, marker_corners=None, warped_image=None)

    scale = compute_scale(marker)
    warped = warp_to_marker_square(image_bgr, marker)
    return CalibrationResult(scale_pixels_per_unit=scale, marker_corners=marker, warped_image=warped)
