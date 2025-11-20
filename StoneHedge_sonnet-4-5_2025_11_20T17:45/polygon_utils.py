"""
Polygon utility functions using Shapely for geometric operations.
"""

import numpy as np
from shapely.geometry import Polygon, Point, MultiPolygon
from shapely.affinity import rotate, translate
from shapely import prepare
from typing import Tuple, List, Optional
import math


class StonePolygon:
    """Wrapper for a stone polygon with transformation and collision detection."""

    def __init__(self, vertices: np.ndarray, stone_id: str = None):
        """
        Initialize stone polygon.

        Args:
            vertices: Nx2 array of polygon vertices
            stone_id: Identifier for the stone
        """
        self.original_vertices = np.array(vertices)
        self.stone_id = stone_id or f"stone_{id(self)}"

        # Create Shapely polygon
        self.polygon = Polygon(vertices)
        if not self.polygon.is_valid:
            # Try to fix invalid polygons
            self.polygon = self.polygon.buffer(0)

        # Current transformation
        self.rotation = 0.0  # degrees
        self.translation = np.array([0.0, 0.0])  # x, y offset

        # Cache for transformed polygon
        self._transformed = self.polygon
        self._dirty = False

    @property
    def area(self) -> float:
        """Get polygon area."""
        return self.polygon.area

    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        """Get bounding box (minx, miny, maxx, maxy)."""
        return self._transformed.bounds

    @property
    def width(self) -> float:
        """Get width of bounding box."""
        minx, _, maxx, _ = self.bounds
        return maxx - minx

    @property
    def height(self) -> float:
        """Get height of bounding box."""
        _, miny, _, maxy = self.bounds
        return maxy - miny

    @property
    def centroid(self) -> np.ndarray:
        """Get centroid of polygon."""
        c = self._transformed.centroid
        return np.array([c.x, c.y])

    def rotate_to(self, angle: float):
        """
        Rotate polygon to absolute angle (around its centroid).

        Args:
            angle: Angle in degrees
        """
        self.rotation = angle
        self._dirty = True

    def translate_to(self, x: float, y: float):
        """
        Translate polygon to absolute position (centroid at x, y).

        Args:
            x, y: New centroid coordinates
        """
        self.translation = np.array([x, y])
        self._dirty = True

    def set_position(self, x: float, y: float, angle: float = None):
        """
        Set position and optionally rotation.

        Args:
            x, y: New centroid coordinates
            angle: Optional rotation angle
        """
        if angle is not None:
            self.rotation = angle
        self.translation = np.array([x, y])
        self._dirty = True

    def _update_transform(self):
        """Apply transformations if needed."""
        if self._dirty:
            # Start with original polygon
            p = self.polygon

            # Rotate around centroid
            if self.rotation != 0:
                p = rotate(p, self.rotation, origin='centroid', use_radians=False)

            # Translate centroid to target position
            current_centroid = p.centroid
            dx = self.translation[0] - current_centroid.x
            dy = self.translation[1] - current_centroid.y
            p = translate(p, xoff=dx, yoff=dy)

            self._transformed = p
            prepare(self._transformed)  # Optimize for repeated operations
            self._dirty = False

    def get_transformed_polygon(self) -> Polygon:
        """Get the current transformed Shapely polygon."""
        self._update_transform()
        return self._transformed

    def get_vertices(self) -> np.ndarray:
        """Get current transformed vertices as Nx2 array."""
        self._update_transform()
        coords = np.array(self._transformed.exterior.coords)
        return coords[:-1]  # Remove duplicate last point

    def intersects(self, other: 'StonePolygon') -> bool:
        """Check if this polygon intersects another."""
        self._update_transform()
        other._update_transform()
        return self._transformed.intersects(other._transformed)

    def intersection_area(self, other: 'StonePolygon') -> float:
        """Get area of intersection with another polygon."""
        self._update_transform()
        other._update_transform()
        intersection = self._transformed.intersection(other._transformed)
        return intersection.area if intersection else 0.0

    def is_within(self, container: Polygon, tolerance: float = 0.0) -> bool:
        """
        Check if polygon is within a container polygon.

        Args:
            container: Container polygon
            tolerance: Allow small violations (in same units as polygon)
        """
        self._update_transform()
        if tolerance > 0:
            buffered = self._transformed.buffer(-tolerance)
            return container.contains(buffered)
        return container.contains(self._transformed)

    def distance_to(self, other: 'StonePolygon') -> float:
        """Get minimum distance to another polygon."""
        self._update_transform()
        other._update_transform()
        return self._transformed.distance(other._transformed)

    def buffer(self, distance: float) -> Polygon:
        """Get buffered (expanded/contracted) version of polygon."""
        self._update_transform()
        return self._transformed.buffer(distance)

    def copy(self) -> 'StonePolygon':
        """Create a copy of this stone polygon."""
        copy = StonePolygon(self.original_vertices.copy(), self.stone_id)
        copy.rotation = self.rotation
        copy.translation = self.translation.copy()
        copy._dirty = True
        return copy


def create_target_polygon(shape: str = "rectangle",
                         width: float = 100.0,
                         height: float = 100.0,
                         **kwargs) -> Polygon:
    """
    Create a target polygon for packing.

    Args:
        shape: "rectangle", "square", "hexagon", "l_shape"
        width: Width in inches
        height: Height in inches
        **kwargs: Additional shape-specific parameters

    Returns:
        Shapely Polygon
    """
    if shape == "rectangle":
        return Polygon([
            (0, 0),
            (width, 0),
            (width, height),
            (0, height)
        ])

    elif shape == "square":
        side = kwargs.get('side', width)
        return Polygon([
            (0, 0),
            (side, 0),
            (side, side),
            (0, side)
        ])

    elif shape == "hexagon":
        # Regular hexagon centered at (width/2, height/2)
        radius = min(width, height) / 2
        cx, cy = width / 2, height / 2
        angles = np.linspace(0, 2 * np.pi, 7)
        points = [(cx + radius * np.cos(a), cy + radius * np.sin(a))
                  for a in angles[:-1]]
        return Polygon(points)

    elif shape == "l_shape":
        # L-shaped polygon
        w1 = kwargs.get('w1', width * 0.6)
        h1 = kwargs.get('h1', height)
        w2 = kwargs.get('w2', width)
        h2 = kwargs.get('h2', height * 0.6)
        return Polygon([
            (0, 0),
            (w1, 0),
            (w1, h2),
            (w2, h2),
            (w2, height),
            (0, height)
        ])

    else:
        raise ValueError(f"Unknown shape: {shape}")


def calculate_gap_distances(placed_stones: List[StonePolygon],
                           sample_points: int = 100) -> List[float]:
    """
    Calculate distances between placed stones (gap widths).

    Args:
        placed_stones: List of placed stone polygons
        sample_points: Number of points to sample per stone

    Returns:
        List of gap distances
    """
    if len(placed_stones) < 2:
        return []

    gaps = []
    for i, stone1 in enumerate(placed_stones):
        for stone2 in placed_stones[i + 1:]:
            dist = stone1.distance_to(stone2)
            if dist > 0:  # Not overlapping
                gaps.append(dist)

    return gaps


def calculate_size_distribution_metric(placed_stones: List[StonePolygon],
                                       grid_size: int = 5) -> float:
    """
    Calculate size distribution metric (lower is better).

    Divides area into grid and checks if each cell has mix of sizes.

    Args:
        placed_stones: List of placed stones
        grid_size: Number of grid cells per dimension

    Returns:
        Distribution metric (0 = perfect distribution, higher = more clustered)
    """
    if not placed_stones:
        return 0.0

    # Get overall bounds
    all_bounds = [s.bounds for s in placed_stones]
    min_x = min(b[0] for b in all_bounds)
    min_y = min(b[1] for b in all_bounds)
    max_x = max(b[2] for b in all_bounds)
    max_y = max(b[3] for b in all_bounds)

    # Categorize stones by size (terciles)
    areas = np.array([s.area for s in placed_stones])
    terciles = np.percentile(areas, [33.33, 66.67])

    size_categories = []
    for stone in placed_stones:
        if stone.area < terciles[0]:
            size_categories.append(0)  # Small
        elif stone.area < terciles[1]:
            size_categories.append(1)  # Medium
        else:
            size_categories.append(2)  # Large

    # Create grid
    cell_width = (max_x - min_x) / grid_size
    cell_height = (max_y - min_y) / grid_size

    # Count size distribution in each cell
    cell_distributions = {}
    for stone, size_cat in zip(placed_stones, size_categories):
        centroid = stone.centroid
        cell_x = int((centroid[0] - min_x) / cell_width)
        cell_y = int((centroid[1] - min_y) / cell_height)
        cell_x = min(cell_x, grid_size - 1)
        cell_y = min(cell_y, grid_size - 1)

        cell_key = (cell_x, cell_y)
        if cell_key not in cell_distributions:
            cell_distributions[cell_key] = [0, 0, 0]
        cell_distributions[cell_key][size_cat] += 1

    # Calculate variance of size distribution
    # Ideal: each cell has equal representation of sizes
    variances = []
    for counts in cell_distributions.values():
        if sum(counts) > 0:
            proportions = np.array(counts) / sum(counts)
            ideal = np.array([1/3, 1/3, 1/3])
            variance = np.sum((proportions - ideal) ** 2)
            variances.append(variance)

    return np.mean(variances) if variances else 0.0
