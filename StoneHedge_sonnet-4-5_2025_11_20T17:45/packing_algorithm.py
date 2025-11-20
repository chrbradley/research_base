"""
Irregular polygon packing algorithm for stone layout planning.
"""

import numpy as np
from shapely.geometry import Polygon, Point
from shapely import prepare
from typing import List, Tuple, Dict, Optional
import json
from polygon_utils import StonePolygon, create_target_polygon, calculate_gap_distances, calculate_size_distribution_metric
import math


class StonePacker:
    """Packs irregular stone polygons into a target area."""

    def __init__(self,
                 target_polygon: Polygon,
                 target_gap: float = 1.0,
                 gap_tolerance: float = 0.25,
                 allow_overlap: bool = True,
                 max_overlap_ratio: float = 0.10):
        """
        Initialize the packer.

        Args:
            target_polygon: Target area to fill
            target_gap: Target gap between stones (inches)
            gap_tolerance: Acceptable deviation from target gap
            allow_overlap: Whether to allow stone overlaps (for cutting)
            max_overlap_ratio: Maximum overlap as ratio of stone area
        """
        self.target = target_polygon
        prepare(self.target)  # Optimize for repeated operations

        self.target_gap = target_gap
        self.gap_tolerance = gap_tolerance
        self.allow_overlap = allow_overlap
        self.max_overlap_ratio = max_overlap_ratio

        self.placed_stones: List[StonePolygon] = []
        self.placement_log = []

    def load_stones_from_library(self, library_path: str) -> List[StonePolygon]:
        """
        Load stones from JSON library.

        Args:
            library_path: Path to stone_library.json

        Returns:
            List of StonePolygon objects
        """
        with open(library_path, 'r') as f:
            library = json.load(f)

        stones = []
        for i, stone_data in enumerate(library):
            vertices = np.array(stone_data['polygon_pixels'])
            # Convert from pixels to inches
            ppi = stone_data['pixels_per_inch']
            vertices_inches = vertices / ppi

            stone = StonePolygon(vertices_inches, stone_id=stone_data['image_name'])
            stones.append(stone)

        return stones

    def categorize_stones_by_size(self, stones: List[StonePolygon]) -> Dict[str, List[StonePolygon]]:
        """
        Categorize stones into small/medium/large by area terciles.

        Returns:
            Dictionary with keys 'small', 'medium', 'large'
        """
        areas = np.array([s.area for s in stones])
        terciles = np.percentile(areas, [33.33, 66.67])

        categories = {'small': [], 'medium': [], 'large': []}

        for stone in stones:
            if stone.area < terciles[0]:
                categories['small'].append(stone)
            elif stone.area < terciles[1]:
                categories['medium'].append(stone)
            else:
                categories['large'].append(stone)

        return categories

    def try_placement(self,
                     stone: StonePolygon,
                     x: float,
                     y: float,
                     angle: float) -> Tuple[bool, float]:
        """
        Try placing a stone at given position and rotation.

        Returns:
            Tuple of (is_valid, score)
        """
        # Set position
        stone_copy = stone.copy()
        stone_copy.set_position(x, y, angle)

        # Get transformed polygon
        stone_poly = stone_copy.get_transformed_polygon()

        # Check if within target
        if not self.target.contains(stone_poly):
            # Allow small violations if stone mostly inside
            intersection = self.target.intersection(stone_poly)
            if intersection.area < 0.9 * stone_poly.area:
                return False, 0.0

        # Check collisions with placed stones
        total_overlap = 0.0
        min_gap = float('inf')
        gap_violations = 0

        for placed in self.placed_stones:
            placed_poly = placed.get_transformed_polygon()

            if stone_poly.intersects(placed_poly):
                overlap_area = stone_poly.intersection(placed_poly).area

                if not self.allow_overlap or overlap_area > self.max_overlap_ratio * stone.area:
                    return False, 0.0

                total_overlap += overlap_area
            else:
                # Check gap distance
                gap = stone_poly.distance(placed_poly)
                min_gap = min(min_gap, gap)

                if gap < self.target_gap - self.gap_tolerance:
                    gap_violations += 1
                elif gap > self.target_gap + self.gap_tolerance:
                    gap_violations += 0.5  # Less severe

        # Calculate score (higher is better)
        score = 100.0

        # Penalty for overlaps
        if total_overlap > 0:
            score -= total_overlap * 5.0

        # Penalty for gap violations
        score -= gap_violations * 10.0

        # Bonus for good gap distance
        if min_gap != float('inf'):
            gap_error = abs(min_gap - self.target_gap)
            score -= gap_error * 2.0

        # Bonus for being well within target
        intersection_ratio = self.target.intersection(stone_poly).area / stone_poly.area
        score += intersection_ratio * 10.0

        return True, score

    def find_best_placement(self,
                           stone: StonePolygon,
                           num_positions: int = 50,
                           rotation_steps: int = 12) -> Optional[Tuple[float, float, float, float]]:
        """
        Find best placement for a stone.

        Args:
            stone: Stone to place
            num_positions: Number of positions to try
            rotation_steps: Number of rotation angles to test

        Returns:
            Tuple of (x, y, angle, score) or None
        """
        # Get target bounds
        minx, miny, maxx, maxy = self.target.bounds

        # Generate candidate positions
        # Strategy: Grid-based with some randomness
        grid_size = int(math.sqrt(num_positions))
        positions = []

        # Grid points
        for i in range(grid_size):
            for j in range(grid_size):
                x = minx + (maxx - minx) * (i + 0.5) / grid_size
                y = miny + (maxy - miny) * (j + 0.5) / grid_size
                positions.append((x, y))

        # Add some random positions for diversity
        for _ in range(num_positions - len(positions)):
            x = np.random.uniform(minx, maxx)
            y = np.random.uniform(miny, maxy)
            # Check if point is in target
            if self.target.contains(Point(x, y)):
                positions.append((x, y))

        # Test rotations
        angles = [i * (360.0 / rotation_steps) for i in range(rotation_steps)]

        # Find best placement
        best_score = -float('inf')
        best_placement = None

        for x, y in positions:
            for angle in angles:
                is_valid, score = self.try_placement(stone, x, y, angle)

                if is_valid and score > best_score:
                    best_score = score
                    best_placement = (x, y, angle, score)

        return best_placement

    def pack_stones(self,
                   stones: List[StonePolygon],
                   distribute_sizes: bool = True,
                   verbose: bool = True) -> Dict:
        """
        Pack stones into target area.

        Args:
            stones: List of stones to pack
            distribute_sizes: Whether to distribute sizes evenly
            verbose: Print progress

        Returns:
            Dictionary with packing results and metrics
        """
        if verbose:
            print(f"Packing {len(stones)} stones...")
            print(f"Target area: {self.target.area:.1f} sq inches")
            print(f"Total stone area: {sum(s.area for s in stones):.1f} sq inches")

        # Categorize stones if distributing sizes
        if distribute_sizes:
            categories = self.categorize_stones_by_size(stones)
            if verbose:
                print(f"Size categories: Small={len(categories['small'])}, "
                      f"Medium={len(categories['medium'])}, Large={len(categories['large'])}")

            # Create round-robin sequence
            stone_sequence = []
            max_len = max(len(cat) for cat in categories.values())
            for i in range(max_len):
                for size in ['large', 'medium', 'small']:  # Prioritize large first
                    if i < len(categories[size]):
                        stone_sequence.append(categories[size][i])
        else:
            # Sort by area (largest first)
            stone_sequence = sorted(stones, key=lambda s: s.area, reverse=True)

        # Place stones
        placed_count = 0
        failed_count = 0

        for i, stone in enumerate(stone_sequence):
            if verbose and (i + 1) % 5 == 0:
                print(f"  Placing stone {i + 1}/{len(stone_sequence)}...")

            placement = self.find_best_placement(stone, num_positions=30, rotation_steps=12)

            if placement:
                x, y, angle, score = placement
                stone_copy = stone.copy()
                stone_copy.set_position(x, y, angle)
                self.placed_stones.append(stone_copy)

                self.placement_log.append({
                    'stone_id': stone.stone_id,
                    'position': (x, y),
                    'rotation': angle,
                    'score': score,
                    'area': stone.area
                })

                placed_count += 1
            else:
                failed_count += 1
                if verbose:
                    print(f"  ⚠ Could not place stone {stone.stone_id}")

        # Calculate metrics
        metrics = self.calculate_metrics()

        if verbose:
            print(f"\nPacking complete!")
            print(f"  Placed: {placed_count}/{len(stones)}")
            print(f"  Failed: {failed_count}")
            print(f"  Coverage: {metrics['coverage_ratio']*100:.1f}%")
            print(f"  Avg gap: {metrics['avg_gap']:.2f}\" (target: {self.target_gap}\")")
            print(f"  Size distribution metric: {metrics['size_distribution']:.3f}")

        return {
            'placed_count': placed_count,
            'failed_count': failed_count,
            'metrics': metrics,
            'placement_log': self.placement_log
        }

    def calculate_metrics(self) -> Dict:
        """Calculate packing quality metrics."""
        if not self.placed_stones:
            return {}

        # Coverage ratio
        total_placed_area = sum(s.area for s in self.placed_stones)
        coverage_ratio = total_placed_area / self.target.area

        # Gap statistics
        gaps = calculate_gap_distances(self.placed_stones)
        avg_gap = np.mean(gaps) if gaps else 0.0
        std_gap = np.std(gaps) if gaps else 0.0

        # Size distribution
        size_dist_metric = calculate_size_distribution_metric(self.placed_stones, grid_size=5)

        # Count overlaps
        overlap_count = 0
        total_overlap_area = 0.0
        for i, stone1 in enumerate(self.placed_stones):
            for stone2 in self.placed_stones[i + 1:]:
                if stone1.intersects(stone2):
                    overlap_count += 1
                    total_overlap_area += stone1.intersection_area(stone2)

        return {
            'coverage_ratio': coverage_ratio,
            'avg_gap': avg_gap,
            'std_gap': std_gap,
            'size_distribution': size_dist_metric,
            'overlap_count': overlap_count,
            'total_overlap_area': total_overlap_area
        }


def test_packing(target_area_sq_ft: float = 100.0,
                shape: str = "rectangle",
                library_path: str = "stone_library.json"):
    """
    Test packing algorithm with a target area.

    Args:
        target_area_sq_ft: Target area in square feet
        shape: Target shape
        library_path: Path to stone library
    """
    print(f"Testing packing with {target_area_sq_ft} sq ft {shape}")

    # Convert to inches
    target_area_sq_in = target_area_sq_ft * 144

    # Create target polygon
    if shape == "rectangle":
        # Make it roughly 2:1 aspect ratio
        width = math.sqrt(target_area_sq_in * 2)
        height = target_area_sq_in / width
    else:
        side = math.sqrt(target_area_sq_in)
        width = height = side

    target = create_target_polygon(shape, width, height)
    print(f"Target dimensions: {width:.1f}\" x {height:.1f}\"")

    # Create packer
    packer = StonePacker(target, target_gap=1.0, gap_tolerance=0.25)

    # Load stones
    stones = packer.load_stones_from_library(library_path)
    print(f"Loaded {len(stones)} stones from library")

    # Pack stones
    results = packer.pack_stones(stones, distribute_sizes=True)

    return packer, results


if __name__ == "__main__":
    # Test with 100 sq ft rectangle
    packer, results = test_packing(target_area_sq_ft=100.0, shape="rectangle")
