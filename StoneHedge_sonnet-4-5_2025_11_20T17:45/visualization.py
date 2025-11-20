"""
Visualization module for stone packing layouts.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
import numpy as np
from shapely.geometry import Polygon
from typing import List, Optional
import cv2
from polygon_utils import StonePolygon


class LayoutVisualizer:
    """Visualizes stone packing layouts."""

    def __init__(self, figsize=(16, 12), dpi=100):
        """
        Initialize visualizer.

        Args:
            figsize: Figure size in inches
            dpi: Resolution
        """
        self.figsize = figsize
        self.dpi = dpi

    def plot_layout(self,
                   target: Polygon,
                   placed_stones: List[StonePolygon],
                   show_gaps: bool = True,
                   show_grid: bool = True,
                   show_labels: bool = False,
                   title: str = "Stone Layout",
                   save_path: Optional[str] = None):
        """
        Plot the packing layout.

        Args:
            target: Target polygon
            placed_stones: List of placed stones
            show_gaps: Highlight gaps between stones
            show_grid: Show background grid
            show_labels: Label stones with IDs
            title: Plot title
            save_path: Path to save figure
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        # Plot target area
        target_coords = np.array(target.exterior.coords)
        ax.plot(target_coords[:, 0], target_coords[:, 1],
               'k-', linewidth=2, label='Target Area')

        # Generate colors for stones (by size)
        if placed_stones:
            areas = np.array([s.area for s in placed_stones])
            terciles = np.percentile(areas, [33.33, 66.67])

            # Plot stones
            for i, stone in enumerate(placed_stones):
                poly = stone.get_transformed_polygon()

                # Handle MultiPolygon case
                from shapely.geometry import MultiPolygon
                if isinstance(poly, MultiPolygon):
                    # Take the largest polygon
                    poly = max(poly.geoms, key=lambda p: p.area)

                coords = np.array(poly.exterior.coords)

                # Color by size
                if stone.area < terciles[0]:
                    color = '#87CEEB'  # Light blue (small)
                    label = 'Small' if i == 0 else ''
                elif stone.area < terciles[1]:
                    color = '#4682B4'  # Steel blue (medium)
                    label = 'Medium' if i == 0 else ''
                else:
                    color = '#191970'  # Midnight blue (large)
                    label = 'Large' if i == 0 else ''

                # Plot filled polygon
                ax.fill(coords[:, 0], coords[:, 1], color=color, alpha=0.6,
                       edgecolor='black', linewidth=1.5, label=label)

                # Plot vertices
                if show_labels:
                    centroid = stone.centroid
                    ax.text(centroid[0], centroid[1], f"{i}",
                           ha='center', va='center', fontsize=6,
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

            # Show gaps if requested
            if show_gaps:
                self._plot_gaps(ax, placed_stones)

        # Grid
        if show_grid:
            ax.grid(True, alpha=0.3, linestyle='--')

        # Labels and formatting
        ax.set_xlabel('Width (inches)', fontsize=12)
        ax.set_ylabel('Height (inches)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_aspect('equal')

        # Legend - remove duplicates
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper right')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"Saved visualization: {save_path}")

        return fig, ax

    def _plot_gaps(self, ax, placed_stones: List[StonePolygon]):
        """Plot gap lines between stones."""
        # This is computationally expensive for large layouts
        # Sample a few gap lines for visualization
        max_gaps_to_show = 50
        gap_count = 0

        for i, stone1 in enumerate(placed_stones):
            if gap_count >= max_gaps_to_show:
                break

            for stone2 in placed_stones[i + 1:]:
                if gap_count >= max_gaps_to_show:
                    break

                dist = stone1.distance_to(stone2)
                if dist > 0 and dist < 5.0:  # Only show small gaps
                    # Find closest points
                    poly1 = stone1.get_transformed_polygon()
                    poly2 = stone2.get_transformed_polygon()

                    # Simplified: draw line between centroids
                    c1 = stone1.centroid
                    c2 = stone2.centroid
                    ax.plot([c1[0], c2[0]], [c1[1], c2[1]],
                           'r--', linewidth=0.5, alpha=0.3)
                    gap_count += 1

    def plot_with_textures(self,
                          target: Polygon,
                          placed_stones: List[StonePolygon],
                          stone_images: dict,
                          save_path: Optional[str] = None):
        """
        Plot layout with actual stone textures mapped onto polygons.

        Args:
            target: Target polygon
            placed_stones: Placed stones
            stone_images: Dict mapping stone_id to image path
            save_path: Path to save image
        """
        # Get bounds
        minx, miny, maxx, maxy = target.bounds
        width_inches = maxx - minx
        height_inches = maxy - miny

        # Create high-res canvas (e.g., 10 pixels per inch)
        ppi = 10
        canvas_width = int(width_inches * ppi)
        canvas_height = int(height_inches * ppi)

        # Initialize canvas (white background)
        canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255

        # Draw each stone with texture
        for stone in placed_stones:
            poly = stone.get_transformed_polygon()
            coords = np.array(poly.exterior.coords)

            # Convert to pixel coordinates
            coords_px = coords.copy()
            coords_px[:, 0] = (coords_px[:, 0] - minx) * ppi
            coords_px[:, 1] = (height_inches - (coords_px[:, 1] - miny)) * ppi  # Flip Y

            # Load stone image if available
            if stone.stone_id in stone_images:
                try:
                    stone_img = cv2.imread(stone_images[stone.stone_id])
                    if stone_img is not None:
                        # TODO: Warp stone texture onto polygon
                        # For now, just fill with a color based on stone ID
                        pass
                except:
                    pass

            # Fill polygon on canvas (for now, use random color)
            color = tuple(np.random.randint(100, 200, 3).tolist())
            cv2.fillPoly(canvas, [coords_px.astype(np.int32)], color)

            # Draw outline
            cv2.polylines(canvas, [coords_px.astype(np.int32)], True, (0, 0, 0), 2)

        # Draw target boundary
        target_coords = np.array(target.exterior.coords)
        target_coords_px = target_coords.copy()
        target_coords_px[:, 0] = (target_coords_px[:, 0] - minx) * ppi
        target_coords_px[:, 1] = (height_inches - (target_coords_px[:, 1] - miny)) * ppi
        cv2.polylines(canvas, [target_coords_px.astype(np.int32)], True, (0, 0, 0), 3)

        # Convert BGR to RGB for matplotlib
        canvas_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

        # Display/save
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        ax.imshow(canvas_rgb, origin='upper')
        ax.axis('off')
        ax.set_title('Stone Layout with Textures', fontsize=14, fontweight='bold')

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"Saved textured visualization: {save_path}")

        return fig, ax

    def plot_comparison(self,
                       results: List[dict],
                       save_path: Optional[str] = None):
        """
        Plot comparison of multiple packing results.

        Args:
            results: List of dicts with 'target', 'placed_stones', 'title'
            save_path: Path to save figure
        """
        n = len(results)
        cols = min(n, 3)
        rows = (n + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 5), dpi=self.dpi)
        if n == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        for i, result in enumerate(results):
            ax = axes[i]
            target = result['target']
            placed_stones = result['placed_stones']
            title = result.get('title', f'Layout {i+1}')

            # Plot target
            target_coords = np.array(target.exterior.coords)
            ax.plot(target_coords[:, 0], target_coords[:, 1], 'k-', linewidth=2)

            # Plot stones
            if placed_stones:
                areas = np.array([s.area for s in placed_stones])
                terciles = np.percentile(areas, [33.33, 66.67])

                for stone in placed_stones:
                    poly = stone.get_transformed_polygon()

                    # Handle MultiPolygon case
                    from shapely.geometry import MultiPolygon
                    if isinstance(poly, MultiPolygon):
                        poly = max(poly.geoms, key=lambda p: p.area)

                    coords = np.array(poly.exterior.coords)

                    if stone.area < terciles[0]:
                        color = '#87CEEB'
                    elif stone.area < terciles[1]:
                        color = '#4682B4'
                    else:
                        color = '#191970'

                    ax.fill(coords[:, 0], coords[:, 1], color=color, alpha=0.6,
                           edgecolor='black', linewidth=1)

            ax.set_aspect('equal')
            ax.set_title(title, fontsize=10, fontweight='bold')
            ax.grid(True, alpha=0.3)

        # Hide unused subplots
        for i in range(n, len(axes)):
            axes[i].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"Saved comparison: {save_path}")

        return fig, axes


def test_visualization():
    """Test visualization with a simple packing."""
    from packing_algorithm import test_packing

    print("Running packing test...")
    packer, results = test_packing(target_area_sq_ft=100.0, shape="rectangle")

    print("\nCreating visualization...")
    visualizer = LayoutVisualizer()

    visualizer.plot_layout(
        packer.target,
        packer.placed_stones,
        show_gaps=True,
        show_grid=True,
        title=f"Stone Layout - 100 sq ft Rectangle\n"
              f"Coverage: {results['metrics']['coverage_ratio']*100:.1f}% | "
              f"Placed: {results['placed_count']}/{results['placed_count'] + results['failed_count']}",
        save_path="layout_100sqft.png"
    )

    print("\nVisualization complete!")


if __name__ == "__main__":
    test_visualization()
