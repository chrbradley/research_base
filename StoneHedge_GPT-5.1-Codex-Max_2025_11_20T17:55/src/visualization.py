"""Matplotlib rendering helpers for layouts."""

from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from shapely.geometry import Polygon

from .packing import Placement


COLORS = ["#e76f51", "#f4a261", "#2a9d8f", "#264653", "#8ab17d", "#577590", "#f9844a"]


def _color(idx: int) -> str:
    return COLORS[idx % len(COLORS)]


def plot_layout(target: Polygon, placements: Sequence[Placement], gap_inches: float = 1.0) -> None:
    fig, ax = plt.subplots(figsize=(8, 8))

    # Target boundary
    x, y = target.exterior.xy
    ax.plot(x, y, color="black", linewidth=2, label="Target boundary")

    for idx, placement in enumerate(placements):
        px, py = placement.shape.exterior.xy
        polygon = MplPolygon(list(zip(px, py)), closed=True, facecolor=_color(idx), alpha=0.55, edgecolor="black")
        ax.add_patch(polygon)
        ax.text(placement.shape.centroid.x, placement.shape.centroid.y, str(idx), fontsize=8, ha="center")

    ax.set_aspect("equal")
    ax.set_title(f"Layout with ~{gap_inches}\" gaps")
    ax.set_xlabel("Inches")
    ax.set_ylabel("Inches")
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.show()
