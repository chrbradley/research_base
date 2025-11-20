"""
Demonstrate packing algorithm with multiple target sizes and shapes.
"""

import numpy as np
from packing_algorithm import StonePacker, create_target_polygon
from visualization import LayoutVisualizer
import math


def run_comprehensive_demo():
    """Run comprehensive demo with multiple target configurations."""

    print("="*70)
    print("STONEHEDGE PACKING ALGORITHM DEMO")
    print("="*70)

    # Load stone library
    library_path = "stone_library.json"

    # Test configurations
    configs = [
        {"area_sq_ft": 100, "shape": "rectangle", "desc": "100 sq ft Rectangle (Walkway)"},
        {"area_sq_ft": 150, "shape": "rectangle", "desc": "150 sq ft Rectangle (Large Walkway)"},
        {"area_sq_ft": 200, "shape": "square", "desc": "200 sq ft Square (Patio)"},
    ]

    results = []
    visualizer = LayoutVisualizer(figsize=(14, 10))

    for i, config in enumerate(configs, 1):
        print(f"\n{'-'*70}")
        print(f"Test {i}/{len(configs)}: {config['desc']}")
        print(f"{'-'*70}")

        # Convert to inches
        area_sq_in = config['area_sq_ft'] * 144

        # Create target
        if config['shape'] == "rectangle":
            width = math.sqrt(area_sq_in * 2)
            height = area_sq_in / width
        else:
            side = math.sqrt(area_sq_in)
            width = height = side

        target = create_target_polygon(config['shape'], width, height)

        # Create packer
        packer = StonePacker(target, target_gap=1.0, gap_tolerance=0.25)

        # Load stones
        stones = packer.load_stones_from_library(library_path)

        # Pack
        pack_results = packer.pack_stones(stones, distribute_sizes=True, verbose=True)

        # Visualize
        title = (f"{config['desc']}\n"
                f"Coverage: {pack_results['metrics']['coverage_ratio']*100:.1f}% | "
                f"Placed: {pack_results['placed_count']}/{len(stones)} | "
                f"Avg Gap: {pack_results['metrics']['avg_gap']:.1f}\"")

        visualizer.plot_layout(
            target,
            packer.placed_stones,
            show_gaps=False,
            show_grid=True,
            title=title,
            save_path=f"layout_{config['area_sq_ft']}sqft_{config['shape']}.png"
        )

        # Store for comparison
        results.append({
            'target': target,
            'placed_stones': packer.placed_stones,
            'title': f"{config['area_sq_ft']} sq ft {config['shape']}\n"
                    f"{pack_results['placed_count']}/{len(stones)} stones",
            'config': config,
            'results': pack_results
        })

    # Create comparison plot
    print(f"\n{'-'*70}")
    print("Creating comparison visualization...")
    print(f"{'-'*70}")
    visualizer.plot_comparison(results, save_path="layout_comparison.png")

    # Print summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"{'Configuration':<40} {'Placed':<10} {'Coverage':<10} {'Avg Gap'}")
    print(f"{'-'*70}")
    for r in results:
        config_str = r['config']['desc']
        placed = f"{r['results']['placed_count']}/{len(stones)}"
        coverage = f"{r['results']['metrics']['coverage_ratio']*100:.1f}%"
        avg_gap = f"{r['results']['metrics']['avg_gap']:.1f}\""
        print(f"{config_str:<40} {placed:<10} {coverage:<10} {avg_gap}")

    print(f"{'='*70}")
    print("\nAll layouts saved to PNG files.")
    print("Stone library covers ~185 sq ft total.")


if __name__ == "__main__":
    run_comprehensive_demo()
