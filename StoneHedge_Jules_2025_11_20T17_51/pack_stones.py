import json
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, box
from shapely import affinity
import random
import math

# Config
STONES_FILE = "stones.json"
OUTPUT_IMAGE = "layout.png"
TARGET_WIDTH = 80 # inches, increased to fit large stones better
TARGET_HEIGHT = 60 # inches
GAP = 1.0 # inch
ATTEMPTS_PER_STONE = 1000

class StonePacker:
    def __init__(self, width, height, gap):
        self.width = width
        self.height = height
        self.gap = gap
        self.target_poly = box(0, 0, width, height)
        self.placed_stones = [] # List of Polygons

    def load_stones(self, filename):
        with open(filename, 'r') as f:
            data = json.load(f)

        stones = []
        for s in data:
            # Create shapely polygon
            # Valid stones only
            if len(s['vertices']) < 3: continue
            poly = Polygon(s['vertices'])
            # Center the polygon at (0,0) for easier manipulation
            centroid = poly.centroid
            poly = affinity.translate(poly, -centroid.x, -centroid.y)
            stones.append({
                'poly': poly,
                'area': poly.area,
                'original_data': s
            })

        # Sort by area descending
        stones.sort(key=lambda x: x['area'], reverse=True)
        return stones

    def is_valid(self, stone_poly):
        # Check bounds
        if not self.target_poly.contains(stone_poly):
            return False

        # Check overlap with existing stones (considering gap)
        # Buffer the new stone by gap/2 and existing stones by gap/2?
        # Or just buffer new stone by gap and check intersection with existing unbuffered?
        # Effective gap is sum of buffers.
        # Let's check distance.
        # stone_poly.distance(other) >= gap

        for placed in self.placed_stones:
            if stone_poly.distance(placed) < self.gap:
                return False
        return True

    def pack(self, stones):
        print(f"Packing {len(stones)} stones into {self.width}x{self.height} area...")

        for i, stone in enumerate(stones):
            poly = stone['poly']
            best_poly = None
            # Scoring: for first 30% stones, prefer edges. For rest, pack tight.
            is_anchor = i < (len(stones) * 0.4)

            best_score = -float('inf') if is_anchor else float('inf')

            # Monte Carlo attempts
            for _ in range(ATTEMPTS_PER_STONE):
                # Random rotation
                angle = random.uniform(0, 360)
                rotated = affinity.rotate(poly, angle)

                # Random position
                minx, miny, maxx, maxy = rotated.bounds
                # Range for translation
                # We want bounds within 0..W, 0..H
                # center is at 0,0. bounds are relative to center.
                # We need to shift so minx >= 0, maxx <= W
                # But simpler: pick random x,y in Target and check containment.

                rand_x = random.uniform(0, self.width)
                rand_y = random.uniform(0, self.height)

                candidate = affinity.translate(rotated, rand_x, rand_y)

                if self.is_valid(candidate):
                    # Calculate score
                    centroid = candidate.centroid
                    # Distance to center
                    dist_center = math.sqrt((centroid.x - self.width/2)**2 + (centroid.y - self.height/2)**2)

                    if is_anchor:
                        # Maximize distance to center (push to edges)
                        score = dist_center
                        if score > best_score:
                            best_score = score
                            best_poly = candidate
                    else:
                        # Minimize distance to nearest neighbor or center?
                        # Let's try: minimize distance to "placed stones centroid" or just closest stone.
                        # To pack tight, we want to minimize distance to existing structure.
                        # Let's just minimize distance to center to fill inwards.
                        score = dist_center
                        if score < best_score:
                            best_score = score
                            best_poly = candidate

            if best_poly:
                self.placed_stones.append(best_poly)
                print(f"  Placed stone {i} (Area: {stone['area']:.1f})")
            else:
                print(f"  Failed to place stone {i} (Area: {stone['area']:.1f})")

    def visualize(self, filename):
        fig, ax = plt.subplots(figsize=(10, 10 * self.height / self.width))
        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)
        ax.set_aspect('equal')

        # Draw Target
        target_x, target_y = self.target_poly.exterior.xy
        ax.plot(target_x, target_y, 'k-', linewidth=2)

        # Draw Stones
        for poly in self.placed_stones:
            x, y = poly.exterior.xy
            ax.fill(x, y, alpha=0.5, fc='gray', ec='black')

        plt.title(f"StoneHedge Layout (Gap={self.gap}\")")
        plt.savefig(filename)
        print(f"Saved layout to {filename}")

if __name__ == "__main__":
    packer = StonePacker(TARGET_WIDTH, TARGET_HEIGHT, GAP)
    stones = packer.load_stones(STONES_FILE)
    packer.pack(stones)
    packer.visualize(OUTPUT_IMAGE)
