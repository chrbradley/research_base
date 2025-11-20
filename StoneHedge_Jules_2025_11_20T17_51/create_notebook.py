import nbformat as nbf
import os

def create_notebook():
    nb = nbf.v4.new_notebook()

    # Title
    nb.cells.append(nbf.v4.new_markdown_cell(
        "# StoneHedge Prototype\n"
        "Automated stone patio layout planning using Computer Vision and AI.\n"
        "\n"
        "## Workflow\n"
        "1. **Segmentation**: Extract stone polygons from images using MobileSAM.\n"
        "2. **Calibration**: Use ArUco markers to scale polygons to real-world inches.\n"
        "3. **Packing**: Pack irregular stone polygons into a target area.\n"
    ))

    # Imports
    nb.cells.append(nbf.v4.new_code_cell(
        "import cv2\n"
        "import numpy as np\n"
        "import matplotlib.pyplot as plt\n"
        "from ultralytics import SAM\n"
        "from shapely.geometry import Polygon, box\n"
        "from shapely import affinity\n"
        "import random\n"
        "import math\n"
        "import os\n"
        "import json\n"
        "\n"
        "# Helper to display images in notebook\n"
        "def show_img(img, title=''):\n"
        "    plt.figure(figsize=(10,10))\n"
        "    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))\n"
        "    plt.title(title)\n"
        "    plt.axis('off')\n"
        "    plt.show()"
    ))

    # ArUco & Calibration
    nb.cells.append(nbf.v4.new_markdown_cell("## 1. Calibration & Segmentation"))
    nb.cells.append(nbf.v4.new_code_cell(
        """
def get_homography(img):
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_50)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners, ids, rejected = detector.detectMarkers(gray)

    if ids is None or len(ids) == 0:
        return None, None

    # Marker is 5x5 inches
    src_pts = corners[0][0]
    dst_pts = np.array([[0, 0], [5, 0], [5, 5], [0, 5]], dtype=np.float32)

    H, mask = cv2.findHomography(src_pts, dst_pts)
    marker_area_px = cv2.contourArea(src_pts)

    return H, marker_area_px

def transform_polygon(polygon, H):
    if len(polygon) == 0: return []
    pts = np.array(polygon, dtype=np.float32).reshape(-1, 1, 2)
    transformed_pts = cv2.perspectiveTransform(pts, H)
    return transformed_pts.reshape(-1, 2).tolist()
"""
    ))

    # Processing Loop
    nb.cells.append(nbf.v4.new_code_cell(
        """
# Load MobileSAM model
# Ensure mobile_sam.pt is available
try:
    model = SAM('mobile_sam.pt')
except:
    print("Please download mobile_sam.pt")

# Process a sample image
img_path = "../data/stonehedge/aruco_stones/IMG_3411.JPEG"
if os.path.exists(img_path):
    img = cv2.imread(img_path)
    H, marker_area = get_homography(img)

    if H is not None:
        print("Marker detected.")
        # Segment
        results = model(img_path, verbose=False)
        segments = results[0].masks.xy

        valid_stones = []
        debug_img = img.copy()

        for seg in segments:
            if len(seg) < 3: continue
            poly_px = Polygon(seg)

            # Filter small noise
            if poly_px.area < (marker_area * 0.5): continue

            # Transform
            real_pts = transform_polygon(seg, H)
            real_poly = Polygon(real_pts)

            # Filter huge background or marker
            if real_poly.area > 7200: continue
            marker_box = Polygon([(0,0), (5,0), (5,5), (0,5)])
            if real_poly.intersects(marker_box) and real_poly.area < 50: continue

            valid_stones.append(real_poly)

            # Draw
            pts = np.array(seg, np.int32).reshape((-1, 1, 2))
            cv2.polylines(debug_img, [pts], True, (0, 255, 0), 5)

        print(f"Found {len(valid_stones)} stones.")
        show_img(debug_img, "Segmented Stones")
    else:
        print("No marker found.")
"""
    ))

    # Packing Logic
    nb.cells.append(nbf.v4.new_markdown_cell("## 2. Irregular Packing"))
    nb.cells.append(nbf.v4.new_code_cell(
        """
class StonePacker:
    def __init__(self, width, height, gap):
        self.width = width
        self.height = height
        self.gap = gap
        self.target_poly = box(0, 0, width, height)
        self.placed_stones = []

    def is_valid(self, stone_poly):
        if not self.target_poly.contains(stone_poly):
            return False
        for placed in self.placed_stones:
            if stone_poly.distance(placed) < self.gap:
                return False
        return True

    def pack(self, stones, attempts=200):
        # stones: list of Polygon objects
        # Sort by area
        stones = sorted(stones, key=lambda p: p.area, reverse=True)

        for i, poly in enumerate(stones):
            # Center polygon
            centroid = poly.centroid
            poly = affinity.translate(poly, -centroid.x, -centroid.y)

            best_poly = None
            # Heuristic: large stones near edges
            is_anchor = i < (len(stones) * 0.4)
            best_score = -float('inf') if is_anchor else float('inf')

            for _ in range(attempts):
                angle = random.uniform(0, 360)
                rotated = affinity.rotate(poly, angle)

                rand_x = random.uniform(0, self.width)
                rand_y = random.uniform(0, self.height)
                candidate = affinity.translate(rotated, rand_x, rand_y)

                if self.is_valid(candidate):
                    cent = candidate.centroid
                    dist = math.sqrt((cent.x - self.width/2)**2 + (cent.y - self.height/2)**2)

                    if is_anchor:
                        score = dist # Maximize dist to center
                        if score > best_score:
                            best_score = score
                            best_poly = candidate
                    else:
                        score = dist # Minimize dist to center
                        if score < best_score:
                            best_score = score
                            best_poly = candidate

            if best_poly:
                self.placed_stones.append(best_poly)
                print(f"Placed stone {i} (Area: {poly.area:.1f})")
            else:
                print(f"Failed to place stone {i}")

    def visualize(self):
        fig, ax = plt.subplots(figsize=(10, 10 * self.height / self.width))
        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)
        ax.set_aspect('equal')

        target_x, target_y = self.target_poly.exterior.xy
        ax.plot(target_x, target_y, 'k-', linewidth=2)

        for poly in self.placed_stones:
            x, y = poly.exterior.xy
            ax.fill(x, y, alpha=0.6, fc='gray', ec='black')
        plt.show()
"""
    ))

    # Run Packing
    nb.cells.append(nbf.v4.new_code_cell(
        """
# Use the valid stones detected earlier
if 'valid_stones' in locals() and len(valid_stones) > 0:
    packer = StonePacker(80, 60, gap=1.0)
    packer.pack(valid_stones, attempts=500)
    packer.visualize()
else:
    print("Run segmentation first.")
"""
    ))

    with open('StoneHedge_Prototype.ipynb', 'w') as f:
        nbf.write(nb, f)
    print("Created StoneHedge_Prototype.ipynb")

if __name__ == "__main__":
    create_notebook()
