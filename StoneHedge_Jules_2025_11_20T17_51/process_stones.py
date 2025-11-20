import cv2
import numpy as np
import os
import json
from ultralytics import SAM
from shapely.geometry import Polygon
import time

# Config
IMAGE_DIR = "../data/stonehedge/aruco_stones"
# List all images you want to try.
# We will process them one by one and skip if already done.
SELECTED_IMAGES = ["IMG_3411.JPEG", "IMG_3412.JPEG", "IMG_3413.JPEG", "IMG_3414.JPEG"]
OUTPUT_FILE = "stones.json"
DEBUG_DIR = "debug_process"
os.makedirs(DEBUG_DIR, exist_ok=True)

def get_homography(img):
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_50)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners, ids, rejected = detector.detectMarkers(gray)

    if ids is None or len(ids) == 0:
        return None, None

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

def process_images():
    # Load existing library
    stone_library = []
    if os.path.exists(OUTPUT_FILE):
        try:
            with open(OUTPUT_FILE, 'r') as f:
                stone_library = json.load(f)
            print(f"Loaded {len(stone_library)} existing stones.")
        except:
            print("Could not load existing library, starting fresh.")

    processed_files = set(s['source_image'] for s in stone_library)

    # Load Model
    print("Loading Model...")
    cwd = os.getcwd()
    os.chdir('/tmp')
    try:
        model = SAM('mobile_sam.pt')
    except Exception:
        import torch
        torch.hub.download_url_to_file("https://github.com/ultralytics/assets/releases/download/v8.3.0/mobile_sam.pt", "mobile_sam.pt")
        model = SAM('mobile_sam.pt')
    os.chdir(cwd)

    for img_name in SELECTED_IMAGES:
        if img_name in processed_files:
            print(f"Skipping {img_name}, already processed.")
            continue

        img_path = os.path.join(IMAGE_DIR, img_name)
        print(f"Processing {img_name}...")

        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to read {img_path}")
            continue

        H, marker_area_px = get_homography(img)
        if H is None:
            print(f"No ArUco marker found in {img_name}, skipping.")
            continue

        print(f"  Marker found. Area: {marker_area_px} px^2. Running segmentation...")

        results = model(img_path, verbose=False)

        if not results[0].masks:
            print("  No masks found.")
            continue

        segments = results[0].masks.xy
        stones_in_img = 0
        debug_img = img.copy()

        for seg in segments:
            if len(seg) < 3: continue

            poly_px = Polygon(seg)
            if poly_px.area < (marker_area_px * 0.5):
                continue

            real_poly_pts = transform_polygon(seg, H)
            real_poly = Polygon(real_poly_pts)

            if real_poly.area > 7200: continue

            marker_box = Polygon([(0,0), (5,0), (5,5), (0,5)])
            if real_poly.intersects(marker_box) and real_poly.area < 50:
                 if real_poly.intersection(marker_box).area > 10:
                     continue

            stone_entry = {
                "source_image": img_name,
                "area_sq_in": real_poly.area,
                "perimeter_in": real_poly.length,
                "vertices": real_poly_pts
            }
            stone_library.append(stone_entry)
            stones_in_img += 1

            pts = np.array(seg, np.int32).reshape((-1, 1, 2))
            cv2.polylines(debug_img, [pts], True, (0, 255, 0), 3)

        print(f"  Found {stones_in_img} valid stones.")
        cv2.imwrite(f"{DEBUG_DIR}/debug_{img_name}", debug_img)

        # Save incrementally
        with open(OUTPUT_FILE, 'w') as f:
            json.dump(stone_library, f, indent=2)
        print(f"  Saved progress to {OUTPUT_FILE}")

    print(f"Finished. Total stones: {len(stone_library)}")

if __name__ == "__main__":
    process_images()
