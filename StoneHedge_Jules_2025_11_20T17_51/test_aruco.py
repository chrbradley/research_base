import cv2
import cv2.aruco as aruco
import numpy as np
import os

# Path to test image
IMAGE_PATH = "../data/stonehedge/aruco_stones/IMG_3411.JPEG"
OUTPUT_DIR = "output_debug"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def test_aruco_dicts():
    img = cv2.imread(IMAGE_PATH)
    if img is None:
        print(f"Error: Could not read image {IMAGE_PATH}")
        return

    # Resize for speed/debugging
    scale_percent = 50
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    img_resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

    aruco_dicts = {
        "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
        "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
        "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
        "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
        "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
        "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL
    }

    for name, dict_id in aruco_dicts.items():
        dictionary = cv2.aruco.getPredefinedDictionary(dict_id)
        parameters = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(dictionary, parameters)

        corners, ids, rejected = detector.detectMarkers(gray)

        if ids is not None:
            print(f"Found {len(ids)} markers with {name}")
            print(f"IDs: {ids.flatten()}")

            # Draw markers
            debug_img = img_resized.copy()
            cv2.aruco.drawDetectedMarkers(debug_img, corners, ids)
            cv2.imwrite(f"{OUTPUT_DIR}/aruco_test_{name}.jpg", debug_img)

            # Calculate pixel scale (assuming 5x5 inches)
            # corners is list of (1, 4, 2)
            for i in range(len(ids)):
                c = corners[i][0] # 4 points
                # Perimeter in pixels
                perimeter = cv2.arcLength(c, True)
                # A 5x5 square has perimeter 20 inches.
                # But perspective might distort.
                # Better: average side length.
                side_lengths = [np.linalg.norm(c[j] - c[(j+1)%4]) for j in range(4)]
                avg_side_px = np.mean(side_lengths)
                px_per_inch = avg_side_px / 5.0
                print(f"  Marker {ids[i][0]}: Avg Side = {avg_side_px:.2f} px, Scale = {px_per_inch:.2f} px/inch (Resized 50%)")

        else:
            pass
            # print(f"No markers found with {name}")

if __name__ == "__main__":
    test_aruco_dicts()
