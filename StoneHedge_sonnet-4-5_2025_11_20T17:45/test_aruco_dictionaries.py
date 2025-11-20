"""Test different ArUco dictionary types to find the correct one."""

import cv2
import numpy as np

def test_all_dictionaries(image_path):
    """Test all available ArUco dictionaries on an image."""

    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not load image: {image_path}")
        return

    # List of common ArUco dictionaries
    dict_types = [
        ("DICT_4X4_50", cv2.aruco.DICT_4X4_50),
        ("DICT_4X4_100", cv2.aruco.DICT_4X4_100),
        ("DICT_4X4_250", cv2.aruco.DICT_4X4_250),
        ("DICT_4X4_1000", cv2.aruco.DICT_4X4_1000),
        ("DICT_5X5_50", cv2.aruco.DICT_5X5_50),
        ("DICT_5X5_100", cv2.aruco.DICT_5X5_100),
        ("DICT_5X5_250", cv2.aruco.DICT_5X5_250),
        ("DICT_5X5_1000", cv2.aruco.DICT_5X5_1000),
        ("DICT_6X6_50", cv2.aruco.DICT_6X6_50),
        ("DICT_6X6_100", cv2.aruco.DICT_6X6_100),
        ("DICT_6X6_250", cv2.aruco.DICT_6X6_250),
        ("DICT_6X6_1000", cv2.aruco.DICT_6X6_1000),
        ("DICT_7X7_50", cv2.aruco.DICT_7X7_50),
        ("DICT_7X7_100", cv2.aruco.DICT_7X7_100),
        ("DICT_7X7_250", cv2.aruco.DICT_7X7_250),
        ("DICT_7X7_1000", cv2.aruco.DICT_7X7_1000),
        ("DICT_ARUCO_ORIGINAL", cv2.aruco.DICT_ARUCO_ORIGINAL),
    ]

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    print(f"Testing image: {image_path}")
    print(f"Image size: {image.shape}\n")

    found_markers = False

    for name, dict_type in dict_types:
        aruco_dict = cv2.aruco.getPredefinedDictionary(dict_type)
        parameters = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

        corners, ids, rejected = detector.detectMarkers(gray)

        if corners is not None and len(corners) > 0:
            print(f"✓ {name}: Found {len(corners)} marker(s)")
            print(f"  IDs: {ids.flatten()}")
            found_markers = True

            # Calculate approximate size
            if len(corners) > 0:
                marker_corners = corners[0][0]
                edge_lengths = []
                for i in range(4):
                    p1 = marker_corners[i]
                    p2 = marker_corners[(i + 1) % 4]
                    length = np.linalg.norm(p2 - p1)
                    edge_lengths.append(length)
                avg_edge = np.mean(edge_lengths)
                print(f"  Average edge length: {avg_edge:.2f} pixels")
                print(f"  Estimated scale: {avg_edge / 5.0:.2f} pixels/inch\n")
        else:
            print(f"✗ {name}: No markers detected")

    if not found_markers:
        print("\n⚠ No markers found with any dictionary!")
        print("Possible issues:")
        print("  - Markers may not be ArUco markers")
        print("  - Image quality/lighting issues")
        print("  - Markers may be partially occluded")
        print("  - Different marker type (QR code, AprilTag, etc.)")

if __name__ == "__main__":
    test_image = "/home/user/research_base/data/stonehedge/aruco_stones/IMG_3411.JPEG"
    test_all_dictionaries(test_image)
