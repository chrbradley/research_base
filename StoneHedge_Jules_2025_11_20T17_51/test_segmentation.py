import cv2
import numpy as np
import os
from ultralytics import SAM
import matplotlib.pyplot as plt

# Path to test image
IMAGE_PATH = "../data/stonehedge/aruco_stones/IMG_3411.JPEG"
OUTPUT_DIR = "output_debug"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def test_segmentation():
    print("Loading SAM model...")
    # Download mobile_sam.pt to /tmp to avoid repo size limits
    model_path = '/tmp/mobile_sam.pt'
    if not os.path.exists(model_path):
        print(f"Downloading mobile_sam.pt to {model_path}...")
        # URL for MobileSAM weights (original repo)
        # url = "https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt"
        # Ultralytics might use a different URL or expect a different format?
        # Let's try letting Ultralytics download it by changing CWD to /tmp momentarily?
        # Or better, just use the Ultralytics download logic if possible.
        # But simpler: just try to use it.
        pass

    # Try using a small model first. Ultralytics supports 'mobile_sam.pt'
    # We will point to /tmp/mobile_sam.pt if it exists, otherwise we try to let it download there.

    # Actually, Ultralytics downloads to the current directory by default.
    # We can force it by copying to /tmp manually if we had it?
    # Let's try to download it manually first.
    import torch
    url = "https://github.com/ultralytics/assets/releases/download/v8.2.0/mobile_sam.pt" # Guessed URL or standard asset
    # Let's use a known valid URL for mobile_sam.pt compatible with Ultralytics.
    # Ultralytics usually hosts it.

    # Let's try to run SAM with the path.
    # If we give 'mobile_sam.pt', it checks local, then downloads.
    # We want to download to /tmp.

    # Hack: change cwd to /tmp, init model, change back.
    cwd = os.getcwd()
    os.chdir('/tmp')
    try:
        model = SAM('mobile_sam.pt')
        print("Loaded mobile_sam.pt from /tmp")
    except Exception as e:
        print(f"Could not load mobile_sam.pt in /tmp: {e}")
        os.chdir(cwd)
        return
    os.chdir(cwd)

    print(f"Processing {IMAGE_PATH}...")
    # Run inference
    # We can prompt it, or just let it segment everything.
    # Since we want stones, maybe 'segment everything' is best, then filter.
    results = model(IMAGE_PATH)

    # Process results
    for i, result in enumerate(results):
        img = result.orig_img

        if result.masks is None:
            print("No masks found.")
            return

        masks = result.masks.data.cpu().numpy() # masks
        segments = result.masks.xy # list of polygon points

        print(f"Found {len(segments)} segments.")

        # Draw contours on image
        debug_img = img.copy()

        # Filter segments?
        # Stones are likely large.
        # ArUco marker is likely small square.

        valid_stones = []

        for j, seg in enumerate(segments):
            if len(seg) == 0: continue

            # Convert to int32 for drawing
            pts = np.array(seg, np.int32)
            pts = pts.reshape((-1, 1, 2))

            area = cv2.contourArea(pts)

            # Filter by area (heuristic)
            # Image is large (4032x3024 approx).
            # ArUco is 5x5 inches.
            # Let's assume stones are significant in size.
            if area > 10000:
                cv2.polylines(debug_img, [pts], True, (0, 255, 0), 5)
                # Calculate centroid
                M = cv2.moments(pts)
                if M['m00'] != 0:
                    cx = int(M['m10']/M['m00'])
                    cy = int(M['m01']/M['m00'])
                    cv2.putText(debug_img, f"#{j}", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

                valid_stones.append(seg)

        output_path = f"{OUTPUT_DIR}/seg_result.jpg"
        cv2.imwrite(output_path, debug_img)
        print(f"Saved visualization to {output_path}")
        print(f"Found {len(valid_stones)} potential stones (area > 10000)")

if __name__ == "__main__":
    test_segmentation()
