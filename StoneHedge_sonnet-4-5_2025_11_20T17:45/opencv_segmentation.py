"""
Traditional OpenCV-based Stone Segmentation Module

Uses classical computer vision techniques:
- Background subtraction
- Edge detection
- Contour finding
- Polygon simplification
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict
import json


class OpenCVSegmenter:
    """Traditional CV-based stone segmentation."""

    def __init__(self, min_area: int = 1000):
        """
        Initialize the segmenter.

        Args:
            min_area: Minimum contour area in pixels to consider
        """
        self.min_area = min_area

    def segment_stone(self, image: np.ndarray,
                      mask_aruco: bool = True) -> List[np.ndarray]:
        """
        Segment stone from background using traditional CV techniques.

        Args:
            image: Input image (BGR)
            mask_aruco: Whether to mask out ArUco marker region

        Returns:
            List of contours (each contour is Nx2 array of points)
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Try multiple edge detection approaches
        # 1. Canny edge detection
        edges_canny = cv2.Canny(blurred, 50, 150)

        # 2. Adaptive thresholding
        thresh_adaptive = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )

        # 3. Otsu's thresholding
        _, thresh_otsu = cv2.threshold(
            blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        # Combine edge maps
        combined = cv2.bitwise_or(edges_canny, thresh_adaptive)
        combined = cv2.bitwise_or(combined, thresh_otsu)

        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        morph = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
        morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)

        # Find contours
        contours, hierarchy = cv2.findContours(
            morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Filter contours by area
        valid_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > self.min_area:
                valid_contours.append(contour)

        # Sort by area (largest first)
        valid_contours = sorted(valid_contours, key=cv2.contourArea, reverse=True)

        return valid_contours

    def segment_with_grabcut(self, image: np.ndarray,
                            iterations: int = 5) -> np.ndarray:
        """
        Segment using GrabCut algorithm.

        Args:
            image: Input image (BGR)
            iterations: Number of GrabCut iterations

        Returns:
            Binary mask of foreground
        """
        # Initialize mask
        mask = np.zeros(image.shape[:2], np.uint8)

        # Initialize background and foreground models
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)

        # Define ROI (assume stone is in center ~60% of image)
        h, w = image.shape[:2]
        margin_w = int(w * 0.2)
        margin_h = int(h * 0.2)
        rect = (margin_w, margin_h, w - 2 * margin_w, h - 2 * margin_h)

        # Apply GrabCut
        cv2.grabCut(image, mask, rect, bgd_model, fgd_model,
                   iterations, cv2.GC_INIT_WITH_RECT)

        # Create binary mask (0 and 2 are background, 1 and 3 are foreground)
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

        return mask2

    def simplify_polygon(self, contour: np.ndarray,
                        epsilon_factor: float = 0.001) -> np.ndarray:
        """
        Simplify polygon using Douglas-Peucker algorithm.

        Args:
            contour: Input contour (Nx1x2 or Nx2 array)
            epsilon_factor: Approximation accuracy as fraction of perimeter

        Returns:
            Simplified contour
        """
        perimeter = cv2.arcLength(contour, True)
        epsilon = epsilon_factor * perimeter
        approx = cv2.approxPolyDP(contour, epsilon, True)
        return approx

    def extract_polygon(self, image: np.ndarray,
                       method: str = "traditional",
                       simplify: bool = True) -> Tuple[np.ndarray, Dict]:
        """
        Extract polygon from image using specified method.

        Args:
            image: Input image
            method: "traditional" or "grabcut"
            simplify: Whether to simplify the polygon

        Returns:
            Tuple of (polygon_points, metadata)
        """
        if method == "grabcut":
            # GrabCut approach
            mask = self.segment_with_grabcut(image)
            # Find contours from mask
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
        else:
            # Traditional approach
            contours = self.segment_stone(image)

        if not contours:
            raise ValueError("No stone contour found")

        # Take largest contour
        main_contour = max(contours, key=cv2.contourArea)

        # Simplify if requested
        if simplify:
            main_contour = self.simplify_polygon(main_contour)

        # Reshape to Nx2 if needed
        if main_contour.shape[1] == 1:
            main_contour = main_contour.reshape(-1, 2)

        # Calculate metadata
        area = cv2.contourArea(main_contour)
        perimeter = cv2.arcLength(main_contour, True)

        # Fit bounding rectangle
        rect = cv2.minAreaRect(main_contour)
        box = cv2.boxPoints(rect)

        metadata = {
            'method': method,
            'num_vertices': len(main_contour),
            'area_pixels': float(area),
            'perimeter_pixels': float(perimeter),
            'bounding_box': box.tolist(),
            'simplified': simplify
        }

        return main_contour, metadata

    def visualize_segmentation(self, image: np.ndarray,
                              polygon: np.ndarray) -> np.ndarray:
        """
        Visualize the extracted polygon on the image.

        Args:
            image: Input image
            polygon: Polygon points (Nx2 array)

        Returns:
            Image with polygon drawn
        """
        output = image.copy()

        # Reshape for drawing if needed
        if polygon.shape[1] == 2:
            polygon = polygon.reshape(-1, 1, 2)

        # Draw polygon
        cv2.drawContours(output, [polygon.astype(np.int32)], -1, (0, 255, 0), 3)

        # Draw vertices
        for i, point in enumerate(polygon):
            pt = tuple(point[0] if len(point.shape) > 1 else point)
            cv2.circle(output, pt, 5, (0, 0, 255), -1)
            cv2.putText(output, str(i), pt, cv2.FONT_HERSHEY_SIMPLEX,
                       0.5, (255, 255, 0), 1)

        return output


def test_opencv_segmentation(image_path: str, output_dir: str = "."):
    """
    Test OpenCV segmentation on a calibrated stone image.

    Args:
        image_path: Path to input image
        output_dir: Directory for output files
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")

    print(f"Testing OpenCV segmentation on: {image_path}")
    print(f"Image shape: {image.shape}")

    # Create segmenter
    segmenter = OpenCVSegmenter(min_area=5000)

    # Test traditional method
    try:
        polygon_trad, metadata_trad = segmenter.extract_polygon(
            image, method="traditional", simplify=True
        )
        print(f"\nTraditional method:")
        print(f"  Vertices: {len(polygon_trad)}")
        print(f"  Area: {metadata_trad['area_pixels']:.2f} pixels²")
        print(f"  Perimeter: {metadata_trad['perimeter_pixels']:.2f} pixels")

        # Visualize
        vis_trad = segmenter.visualize_segmentation(image, polygon_trad)
        cv2.imwrite(f"{output_dir}/opencv_seg_traditional.jpg", vis_trad)
        print(f"  Saved: {output_dir}/opencv_seg_traditional.jpg")

        # Save polygon data
        with open(f"{output_dir}/polygon_traditional.json", 'w') as f:
            data = {
                'polygon': polygon_trad.tolist(),
                'metadata': metadata_trad
            }
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"Traditional method failed: {e}")

    # Test GrabCut method
    try:
        polygon_gc, metadata_gc = segmenter.extract_polygon(
            image, method="grabcut", simplify=True
        )
        print(f"\nGrabCut method:")
        print(f"  Vertices: {len(polygon_gc)}")
        print(f"  Area: {metadata_gc['area_pixels']:.2f} pixels²")
        print(f"  Perimeter: {metadata_gc['perimeter_pixels']:.2f} pixels")

        # Visualize
        vis_gc = segmenter.visualize_segmentation(image, polygon_gc)
        cv2.imwrite(f"{output_dir}/opencv_seg_grabcut.jpg", vis_gc)
        print(f"  Saved: {output_dir}/opencv_seg_grabcut.jpg")

        # Save polygon data
        with open(f"{output_dir}/polygon_grabcut.json", 'w') as f:
            data = {
                'polygon': polygon_gc.tolist(),
                'metadata': metadata_gc
            }
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"GrabCut method failed: {e}")


if __name__ == "__main__":
    # Test on calibrated image
    test_image = "aruco_warped.jpg"
    test_opencv_segmentation(test_image, ".")
