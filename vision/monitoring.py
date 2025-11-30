import cv2
import numpy as np


def measure_nozzle_diameter(image_path: str, pixels_to_mm_ratio: float):
    """
    Measures the inner bore AND outer diameter of the nozzle from an image.

    Returns:
        A tuple: (inner_diameter_mm, outer_diameter_mm)
        Returns (None, None) if two circles aren't found.
    """
    # Load the image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Could not load image from {image_path}")
        return None, None

    # Apply a Gaussian blur to reduce noise before Canny
    img_blur = cv2.GaussianBlur(img, (5, 5), 0)

    # 1. Apply Canny Edge Detection
    edges = cv2.Canny(image=img_blur, threshold1=100, threshold2=200)

    # 2. Apply Hough Transform for Circles
    # Tuned parameters to find concentric circles
    circles = cv2.HoughCircles(
        edges,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=20,  # Lowered min distance to find concentric circles
        param1=200,  # Upper threshold for Canny
        param2=25,  # Lowered accumulator threshold to find more circles
        minRadius=5,  # Smallest possible inner radius
        maxRadius=200  # Largest possible outer radius
    )

    if circles is not None:
        # We need at least two circles (inner and outer)
        if circles.shape[1] < 2:
            print(f"Vision System: Found only {circles.shape[1]} circle(s). Need at least two.")
            return None, None

        # --- This is the new logic ---
        # Sort all found circles by their radius (the 3rd element, index 2)
        all_circles = circles[0, :]
        sorted_circles = sorted(all_circles, key=lambda x: x[2])

        # The smallest circle is the inner bore
        inner_bore = sorted_circles[0]
        inner_radius_px = inner_bore[2]
        inner_dia_mm = (inner_radius_px * 2) * pixels_to_mm_ratio

        # The largest circle is the outer diameter
        outer_edge = sorted_circles[-1]
        outer_radius_px = outer_edge[2]
        outer_dia_mm = (outer_radius_px * 2) * pixels_to_mm_ratio

        print(f"Vision System: Detected Inner Bore = {inner_dia_mm:.3f} mm")
        print(f"Vision System: Detected Outer Diameter = {outer_dia_mm:.3f} mm")

        # Return both diameters
        return inner_dia_mm, outer_dia_mm

    else:
        print("Vision System: No circles detected.")
        return None, None

# --- Example Usage (requires a 'nozzle_tip.jpg' image) ---
# PIXEL_TO_MM = 0.01 # This must be calibrated!
# inner_d, outer_d = measure_nozzle_diameter('vision/test_images/nozzle_tip.jpg', PIXEL_TO_MM)
# if inner_d is not None:
#    print(f"Final diameters: Inner={inner_d}, Outer={outer_d}")