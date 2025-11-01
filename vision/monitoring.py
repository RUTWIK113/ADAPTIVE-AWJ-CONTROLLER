import cv2
import numpy as np

def measure_nozzle_diameter(image_path: str, pixels_to_mm_ratio: float) -> float:
    """
    Measures the inner bore diameter of the nozzle from an image.
    Returns:
        The diameter in mm, or None if no circle is found.
    """
    # Load the image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Could not load image from {image_path}")
        return None

    # Apply a Gaussian blur to reduce noise before Canny
    img_blur = cv2.GaussianBlur(img, (5, 5), 0)

    # 1. Apply Canny Edge Detection
    # These threshold values are crucial and would need tuning.
    edges = cv2.Canny(image=img_blur, threshold1=100, threshold2=200)

    # 2. Apply Hough Transform for Circles
    # cv2.HoughCircles(image, method, dp, minDist, param1, param2, minRadius, maxRadius)
    # - dp=1: Inverse ratio of accumulator resolution (same as input)
    # - minDist=50: Minimum distance between centers of detected circles
    # - param1=200: Upper threshold for the Canny edge detector (internal)
    # - param2=30: Accumulator threshold (lower means more circles detected)
    # - minRadius/maxRadius: Define the expected size range of the bore

    circles = cv2.HoughCircles(
        edges,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=50,
        param1=200,
        param2=30,
        minRadius=10,  # Example: 10 pixels
        maxRadius=100  # Example: 100 pixels
    )

    if circles is not None:
        circles = np.uint16(np.around(circles))

        # --- This is the part described in the paper ---
        # The algorithm would find TWO circles (inner bore, outer edge).
        # We would need logic to select the correct inner bore.
        # For simplicity, we'll assume the first one found is the one we want.

        inner_bore = circles[0, 0]
        radius_pixels = inner_bore[2]
        diameter_pixels = radius_pixels * 2

        # Convert pixels to mm
        diameter_mm = diameter_pixels * pixels_to_mm_ratio

        print(f"Vision System: Detected bore diameter = {diameter_mm:.3f} mm")
        return diameter_mm
    else:
        print("Vision System: No circles detected.")
        return None

# --- Example Usage (requires a 'nozzle_tip.jpg' image) ---
# PIXEL_TO_MM = 0.01 # This must be calibrated!
# measured_diameter = measure_nozzle_diameter('nozzle_tip.jpg', PIXEL_TO_MM)