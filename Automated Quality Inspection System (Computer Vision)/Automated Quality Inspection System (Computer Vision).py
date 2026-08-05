"""
=============================================================================
PROJECT: Automated Quality Inspection System (Computer Vision)
BATCH: 2026 | Powered by DecodeLabs
MODULE: Gear Defect Detection (Digital Gatekeeper)

ARCHITECTURE: Input-Process-Output (IPO)
1. Input: Photometric capture & Grayscale conversion
2. Process: Noise reduction (Gaussian Blur), Binarization, Contour extraction,
Convex Hull & Convexity Defects calculation
3. Output: Real-time visual verdict (PASS/FAIL) & Defect Bounding Boxes

Author:Syed Abrar Hussain

REQUIREMENTS: opencv-python, numpy
=============================================================================
"""

import sys
import cv2
import numpy as np


def inspect_gear(image, distance_threshold=15.0):
    """Performs optical quality inspection on a gear image frame.

    Parameters:
        image (ndarray): Input BGR image frame.
        distance_threshold (float): Tolerance threshold in pixels for defect
        depth.

    Returns:
        tuple: (annotated_image, verdict_str, defect_count)
    """
    if image is None:
        return None, "FAIL", 0

    output_img = image.copy()

    # ==========================================
    # PHASE 1: ISOLATING THE SIGNAL FROM THE NOISE
    # ==========================================
    # 1. Flatten: Convert BGR photons into single-channel grayscale intensity
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 2. Smooth: Apply Gaussian Blur to filter high-frequency sensor noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 3. Binarize: Thresholding to isolate gear silhouette
    _, binarized = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)

    # ==========================================
    # PHASE 2: TOPOLOGY & MEASUREMENT
    # ==========================================
    # 1. Extract boundary vectors (contours)
    contours, _ = cv2.findContours(
        binarized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        cv2.putText(
            output_img,
            "NO COMPONENT DETECTED",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )
        return output_img, "FAIL", 0

    # Isolate the main gear body (largest contour)
    gear_contour = max(contours, key=cv2.contourArea)

    # 2. Compute Convex Hull (returnPoints=False is strictly required for defects)
    hull = cv2.convexHull(gear_contour, returnPoints=False)

    if hull is None or len(hull) < 3:
        return output_img, "PASS", 0

    # 3. Calculate Convexity Defects
    defects = cv2.convexityDefects(gear_contour, hull)

    # ==========================================
    # PHASE 3: TOLERANCE GATE & DECISION
    # ==========================================
    defect_count = 0

    if defects is not None:
        for defect in defects:
            # SHAPE FIX: Reshape handles both (N, 1, 4) and (N, 4) NumPy array formats
            s, e, f, d_raw = defect.reshape(-1)

            # Fixed-Point Correction: OpenCV scales raw distance by 256.0
            actual_distance = float(d_raw) / 256.0

            # Evaluate distance against calibrated THRESHOLD_MAX
            if actual_distance > distance_threshold:
                defect_count += 1

                # Coordinates of the deepest concavity (farthest point)
                far_pt = tuple(gear_contour[int(f)][0])

                # Dynamically construct bounding box around defect
                box_offset = 25
                top_left = (
                    max(0, far_pt[0] - box_offset),
                    max(0, far_pt[1] - box_offset),
                )
                bottom_right = (
                    min(image.shape[1], far_pt[0] + box_offset),
                    min(image.shape[0], far_pt[1] + box_offset),
                )

                # Draw red bounding box and defect point
                cv2.rectangle(
                    output_img, top_left, bottom_right, (0, 0, 255), 2
                )
                cv2.circle(output_img, far_pt, 4, (0, 0, 255), -1)

    # Determine PASS/FAIL decision
    verdict = "FAIL" if defect_count > 0 else "PASS"
    status_color = (0, 0, 255) if verdict == "FAIL" else (0, 255, 0)

    # Overlay Verdict Banner
    label = f"[ {verdict}: {defect_count} DEFECT(S) DETECTED ]"
    cv2.putText(
        output_img,
        label,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        status_color,
        2,
    )

    return output_img, verdict, defect_count


def run_inspection(video_source=0):
    """Runs quality inspection on webcam, video file, or single image."""
    if isinstance(video_source, int):
        cap = (
            cv2.VideoCapture(video_source, cv2.CAP_DSHOW)
            if sys.platform.startswith("win")
            else cv2.VideoCapture(video_source)
        )

        if not cap.isOpened():
            print(
                f"\n[WARNING] Camera index {video_source} not accessible."
                "\n[INFO] Running Synthetic Image Test Mode..."
            )
            run_single_image_test()
            return

        print("[SYSTEM STATUS: ONLINE] Press 'q' to stop conveyor feed.")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Failed to capture video frame.")
                break

            processed_frame, verdict, defects = inspect_gear(frame)
            cv2.imshow("DecodeLabs - Quality Inspection", processed_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()
    else:
        frame = cv2.imread(video_source)
        if frame is not None:
            processed_frame, verdict, defects = inspect_gear(frame)
            print(
                f"Inspection Result: {verdict} | Structural Defects Found:"
                f" {defects}"
            )
            cv2.imshow("DecodeLabs - Quality Inspection", processed_frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print(f"[ERROR] Could not read media file at: {video_source}")


def run_single_image_test():
    """Generates a synthetic gear test image with a notch defect for demonstration."""
    # Create a 400x400 black canvas with a white gear circle
    canvas = np.zeros((400, 400, 3), dtype=np.uint8)
    cv2.circle(canvas, (200, 200), 100, (255, 255, 255), -1)

    # Simulating a 25px structural defect (notch)
    cv2.circle(canvas, (200, 100), 25, (0, 0, 0), -1)

    processed_frame, verdict, defects = inspect_gear(
        canvas, distance_threshold=10.0
    )
    print(f"Inspection Result: {verdict} | Defects Detected: {defects}")

    cv2.imshow(
        "DecodeLabs - Quality Inspection (Synthetic Test)", processed_frame
    )
    print("[INFO] Press any key on the display window to exit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_inspection(video_source=0)