# Automated Quality Inspection System (Computer Vision)

**Batch:** 2026 | **Powered by:** DecodeLabs  
**Module:** Gear Defect Detection (Digital Gatekeeper)  
**Core Architecture:** Input-Process-Output (IPO) Pipeline  

---

## Overview

The **Automated Quality Inspection System** is a real-time computer vision solution designed for industrial gear component inspection. Operating as an automated digital gatekeeper, the system evaluates incoming camera feeds or image frames to detect structural anomalies, missing tooth segments, or deep notches in manufactured gears.

---

## Technical Pipeline (IPO Architecture)

1. **Input:** Photometric image capture via webcam/video feed or static file, converted into grayscale intensity[cite: 1].
2. **Process:**
   - **Noise Filtering:** 5x5 Gaussian Blur to eliminate high-frequency camera sensor noise[cite: 1].
   - **Binarization:** Binary thresholding to isolate component silhouettes[cite: 1].
   - **Contour Extraction:** Boundary vector detection to locate primary component bodies[cite: 1].
   - **Convex Hull & Defects:** Calculates structural concavity and depth using OpenCV's convexity defects algorithm[cite: 1].
3. **Output:** Real-time visual verdict (`PASS`/`FAIL`) overlay with bounding boxes surrounding detected anomalies[cite: 1].

---

## Project Structure

```text
Quality-Inspection-System/
│
├── main.py                # Main inspection script
├── requirements.txt       # Dependencies (OpenCV, NumPy)
└── README.md              # Project documentation