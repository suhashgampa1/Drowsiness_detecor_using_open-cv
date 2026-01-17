# Real-Time Drowsiness Detection System

A safety-focused computer vision application that monitors driver alertness by analyzing facial landmarks in real-time. The system calculates the Eye Aspect Ratio (EAR) to detect prolonged eye closure and triggers an audible alert to prevent accidents caused by fatigue.

## 🚀 Features
* **MediaPipe Face Mesh:** Uses high-fidelity face landmark detection to track eye movements.
* **EAR Calculation:** Implements the Eye Aspect Ratio algorithm to accurately distinguish between normal blinking and drowsiness.
* **Automated Setup:** Automatically downloads the necessary `face_landmarker.task` model file on the first run.
* **Audible Alerts:** Utilizes system-level alerts (`winsound`) to notify the user when drowsiness is detected.
* **Dynamic Thresholding:** Includes a frame-check counter to avoid false positives from natural blinking.

## 🛠️ Tech Stack
* **Python 3.11+**
* **MediaPipe:** For facial landmark detection.
* **OpenCV:** For video stream processing and UI overlays.
* **NumPy:** For geometric calculations and Euclidean distance measurements.

## 📸 How it Works
The system identifies specific landmarks around the eyes (indices 362, 385, 387, 263, 373, 380 for the left eye). It then calculates the **Eye Aspect Ratio (EAR)** using the formula:

$$EAR = \frac{||p_2 - p_6|| + ||p_3 - p_5||}{2 ||p_1 - p_4||}$$

Where $p_n$ are the coordinates of the eye landmarks. If the EAR remains below a threshold (e.g., 0.21) for more than 20 consecutive frames, a "DROWSINESS ALERT!" is displayed and an alarm sounds.
