import cv2
import numpy as np
import mediapipe as mp
import urllib.request
import os
import winsound

# Download the face landmarker model if not present
model_path = 'face_landmarker.task'
if not os.path.exists(model_path):
    url = 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task'
    urllib.request.urlretrieve(url, model_path)

# Use the new FaceLandmarker API
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.IMAGE)

face_landmarker = FaceLandmarker.create_from_options(options)

# Indices (same as before)
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

def get_ear(landmarks, eye_indices, w, h):
    pts = [np.array([landmarks[i].x * w, landmarks[i].y * h]) for i in eye_indices]
    v1 = np.linalg.norm(pts[1] - pts[5])
    v2 = np.linalg.norm(pts[2] - pts[4])
    horiz = np.linalg.norm(pts[0] - pts[3])
    return (v1 + v2) / (2.0 * horiz)

cap = cv2.VideoCapture(0)
EAR_THRESHOLD = 0.21
FRAME_CHECK = 20
counter = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    results = face_landmarker.detect(mp_image)

    if results.face_landmarks:
        for face_lms in results.face_landmarks:
            left_ear = get_ear(face_lms, LEFT_EYE, w, h)
            right_ear = get_ear(face_lms, RIGHT_EYE, w, h)
            avg_ear = (left_ear + right_ear) / 2.0

            if avg_ear < EAR_THRESHOLD:
                counter += 1
                if counter >= FRAME_CHECK:
                    cv2.putText(frame, "DROWSINESS ALERT!", (100, 200),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 4)
                    if not alert_played:
                        winsound.Beep(10000, 10000)  # Beep at 10000 Hz for 1 second
                        alert_played = True
            else:
                counter = 0
                alert_played = False

            cv2.putText(frame, f"EAR: {avg_ear:.2f}", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Driver Monitor", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
