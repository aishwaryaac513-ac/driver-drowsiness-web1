"""
Driver Drowsiness Detection System
------------------------------------
Uses MediaPipe's FaceLandmarker (Tasks API) to track facial landmarks in real
time from a webcam, computes the Eye Aspect Ratio (EAR) and Mouth Aspect
Ratio (MAR), and raises a drowsiness / yawn alert when thresholds are
crossed for a sustained number of frames.

Setup (one-time): download the face landmark model file into this folder:
    python -c "import urllib.request; urllib.request.urlretrieve('https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task', 'face_landmarker.task')"

Run:
    python drowsiness_detector.py

Controls:
    q - quit
"""

import time
import threading
import winsound
from collections import deque

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_PATH = "face_landmarker.task"

EAR_THRESHOLD = 0.21          # below this -> eye considered "closed"
EAR_CONSEC_FRAMES = 20        # consecutive closed-eye frames -> drowsy alert
MAR_THRESHOLD = 0.6           # above this -> mouth considered "open" (yawn)
MAR_CONSEC_FRAMES = 15        # consecutive open-mouth frames -> yawn alert
SMOOTHING_WINDOW = 5          # frames to average EAR/MAR over (reduces jitter)

# Landmark indices (same 468-point topology as before)
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
MOUTH_LEFT = 78
MOUTH_RIGHT = 308
MOUTH_TOP_OUTER = 0
MOUTH_BOTTOM_OUTER = 17

# Head-pose landmark indices (6-point model used with solvePnP)
NOSE_TIP = 1
CHIN = 152
LEFT_EYE_CORNER = 33
RIGHT_EYE_CORNER = 263
MOUTH_LEFT_CORNER = 61
MOUTH_RIGHT_CORNER = 291

# Generic 3D face model points (mm), used as a reference for pose estimation.
# These aren't this specific user's face geometry - they're a stand-in average
# face shape, which is standard practice and works well enough for pitch/yaw.
MODEL_3D_POINTS = np.array([
    (0.0, 0.0, 0.0),         # Nose tip
    (0.0, -330.0, -65.0),    # Chin
    (-225.0, 170.0, -135.0),  # Left eye corner
    (225.0, 170.0, -135.0),   # Right eye corner
    (-150.0, -150.0, -125.0),  # Left mouth corner
    (150.0, -150.0, -125.0),   # Right mouth corner
], dtype=np.float64)

HEAD_PITCH_DOWN_THRESHOLD = 15   # degrees of downward head tilt to count as "nodding"
NOD_CONSEC_FRAMES = 15           # consecutive nodding frames -> head-nod alert

HEAD_PITCH_BACK_THRESHOLD = -15   # degrees of backward head tilt (head lolling back)
BACK_TILT_CONSEC_FRAMES = 15      # consecutive backward-tilt frames -> alert

# If eyes stay closed for this many consecutive frames (much longer than the
# initial drowsy alert), escalate to a critical "high accident risk" alert.
CRITICAL_EYE_CLOSED_FRAMES = 60


def euclidean(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))


def eye_aspect_ratio(landmarks, eye_indices, w, h):
    pts = [(landmarks[i].x * w, landmarks[i].y * h) for i in eye_indices]
    p1, p2, p3, p4, p5, p6 = pts
    vertical1 = euclidean(p2, p6)
    vertical2 = euclidean(p3, p5)
    horizontal = euclidean(p1, p4)
    if horizontal == 0:
        return 0.0
    return (vertical1 + vertical2) / (2.0 * horizontal)


def mouth_aspect_ratio(landmarks, w, h):
    top = np.array([landmarks[MOUTH_TOP_OUTER].x * w, landmarks[MOUTH_TOP_OUTER].y * h])
    bottom = np.array([landmarks[MOUTH_BOTTOM_OUTER].x * w, landmarks[MOUTH_BOTTOM_OUTER].y * h])
    left = np.array([landmarks[MOUTH_LEFT].x * w, landmarks[MOUTH_LEFT].y * h])
    right = np.array([landmarks[MOUTH_RIGHT].x * w, landmarks[MOUTH_RIGHT].y * h])
    vertical = np.linalg.norm(top - bottom)
    horizontal = np.linalg.norm(left - right)
    if horizontal == 0:
        return 0.0
    return vertical / horizontal


def draw_landmarks_subset(frame, landmarks, indices, w, h, color):
    for i in indices:
        x, y = int(landmarks[i].x * w), int(landmarks[i].y * h)
        cv2.circle(frame, (x, y), 2, color, -1)


def get_head_pitch(landmarks, w, h):
    """Estimates head pitch (up/down tilt) in degrees using solvePnP.

    Positive pitch = head tilted down (nodding forward), which is what we
    want to detect. Returns None if pose estimation fails.
    """
    image_points = np.array([
        (landmarks[NOSE_TIP].x * w, landmarks[NOSE_TIP].y * h),
        (landmarks[CHIN].x * w, landmarks[CHIN].y * h),
        (landmarks[LEFT_EYE_CORNER].x * w, landmarks[LEFT_EYE_CORNER].y * h),
        (landmarks[RIGHT_EYE_CORNER].x * w, landmarks[RIGHT_EYE_CORNER].y * h),
        (landmarks[MOUTH_LEFT_CORNER].x * w, landmarks[MOUTH_LEFT_CORNER].y * h),
        (landmarks[MOUTH_RIGHT_CORNER].x * w, landmarks[MOUTH_RIGHT_CORNER].y * h),
    ], dtype=np.float64)

    focal_length = w
    center = (w / 2, h / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1],
    ], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1))  # assume no lens distortion

    success, rotation_vec, _ = cv2.solvePnP(
        MODEL_3D_POINTS, image_points, camera_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not success:
        return None

    rotation_mat, _ = cv2.Rodrigues(rotation_vec)
    pose_mat = np.hstack((rotation_mat, np.zeros((3, 1))))
    _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_mat)
    pitch = euler_angles[0][0]

    # Normalize: solvePnP/decompose can wrap pitch outside +-90 depending on
    # head orientation; fold it back into an intuitive "down is positive" range.
    if pitch < -90:
        pitch = -(180 + pitch)
    elif pitch > 90:
        pitch = 180 - pitch

    return -pitch  # flip sign so downward tilt is positive


def alarm_loop(stop_event, frequency, pattern_gap):
    """Beeps repeatedly on a background thread until stop_event is set."""
    while not stop_event.is_set():
        winsound.Beep(frequency, 300)
        stop_event.wait(pattern_gap)


class Alarm:
    """Starts/stops a background beeping thread, avoiding duplicate threads."""
    def __init__(self, frequency, pattern_gap=0.2):
        self.frequency = frequency
        self.pattern_gap = pattern_gap
        self._thread = None
        self._stop_event = threading.Event()

    def start(self):
        if self._thread is None or not self._thread.is_alive():
            self._stop_event.clear()
            self._thread = threading.Thread(
                target=alarm_loop,
                args=(self._stop_event, self.frequency, self.pattern_gap),
                daemon=True,
            )
            self._thread.start()

    def stop(self):
        self._stop_event.set()


def siren_loop(stop_event):
    """Alternates between two tones quickly - a more urgent 'siren' pattern
    for the critical/high-accident-risk escalation."""
    freqs = [1600, 2800]
    i = 0
    while not stop_event.is_set():
        winsound.Beep(freqs[i % 2], 150)
        i += 1


class SirenAlarm:
    """Same start/stop interface as Alarm, but plays an alternating siren tone."""
    def __init__(self):
        self._thread = None
        self._stop_event = threading.Event()

    def start(self):
        if self._thread is None or not self._thread.is_alive():
            self._stop_event.clear()
            self._thread = threading.Thread(
                target=siren_loop, args=(self._stop_event,), daemon=True
            )
            self._thread.start()

    def stop(self):
        self._stop_event.set()


def main():
    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
    )
    landmarker = vision.FaceLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("ERROR: Could not open webcam. Check camera permissions/index.")
        return

    eye_closed_counter = 0
    mouth_open_counter = 0
    nod_counter = 0
    back_tilt_counter = 0
    ear_history = deque(maxlen=SMOOTHING_WINDOW)
    mar_history = deque(maxlen=SMOOTHING_WINDOW)

    drowsy_alert_active = False
    yawn_alert_active = False
    nod_alert_active = False
    back_tilt_alert_active = False
    critical_alert_active = False

    drowsy_alarm = Alarm(frequency=2500, pattern_gap=0.15)  # urgent, fast beeps
    yawn_alarm = Alarm(frequency=1200, pattern_gap=0.6)     # gentler, slower beeps
    nod_alarm = Alarm(frequency=1800, pattern_gap=0.3)      # mid-urgency beeps
    back_alarm = Alarm(frequency=2000, pattern_gap=0.25)    # backward-tilt beeps
    critical_alarm = SirenAlarm()                           # accident-risk siren

    prev_time = time.time()
    start_time = time.time()

    print("Starting drowsiness detection. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame from webcam.")
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int((time.time() - start_time) * 1000)
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        status_text = "No face detected"
        status_color = (0, 165, 255)

        if not result.face_landmarks:
            drowsy_alarm.stop()
            yawn_alarm.stop()
            nod_alarm.stop()
            back_alarm.stop()
            critical_alarm.stop()

        if result.face_landmarks:
            landmarks = result.face_landmarks[0]

            left_ear = eye_aspect_ratio(landmarks, LEFT_EYE, w, h)
            right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE, w, h)
            ear = (left_ear + right_ear) / 2.0
            mar = mouth_aspect_ratio(landmarks, w, h)

            ear_history.append(ear)
            mar_history.append(mar)
            smoothed_ear = sum(ear_history) / len(ear_history)
            smoothed_mar = sum(mar_history) / len(mar_history)

            draw_landmarks_subset(frame, landmarks, LEFT_EYE + RIGHT_EYE, w, h, (0, 255, 0))
            draw_landmarks_subset(
                frame, landmarks,
                [MOUTH_TOP_OUTER, MOUTH_BOTTOM_OUTER, MOUTH_LEFT, MOUTH_RIGHT],
                w, h, (255, 0, 255)
            )

            # --- Eye closure / drowsiness logic ---
            if smoothed_ear < EAR_THRESHOLD:
                eye_closed_counter += 1
            else:
                eye_closed_counter = 0
                drowsy_alert_active = False
                critical_alert_active = False

            if eye_closed_counter >= EAR_CONSEC_FRAMES:
                drowsy_alert_active = True

            # Escalate to critical "high accident risk" if eyes stay closed
            # far longer than the initial drowsy alert
            if eye_closed_counter >= CRITICAL_EYE_CLOSED_FRAMES:
                critical_alert_active = True

            # --- Yawn logic ---
            if smoothed_mar > MAR_THRESHOLD:
                mouth_open_counter += 1
            else:
                mouth_open_counter = 0
                yawn_alert_active = False

            if mouth_open_counter >= MAR_CONSEC_FRAMES:
                yawn_alert_active = True

            # --- Head-nod (forward) / droop logic ---
            pitch = get_head_pitch(landmarks, w, h)
            if pitch is not None and pitch > HEAD_PITCH_DOWN_THRESHOLD:
                nod_counter += 1
            else:
                nod_counter = 0
                nod_alert_active = False

            if nod_counter >= NOD_CONSEC_FRAMES:
                nod_alert_active = True

            # --- Head tilted back logic (head lolling backward) ---
            if pitch is not None and pitch < HEAD_PITCH_BACK_THRESHOLD:
                back_tilt_counter += 1
            else:
                back_tilt_counter = 0
                back_tilt_alert_active = False

            if back_tilt_counter >= BACK_TILT_CONSEC_FRAMES:
                back_tilt_alert_active = True

            # Priority: critical accident-risk > drowsy eyes > head tilted
            # back > head nod forward > yawn (most to least urgent)
            if critical_alert_active:
                status_text = "CRITICAL! HIGH ACCIDENT RISK - WAKE UP!"
                status_color = (0, 0, 255)
                critical_alarm.start()
                drowsy_alarm.stop()
                nod_alarm.stop()
                back_alarm.stop()
                yawn_alarm.stop()
            elif drowsy_alert_active:
                status_text = "DROWSINESS ALERT!"
                status_color = (0, 0, 255)
                drowsy_alarm.start()
                critical_alarm.stop()
                nod_alarm.stop()
                back_alarm.stop()
                yawn_alarm.stop()
            elif back_tilt_alert_active:
                status_text = "HEAD TILTED BACK - Possible microsleep"
                status_color = (255, 0, 180)
                back_alarm.start()
                critical_alarm.stop()
                drowsy_alarm.stop()
                nod_alarm.stop()
                yawn_alarm.stop()
            elif nod_alert_active:
                status_text = "HEAD NOD DETECTED - Fatigue sign"
                status_color = (0, 90, 255)
                nod_alarm.start()
                critical_alarm.stop()
                drowsy_alarm.stop()
                back_alarm.stop()
                yawn_alarm.stop()
            elif yawn_alert_active:
                status_text = "YAWN DETECTED - Fatigue sign"
                status_color = (0, 140, 255)
                yawn_alarm.start()
                critical_alarm.stop()
                drowsy_alarm.stop()
                nod_alarm.stop()
                back_alarm.stop()
            else:
                status_text = "Active"
                status_color = (0, 255, 0)
                critical_alarm.stop()
                drowsy_alarm.stop()
                yawn_alarm.stop()
                nod_alarm.stop()
                back_alarm.stop()

            cv2.putText(frame, f"EAR: {smoothed_ear:.2f}", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"MAR: {smoothed_mar:.2f}", (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            if pitch is not None:
                cv2.putText(frame, f"Pitch: {pitch:.1f} deg", (20, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            if critical_alert_active:
                # flashing effect: alternate border thickness/brightness by frame parity
                flash_color = (0, 0, 255) if int(time.time() * 4) % 2 == 0 else (255, 255, 255)
                cv2.rectangle(frame, (0, 0), (w, h), flash_color, 12)
            elif drowsy_alert_active:
                cv2.rectangle(frame, (0, 0), (w, h), (0, 0, 255), 8)
            elif back_tilt_alert_active:
                cv2.rectangle(frame, (0, 0), (w, h), (255, 0, 180), 8)
            elif nod_alert_active:
                cv2.rectangle(frame, (0, 0), (w, h), (0, 90, 255), 8)

        # FPS counter
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time) if curr_time != prev_time else 0.0
        prev_time = curr_time
        cv2.putText(frame, f"FPS: {int(fps)}", (w - 130, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

        cv2.putText(frame, status_text, (20, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, status_color, 2)

        cv2.imshow("Driver Drowsiness Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    drowsy_alarm.stop()
    yawn_alarm.stop()
    nod_alarm.stop()
    back_alarm.stop()
    critical_alarm.stop()
    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()


if __name__ == "__main__":
    main()
