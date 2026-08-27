"""
Driver Drowsiness Detection - Web App
--------------------------------------
Browser-based version of the drowsiness detector. Uses streamlit-webrtc to
access the user's webcam directly in-browser, and runs the same MediaPipe
FaceLandmarker + EAR/MAR/head-pose logic as the desktop script.

Run locally:
    streamlit run app.py

Deploy: push this folder to GitHub and deploy on Streamlit Community Cloud
or Hugging Face Spaces (see README.md for steps).
"""

import time
from collections import deque

import av
import cv2
import numpy as np
import streamlit as st
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase

# ---------------------------------------------------------------------------
# Configuration (same defaults as the desktop version)
# ---------------------------------------------------------------------------
MODEL_PATH = "face_landmarker.task"

LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
MOUTH_LEFT = 78
MOUTH_RIGHT = 308
MOUTH_TOP_OUTER = 0
MOUTH_BOTTOM_OUTER = 17

NOSE_TIP = 1
CHIN = 152
LEFT_EYE_CORNER = 33
RIGHT_EYE_CORNER = 263
MOUTH_LEFT_CORNER = 61
MOUTH_RIGHT_CORNER = 291

MODEL_3D_POINTS = np.array([
    (0.0, 0.0, 0.0),
    (0.0, -330.0, -65.0),
    (-225.0, 170.0, -135.0),
    (225.0, 170.0, -135.0),
    (-150.0, -150.0, -125.0),
    (150.0, -150.0, -125.0),
], dtype=np.float64)

SMOOTHING_WINDOW = 5


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
    dist_coeffs = np.zeros((4, 1))

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

    if pitch < -90:
        pitch = -(180 + pitch)
    elif pitch > 90:
        pitch = 180 - pitch

    return -pitch


@st.cache_resource
def load_landmarker():
    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
    )
    return vision.FaceLandmarker.create_from_options(options)


class DrowsinessProcessor(VideoProcessorBase):
    """Runs on every incoming webcam frame in the browser session."""

    def __init__(self):
        self.landmarker = load_landmarker()
        self.start_time = time.time()

        self.ear_history = deque(maxlen=SMOOTHING_WINDOW)
        self.mar_history = deque(maxlen=SMOOTHING_WINDOW)

        self.eye_closed_counter = 0
        self.mouth_open_counter = 0
        self.nod_counter = 0
        self.back_tilt_counter = 0

        # Tunable thresholds - overwritten live from the sidebar each frame
        self.ear_threshold = 0.21
        self.ear_consec_frames = 20
        self.mar_threshold = 0.6
        self.mar_consec_frames = 15
        self.head_pitch_down_threshold = 15
        self.nod_consec_frames = 15
        self.head_pitch_back_threshold = -15
        self.back_tilt_consec_frames = 15
        self.critical_eye_closed_frames = 60

        # Shared with the main Streamlit thread for on-page status display
        self.status_text = "Starting..."
        self.status_level = "ok"  # ok | warning | danger | critical

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        h, w = img.shape[:2]

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int((time.time() - self.start_time) * 1000)
        result = self.landmarker.detect_for_video(mp_image, timestamp_ms)

        status_text = "No face detected"
        status_color = (0, 165, 255)
        status_level = "warning"

        if result.face_landmarks:
            landmarks = result.face_landmarks[0]

            left_ear = eye_aspect_ratio(landmarks, LEFT_EYE, w, h)
            right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE, w, h)
            ear = (left_ear + right_ear) / 2.0
            mar = mouth_aspect_ratio(landmarks, w, h)

            self.ear_history.append(ear)
            self.mar_history.append(mar)
            smoothed_ear = sum(self.ear_history) / len(self.ear_history)
            smoothed_mar = sum(self.mar_history) / len(self.mar_history)

            draw_landmarks_subset(img, landmarks, LEFT_EYE + RIGHT_EYE, w, h, (0, 255, 0))
            draw_landmarks_subset(
                img, landmarks,
                [MOUTH_TOP_OUTER, MOUTH_BOTTOM_OUTER, MOUTH_LEFT, MOUTH_RIGHT],
                w, h, (255, 0, 255)
            )

            # --- Eye closure / drowsiness ---
            if smoothed_ear < self.ear_threshold:
                self.eye_closed_counter += 1
            else:
                self.eye_closed_counter = 0

            drowsy_active = self.eye_closed_counter >= self.ear_consec_frames
            critical_active = self.eye_closed_counter >= self.critical_eye_closed_frames

            # --- Yawn ---
            if smoothed_mar > self.mar_threshold:
                self.mouth_open_counter += 1
            else:
                self.mouth_open_counter = 0
            yawn_active = self.mouth_open_counter >= self.mar_consec_frames

            # --- Head pose ---
            pitch = get_head_pitch(landmarks, w, h)
            if pitch is not None and pitch > self.head_pitch_down_threshold:
                self.nod_counter += 1
            else:
                self.nod_counter = 0
            nod_active = self.nod_counter >= self.nod_consec_frames

            if pitch is not None and pitch < self.head_pitch_back_threshold:
                self.back_tilt_counter += 1
            else:
                self.back_tilt_counter = 0
            back_tilt_active = self.back_tilt_counter >= self.back_tilt_consec_frames

            # Priority: critical > drowsy > back-tilt > nod > yawn > active
            if critical_active:
                status_text, status_color, status_level = "CRITICAL! HIGH ACCIDENT RISK - WAKE UP!", (0, 0, 255), "critical"
            elif drowsy_active:
                status_text, status_color, status_level = "DROWSINESS ALERT!", (0, 0, 255), "danger"
            elif back_tilt_active:
                status_text, status_color, status_level = "HEAD TILTED BACK - Possible microsleep", (255, 0, 180), "danger"
            elif nod_active:
                status_text, status_color, status_level = "HEAD NOD DETECTED - Fatigue sign", (0, 90, 255), "warning"
            elif yawn_active:
                status_text, status_color, status_level = "YAWN DETECTED - Fatigue sign", (0, 140, 255), "warning"
            else:
                status_text, status_color, status_level = "Active", (0, 255, 0), "ok"

            cv2.putText(img, f"EAR: {smoothed_ear:.2f}", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(img, f"MAR: {smoothed_mar:.2f}", (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            if pitch is not None:
                cv2.putText(img, f"Pitch: {pitch:.1f} deg", (20, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            if critical_active:
                flash_color = (0, 0, 255) if int(time.time() * 4) % 2 == 0 else (255, 255, 255)
                cv2.rectangle(img, (0, 0), (w, h), flash_color, 12)
            elif drowsy_active:
                cv2.rectangle(img, (0, 0), (w, h), (0, 0, 255), 8)
            elif back_tilt_active:
                cv2.rectangle(img, (0, 0), (w, h), (255, 0, 180), 8)
            elif nod_active:
                cv2.rectangle(img, (0, 0), (w, h), (0, 90, 255), 8)

        cv2.putText(img, status_text, (20, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, status_color, 2)

        self.status_text = status_text
        self.status_level = status_level

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Driver Drowsiness Detection", page_icon="🚗", layout="wide")

st.title("🚗 Driver Drowsiness Detection")
st.caption("Real-time browser-based drowsiness & fatigue monitoring using MediaPipe FaceLandmarker.")
# Browser alarm
st.markdown("""
<script>
let alarmAudio = null;

function startAlarm() {
    if (!alarmAudio) {
        alarmAudio = new Audio(
            "https://actions.google.com/sounds/v1/alarms/alarm_clock.ogg"
        );
        alarmAudio.loop = true;
    }

    alarmAudio.play().catch(() => {});
}

function stopAlarm() {
    if (alarmAudio) {
        alarmAudio.pause();
        alarmAudio.currentTime = 0;
    }
}
</script>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("Detection Settings")
    ear_threshold = st.slider("EAR threshold (eye closed)", 0.10, 0.35, 0.21, 0.01)
    ear_frames = st.slider("Eye-closed frames -> alert", 5, 60, 20)
    mar_threshold = st.slider("MAR threshold (yawn)", 0.3, 1.0, 0.6, 0.05)
    mar_frames = st.slider("Yawn frames -> alert", 5, 40, 15)
    st.divider()
    st.markdown(
        "**Tip:** run it once, watch the live EAR value while blinking "
        "normally vs. closing your eyes for a couple seconds, and set the "
        "threshold roughly halfway between those two readings."
    )

col1, col2 = st.columns([3, 1])

with col1:
    ctx = webrtc_streamer(
        key="drowsiness-detection",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=DrowsinessProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

with col2:
    st.subheader("Status")
    status_placeholder = st.empty()
    st.caption("Alerts also flash as a colored border around the video.")
    st.markdown(
        "- 🟢 **Active** — normal\n"
        "- 🟠 **Yawn / Head nod** — early fatigue sign\n"
        "- 🔴 **Drowsiness alert** — eyes closed too long\n"
        "- 🚨 **Critical** — high accident risk"
    )

if ctx.video_processor:
    while ctx.state.playing:
        level = ctx.video_processor.status_level
        text = ctx.video_processor.status_text
        icon = {"ok": "🟢", "warning": "🟠", "danger": "🔴", "critical": "🚨"}.get(level, "⚪")
        status_placeholder.markdown(f"### {icon} {text}")
        time.sleep(0.3)
        if not ctx.state.playing:
            break
else:
    status_placeholder.markdown("### ⚪ Click **Start** above to begin")
