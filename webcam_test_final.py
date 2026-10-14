import cv2
import json
import time
import subprocess
import math
from pathlib import Path
from collections import deque

import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image
import mediapipe as mp

# ---------- Configuration ----------
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model_artifacts"
MODEL_PATH = MODEL_DIR / "drowsiness_model_25ep_best.pth"
CONFIG_PATH = MODEL_DIR / "config_25ep.json"
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Image-classifier smoothing
FATIGUE_THRESHOLD = 0.80
WINDOW_SIZE = 30
MIN_FATIGUE_FRAMES = 20

# Landmark thresholds: starting points; tune with webcam measurements
EYE_CLOSURE_THRESHOLD = 0.21
EYE_CLOSURE_ALERT_SECONDS = 1.0
MAR_THRESHOLD = 0.55
YAWN_DURATION = 1.2
ALERT_COOLDOWN = 5.0


def landmark_distance(a, b, frame_width, frame_height):
    return math.hypot(
        (a.x - b.x) * frame_width,
        (a.y - b.y) * frame_height
    )


def calculate_mar(landmarks, width, height):
    v1 = landmark_distance(landmarks[13], landmarks[14], width, height)
    v2 = landmark_distance(landmarks[82], landmarks[87], width, height)
    v3 = landmark_distance(landmarks[312], landmarks[317], width, height)
    horizontal = landmark_distance(landmarks[78], landmarks[308], width, height)
    return 0.0 if horizontal < 1 else (v1 + v2 + v3) / (3.0 * horizontal)


def calculate_ear(landmarks, indices, width, height):
    p1, p2, p3, p4, p5, p6 = [landmarks[i] for i in indices]
    vertical1 = landmark_distance(p2, p6, width, height)
    vertical2 = landmark_distance(p3, p5, width, height)
    horizontal = landmark_distance(p1, p4, width, height)
    return 0.0 if horizontal < 1 else (vertical1 + vertical2) / (2.0 * horizontal)


def play_alert():
    try:
        subprocess.Popen(
            ["osascript", "-e", "beep 1"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
    except OSError:
        pass


# ---------- Load classifier ----------
if not MODEL_PATH.exists() or not CONFIG_PATH.exists():
    raise FileNotFoundError(f"Expected model files at {MODEL_PATH} and {CONFIG_PATH}")

with open(CONFIG_PATH) as f:
    config = json.load(f)

classes = config["classes"]
class_to_idx = config["class_to_idx"]
fatigue_idx = class_to_idx["fatigue"]
active_idx = class_to_idx["active"]
image_size = config.get("image_size", 224)
mean = config.get("mean", [0.485, 0.456, 0.406])
std = config.get("std", [0.229, 0.224, 0.225])

transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

model = models.mobilenet_v3_small(weights=None)
model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, len(classes))
state_dict = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
model.load_state_dict(state_dict)
model = model.to(DEVICE)
model.eval()
print("Loaded model on:", DEVICE, "| Classes:", classes)

# ---------- OpenCV and MediaPipe ----------
cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_detector = cv2.CascadeClassifier(cascade_path)
if face_detector.empty():
    raise RuntimeError("Could not load OpenCV face detector")

face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    face_mesh.close()
    raise RuntimeError("Cannot access webcam. Check camera permissions.")

history = deque(maxlen=WINDOW_SIZE)
last_alert_time = 0.0
yawn_start_time = None
eye_closed_start_time = None
print("Press Q to quit.")

try:
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Failed to read webcam frame")
            break

        frame = cv2.flip(frame, 1)
        height, width = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
        )

        # Per-frame defaults
        fatigue_probability = active_probability = None
        fatigue_alert = eye_closure_alert = yawn_detected = False
        fatigue_frames = sum(history)
        mar = ear = None
        eye_closed_duration = yawn_elapsed = 0.0
        eyes_closed = brief_blink = False
        status, color = "No face detected", (0, 200, 255)

        # ----- MediaPipe landmarks: eyes and mouth -----
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mesh_results = face_mesh.process(rgb_frame)

        if mesh_results.multi_face_landmarks:
            landmarks = mesh_results.multi_face_landmarks[0].landmark
            now = time.monotonic()
            mar = calculate_mar(landmarks, width, height)

            left_ear = calculate_ear(
                landmarks, [33, 160, 158, 133, 153, 144], width, height
            )
            right_ear = calculate_ear(
                landmarks, [362, 385, 387, 263, 373, 380], width, height
            )
            ear = (left_ear + right_ear) / 2.0
            eyes_closed = ear < EYE_CLOSURE_THRESHOLD

            if eyes_closed:
                if eye_closed_start_time is None:
                    eye_closed_start_time = now
                eye_closed_duration = now - eye_closed_start_time
                eye_closure_alert = eye_closed_duration >= EYE_CLOSURE_ALERT_SECONDS
            else:
                eye_closed_start_time = None
                eye_closed_duration = 0.0

            brief_blink = eyes_closed and not eye_closure_alert

            if mar >= MAR_THRESHOLD:
                if yawn_start_time is None:
                    yawn_start_time = now
                yawn_elapsed = now - yawn_start_time
                yawn_detected = yawn_elapsed >= YAWN_DURATION
            else:
                yawn_start_time = None
                yawn_elapsed = 0.0

            for idx in [13, 14, 82, 87, 312, 317, 78, 308]:
                p = landmarks[idx]
                cv2.circle(frame, (int(p.x * width), int(p.y * height)), 3, (255, 0, 255), -1)
        else:
            yawn_start_time = None
            eye_closed_start_time = None

        # ----- MobileNet classifier -----
        if len(faces) > 0:
            x, y, w, h = max(faces, key=lambda face: face[2] * face[3])
            pad_x, pad_y = int(w * 0.12), int(h * 0.12)
            x1, y1 = max(0, x - pad_x), max(0, y - pad_y)
            x2, y2 = min(width, x + w + pad_x), min(height, y + h + pad_y)
            crop = frame[y1:y2, x1:x2]
            rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            tensor = transform(Image.fromarray(rgb_crop)).unsqueeze(0).to(DEVICE)

            with torch.inference_mode():
                probabilities = torch.softmax(model(tensor), dim=1)[0]

            fatigue_probability = probabilities[fatigue_idx].item()
            active_probability = probabilities[active_idx].item()

            # A short blink does not count as a positive fatigue frame.
            history.append(
                fatigue_probability >= FATIGUE_THRESHOLD and not brief_blink
            )
            fatigue_frames = sum(history)
            fatigue_alert = (
                len(history) == WINDOW_SIZE
                and fatigue_frames >= MIN_FATIGUE_FRAMES
            )

            cv2.rectangle(
                frame, (x1, y1), (x2, y2),
                (0, 0, 255) if fatigue_alert else (0, 220, 0), 2
            )
        else:
            history.clear()
            fatigue_frames = 0

        # ----- Alert and status logic -----
        any_alert = fatigue_alert or eye_closure_alert or yawn_detected

        if fatigue_alert and yawn_detected:
            status, color = "FATIGUE + YAWN WARNING", (0, 0, 255)
        elif eye_closure_alert:
            status, color = "PROLONGED EYE CLOSURE", (0, 0, 255)
        elif fatigue_alert:
            status, color = "FATIGUE WARNING", (0, 0, 255)
        elif yawn_detected:
            status, color = "POSSIBLE YAWN", (0, 0, 255)
        elif eyes_closed:
            status, color = "BLINK / EYES CLOSED", (0, 220, 0)
        elif mar is not None and mar >= MAR_THRESHOLD:
            status, color = "Mouth open - checking duration", (0, 165, 255)
        elif fatigue_probability is not None and fatigue_probability >= FATIGUE_THRESHOLD:
            status, color = "Checking fatigue...", (0, 165, 255)
        elif fatigue_probability is not None:
            status, color = "ACTIVE", (0, 220, 0)

        now = time.monotonic()
        if any_alert and now - last_alert_time >= ALERT_COOLDOWN:
            play_alert()
            last_alert_time = now

        # ----- Display values -----
        if fatigue_probability is not None:
            cv2.putText(frame, f"Fatigue: {fatigue_probability:.1%}", (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(frame, f"Active: {active_probability:.1%}", (20, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            cv2.putText(frame, f"Fatigue frames: {fatigue_frames}/{len(history)}", (20, 95),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        if ear is not None:
            cv2.putText(frame, f"EAR: {ear:.2f} | Eye closure: {eye_closed_duration:.1f}s",
                        (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        if mar is not None:
            cv2.putText(frame, f"MAR: {mar:.2f} (threshold {MAR_THRESHOLD:.2f})",
                        (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            if yawn_start_time is not None:
                cv2.putText(frame, f"Mouth-open duration: {yawn_elapsed:.1f}s", (20, 190),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.putText(frame, status, (20, height - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
        cv2.imshow("Driver Drowsiness Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    cap.release()
    face_mesh.close()
    cv2.destroyAllWindows()
