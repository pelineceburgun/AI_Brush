import numpy as np
import cv2
import os
import urllib.request
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- 1. OTOMATİK MODEL İNDİRİCİ ---
MODEL_PATH = "face_landmarker.task"
if not os.path.exists(MODEL_PATH):
    print("MediaPipe modern modeli indiriliyor (2MB)... Lütfen bekleyin.")
    url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    urllib.request.urlretrieve(url, MODEL_PATH)
    print("✅ İndirme tamamlandı!")

# --- 2. MODERN TASKS API KURULUMU (solutions modülü YOK) ---
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False,
    num_faces=1
)
detector = vision.FaceLandmarker.create_from_options(options)


def get_modern_mediapipe_lips(image_bgr):
    """Modern API ile milimetrik dudak tespiti."""
    h, w = image_bgr.shape[:2]

    # OpenCV (BGR) resmini MediaPipe (RGB) formatına çevir
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

    # Yüz noktalarını algıla
    detection_result = detector.detect(mp_image)

    mask = np.zeros((h, w), dtype=np.uint8)

    # Eğer yüz bulunduysa noktaları çiz
    if detection_result.face_landmarks:
        landmarks = detection_result.face_landmarks[0]

        # Dudakları oluşturan noktaların indeksleri
        LIPS_INDICES = [
            61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
            308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 185,
            40, 39, 37, 0, 267, 269, 270, 409, 415, 310, 311,
            312, 13, 82, 81, 80, 191, 78, 62, 76, 61
        ]

        lip_points = []
        for idx in LIPS_INDICES:
            pt = landmarks[idx]
            lip_points.append([int(pt.x * w), int(pt.y * h)])

        lip_points = np.array(lip_points, np.int32)

        # Noktaların içini beyazla doldur (kusursuz dudak şekli)
        cv2.fillPoly(mask, [lip_points], 255)

    return mask


def create_under_eye_mask(face_bbox, image_shape):
    x, y, w, h = face_bbox
    mask = np.zeros(image_shape[:2], dtype="uint8")
    y_start = int(y + h * 0.35)
    y_end = int(y + h * 0.55)
    x_start = int(x + w * 0.15)
    x_end = int(x + w * 0.85)
    mask[y_start:y_end, x_start:x_end] = 255
    return mask