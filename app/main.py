
# main.py
import cv2
import time
import numpy as np
from models.segment_infer import SegmentationInference  # Yeni Class yapısı
from llm.llm_client import parse_prompt_with_llm
from effects.makeup_effects import apply_action
from utils.face_regions import create_under_eye_mask, get_modern_mediapipe_lips

# -------- CONFIG --------
IMAGE_PATH = "data/samples/test.jpg"
ONNX_MODEL_PATH = "models/face_segmentation/face_seg_model.onnx"


# ------------------------

def run_pipeline(image, mask_small, actions, face_bbox):
    result = image.copy()

    # Cilt için senin U-Net modelin (Class 1)
    full_unet_mask = cv2.resize(mask_small, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

    for action in actions:
        target_mask = None

        if action["region"] == "lips":
            # İŞTE BURASI: Artık modern MediaPipe çalışacak!
            target_mask = get_modern_mediapipe_lips(image)

        elif action["region"] == "skin":
            target_mask = (full_unet_mask == 1).astype("uint8") * 255

        elif action["region"] == "under_eye":
            target_mask = create_under_eye_mask(face_bbox, image.shape)

        if target_mask is not None and np.any(target_mask):
            result = apply_action(result, target_mask, action)

    return result

def main():
    # 1. Modeli Başlat (ONNX) - Sadece bir kere yapılır
    print("Loading ONNX model...")
    seg_engine = SegmentationInference(ONNX_MODEL_PATH)

    # 2. Resmi Oku
    image = cv2.imread(IMAGE_PATH)
    if image is None: raise ValueError("Image not found")

    # 3. Face Detection (Haar Cascade - Şimdilik kalsın ama ilerde MediaPipe'a geç)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(100, 100))

    if len(faces) == 0: raise RuntimeError("No face detected")
    face_bbox = faces[0]

    # 4. Segmentasyon (Inference)
    print("Running segmentation (ONNX)...")
    t_start = time.time()
    mask_256 = seg_engine.predict(image)  # Artık numpy array dönüyor (256, 256)
    print(f"Segmentation took: {time.time() - t_start:.4f}s")

    # 5. LLM Prompt (Burası en yavaş kısımdır, network latency)
    prompt = input("What do you want to do? > ")

    # --- LLM Dummy Modu (Test ederken LLM beklememek için) ---
    # llm_result = {"actions": [{"region": "lips", "action": "colorize", "color": (0,0,255), "strength": 0.6}]}
    llm_result = parse_prompt_with_llm(prompt)

    if llm_result and "actions" in llm_result:
        try:
            final_image = run_pipeline(
                image=image,
                mask_small=mask_256,  # Küçük maskeyi gönderiyoruz
                actions=llm_result["actions"],
                face_bbox=face_bbox
            )

            cv2.imshow("AI Makeup Result (Fast)", final_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"Pipeline Hatası: {e}")
    else:
        print("Geçersiz LLM yanıtı.")


if __name__ == "__main__":
    main()
