import cv2
import numpy as np
import onnxruntime as ort


class SegmentationInference:
    def __init__(self, model_path):
        # Oturum (session) bir kere başlatılır, bellekte tutulur
        self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def predict(self, image_bgr):
        # 1. Preprocessing (Resize & Normalize)
        img_resized = cv2.resize(image_bgr, (256, 256))

        # BGR -> RGB ve HWC -> CHW dönüşümü
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_transposed = np.transpose(img_rgb, (2, 0, 1))  # (3, 256, 256)

        # Normalize (0-1 arası) ve Batch boyutu ekle
        input_tensor = np.expand_dims(img_transposed, axis=0).astype(np.float32) / 255.0

        # 2. ONNX Inference (PyTorch'tan çok daha hızlıdır)
        outputs = self.session.run([self.output_name], {self.input_name: input_tensor})

        # 3. Postprocessing (Argmax)
        # outputs[0] shape: (1, 3, 256, 256)
        mask_idx = np.argmax(outputs[0], axis=1)[0]  # Çıktı: (256, 256)

        return mask_idx.astype(np.uint8)

