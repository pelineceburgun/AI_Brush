# export_onnx.py
import torch
import segmentation_models_pytorch as smp

# Modelin kayıtlı olduğu yol
MODEL_PATH = "models/face_segmentation/face_seg_model.pt"
ONNX_PATH = "models/face_segmentation/face_seg_model.onnx"


def convert_to_onnx():
    # 1. Modeli PyTorch olarak yükle (Senin load_segmentation mantığınla aynı)
    model = smp.UnetPlusPlus(
        encoder_name="resnet18",
        encoder_weights=None,
        in_channels=3,
        classes=3
    )

    # Checkpoint yükle
    ckpt = torch.load(MODEL_PATH, map_location="cpu")
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)

    model.eval()

    # 2. Dummy (örnek) bir input oluştur (Modelin giriş boyutu: 1 batch, 3 kanal, 256x256)
    dummy_input = torch.randn(1, 3, 256, 256)

    # 3. Export işlemi
    print("ONNX'e dönüştürülüyor...")
    torch.onnx.export(
        model,
        dummy_input,
        ONNX_PATH,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print(f"✅ Model başarıyla kaydedildi: {ONNX_PATH}")


if __name__ == "__main__":
    convert_to_onnx()