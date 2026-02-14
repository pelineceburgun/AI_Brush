import torch
import segmentation_models_pytorch as smp

def load_segmentation_model(model_path, device="cpu"):

    model = smp.UnetPlusPlus(
        encoder_name="resnet18",   # ✅ DOĞRU ENCODER
        encoder_weights=None,
        in_channels=3,
        classes=3
    )

    ckpt = torch.load(model_path, map_location=device)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    else:
        state = ckpt

    model.load_state_dict(state)

    model.to(device)
    model.eval()

    print("✅ Segmentation model loaded (resnet34)")
    return model
