import segmentation_models_pytorch as smp

def build_model():
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,   # pretrained değil — kendi eğittiğimiz
        in_channels=3,
        classes=3
    )
    return model
