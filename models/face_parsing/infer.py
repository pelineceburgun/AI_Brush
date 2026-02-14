import torch
import cv2
import numpy as np
from torchvision import transforms
from .model import BiSeNet


# Class IDs (önemli olanlar)
SKIN_ID = 1
UPPER_LIP_ID = 7
LOWER_LIP_ID = 8


def parse_face(image_bgr, model_path):
    device = "cpu"

    model = BiSeNet(n_classes=19)
    state = torch.load(model_path, map_location=device)

    if "state_dict" in state:
        state = state["state_dict"]
    elif "model" in state:
        state = state["model"]

    model_dict = model.state_dict()

    filtered_state = {
        k: v
        for k, v in state.items()
        if k in model_dict and v.shape == model_dict[k].shape
    }

    model_dict.update(filtered_state)
    model.load_state_dict(model_dict)

    print(f"Loaded {len(filtered_state)} layers from checkpoint")



    model.eval()

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (256, 256))

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5])
    ])

    input_tensor = transform(image_resized).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)[0]
        parsing = output.argmax(0).cpu().numpy()

    parsing = cv2.resize(
        parsing.astype(np.uint8),
        (image_bgr.shape[1], image_bgr.shape[0]),
        interpolation=cv2.INTER_NEAREST
    )

    skin_mask = (parsing == SKIN_ID).astype(np.uint8) * 255
    lip_mask = np.logical_or(
        parsing == UPPER_LIP_ID,
        parsing == LOWER_LIP_ID
    ).astype(np.uint8) * 255

    return skin_mask, lip_mask
