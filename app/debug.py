import torch

ckpt = torch.load("models/face_segmentation/face_seg_model.pt", map_location="cpu")

print(type(ckpt))

if isinstance(ckpt, dict):
    print("\nKEYS:\n", ckpt.keys())

    if "model_state_dict" in ckpt:
        sd = ckpt["model_state_dict"]
    elif "state_dict" in ckpt:
        sd = ckpt["state_dict"]
    else:
        sd = ckpt

    print("\nFIRST 20 PARAM KEYS:\n")
    for i, k in enumerate(sd.keys()):
        print(k)
        if i == 20:
            break
