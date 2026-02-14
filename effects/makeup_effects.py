import cv2
import numpy as np


# --------------------------------------------------
# LIP COLOR (PHOTOREALISTIC HSV BLENDING)
# --------------------------------------------------
def apply_lip_color_hsv(image, mask, hue_shift=165, strength=0.5):
    """
    hue_shift:
        165 -> soft pink
        0   -> red
        150 -> nude pink
    """

    result = image.copy()
    hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV).astype(np.float32)

    # Maskeyi yumuşat (kenar sertliğini alır)
    blurred_mask = cv2.GaussianBlur(mask, (21, 21), 0)
    alpha = (blurred_mask / 255.0) * strength
    alpha_3ch = np.stack([alpha]*3, axis=-1)

    # Hue değiştir
    hsv[:, :, 0] = np.where(blurred_mask > 0, hue_shift, hsv[:, :, 0])

    # Saturation artır (lipstick etkisi)
    hsv[:, :, 1] = np.where(
        blurred_mask > 0,
        np.clip(hsv[:, :, 1] * 1.4, 0, 255),
        hsv[:, :, 1]
    )

    colored = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    blended = result * (1 - alpha_3ch) + colored * alpha_3ch

    return blended.astype(np.uint8)


# --------------------------------------------------
# SMOOTH (Skin blur)
# --------------------------------------------------
def apply_smooth(image, mask, strength=0.5):

    blurred = cv2.GaussianBlur(image, (21, 21), 0)

    blurred_mask = cv2.GaussianBlur(mask, (15, 15), 0)
    alpha = (blurred_mask / 255.0) * strength
    alpha_3ch = np.stack([alpha]*3, axis=-1)

    result = image * (1 - alpha_3ch) + blurred * alpha_3ch

    return result.astype(np.uint8)


# --------------------------------------------------
# CONCEAL (Under eye brighten)
# --------------------------------------------------
def apply_conceal(image, mask, strength=0.5):

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    L, A, B = cv2.split(lab)

    blurred_mask = cv2.GaussianBlur(mask, (15, 15), 0)
    lift = (blurred_mask / 255.0) * (25 * strength)

    L = np.clip(L + lift, 0, 255)

    merged = cv2.merge([L, A, B])
    return cv2.cvtColor(merged.astype(np.uint8), cv2.COLOR_LAB2BGR)


# --------------------------------------------------
# HIGHLIGHT / CONTOUR
# --------------------------------------------------
def apply_highlight_contour(image, mask, strength=0.3, mode="highlight"):

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    L, A, B = cv2.split(lab)

    blurred_mask = cv2.GaussianBlur(mask, (21, 21), 0)
    delta = (blurred_mask / 255.0) * (30 * strength)

    if mode == "highlight":
        L = np.clip(L + delta, 0, 255)
    else:
        L = np.clip(L - delta, 0, 255)

    merged = cv2.merge([L, A, B])
    return cv2.cvtColor(merged.astype(np.uint8), cv2.COLOR_LAB2BGR)


# --------------------------------------------------
# CENTRAL CONTROLLER
# --------------------------------------------------
def apply_action(image, mask, action):

    # Boyut kontrolü
    if mask.shape[:2] != image.shape[:2]:
        mask = cv2.resize(
            mask,
            (image.shape[1], image.shape[0]),
            interpolation=cv2.INTER_NEAREST
        )

    if not np.any(mask):
        return image

    action_type = action.get("action")
    strength = action.get("strength", 0.5)

    # ---------------- LIPS ----------------
    if action_type == "colorize":

        color_name = action.get("color", "pink")

        if color_name == "red":
            hue = 0
        elif color_name == "nude":
            hue = 150
        else:  # default pink
            hue = 165

        return apply_lip_color_hsv(
            image,
            mask,
            hue_shift=hue,
            strength=strength
        )

    # ---------------- SKIN SMOOTH ----------------
    elif action_type == "smooth":
        return apply_smooth(image, mask, strength)

    # ---------------- CONCEAL ----------------
    elif action_type == "conceal":
        return apply_conceal(image, mask, strength)

    # ---------------- HIGHLIGHT ----------------
    elif action_type == "highlight":
        return apply_highlight_contour(image, mask, strength, "highlight")

    # ---------------- CONTOUR ----------------
    elif action_type == "contour":
        return apply_highlight_contour(image, mask, strength, "contour")

    return image
