import re

def interpret_prompt(prompt: str):
    prompt = prompt.lower()

    action = {
        "region": None,
        "action": None,
        "strength": 0.5,   # default
        "color": None
    }

    # -------- REGION --------
    if any(k in prompt for k in ["göz alt", "under eye"]):
        action["region"] = "under_eye"

    elif any(k in prompt for k in ["dudak", "lip"]):
        action["region"] = "lips"

    elif any(k in prompt for k in ["cilt", "ten", "skin"]):
        action["region"] = "skin"

    # -------- ACTION --------
    if any(k in prompt for k in ["kapat", "gizle", "blur", "pürüzsüz"]):
        action["action"] = "smooth"

    elif any(k in prompt for k in ["renklendir", "boya", "color"]):
        action["action"] = "colorize"

    elif any(k in prompt for k in ["aydınlat", "bright"]):
        action["action"] = "brighten"

    # -------- STRENGTH --------
    if any(k in prompt for k in ["çok", "fazla"]):
        action["strength"] = 0.7

    elif any(k in prompt for k in ["biraz", "hafif", "az"]):
        action["strength"] = 0.3

    # -------- COLOR (basic) --------
    if "red" in prompt:
        action["color"] = (0, 0, 255)

    elif "pink" in prompt:
        action["color"] = (203, 192, 255)

    return action
