import requests
import json
import re
import streamlit as st

api_key = st.secrets["OPENROUTER_API_KEY"]  # Kendi Key'in
SITE_URL = "https://makeup-app.local"
APP_NAME = "AIBrush"


def extract_json_from_text(text):
    try:
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
        if match: return json.loads(match.group(1))

        match = re.search(r"(\{.*\})", text, re.DOTALL)
        if match: return json.loads(match.group(1))

        return json.loads(text)
    except:
        return None


def parse_prompt_with_llm(user_prompt):
    url = "https://openrouter.ai/api/v1/chat/completions"

    system_prompt = """
    You are an AI makeup assistant. Convert the user's natural language makeup request into a JSON command.

    CAPABILITIES:
    - REGIONS: "lips", "skin", "under_eye"
    - ACTIONS: 
      1. "colorize" (needs 'color' as [R,G,B] and 'strength' 0.0-1.0)
      2. "smooth" (needs 'strength' 0.0-1.0)
      3. "brighten" (needs 'strength' 0.0-1.0)

    RULES:
    1. Output strictly and ONLY a valid JSON object.
    2. Do NOT wrap the JSON in markdown blocks. Do NOT add conversational text.

    EXPECTED FORMAT:
    {"actions": [{"region": "lips", "action": "colorize", "color": [0,0,255], "strength": 0.8}]}
    """

    payload = {

        "model": "openrouter/free",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.1
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": SITE_URL,
        "X-Title": APP_NAME,
    }

    try:
        print("--- API İsteği Gönderiliyor (Otomatik Ücretsiz Model) ---")
        response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=15)

        if response.status_code == 200:
            result = response.json()
            if "error" in result:
                print(f"API HATASI: {result['error']['message']}")
                return None

            raw_content = result['choices'][0]['message']['content']


            used_model = result.get('model', 'Bilinmeyen Model')
            print(f" İsteği Yanıtlayan Model: {used_model}")
            print(f"DEBUG LLM Raw: {raw_content}")

            return extract_json_from_text(raw_content)
        else:
            print(f"API HATASI (HTTP {response.status_code}): {response.text}")
            return None

    except Exception as e:
        print(f"Bağlantı Hatası: {e}")
        return None


if __name__ == "__main__":
    print("Result:", parse_prompt_with_llm("I want soft pink lips and clear skin"))