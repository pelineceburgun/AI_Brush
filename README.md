

# ✨ AI Makeup Studio: Natural Language to Deterministic Image Transformation

## 📌 Overview

**AI Makeup Studio** is a modular AI system that bridges human natural language intent with a deterministic computer vision and image editing pipeline.

Instead of relying on generative AI (like Diffusion models) which can hallucinate or alter user identity, this project uses a structured multi-component architecture: **an LLM for semantic understanding** and a **Custom CV Pipeline for pixel-level, deterministic transformations.**

It shifts the traditional photo-editing UX from "slider-based manipulation" to "natural language-based interaction."

## 🏗️ System Architecture & Core Components

This project is not a simple script or a basic API wrapper; it is a small-scale **AI Product Backend** composed of three main layers:

### 1. Language Understanding (The Intent Parser)

* **Role:** The LLM does *not* see or generate images. It acts strictly as an intent parser.
* **Mechanism:** It translates unpredictable human language into a structured, machine-readable JSON object.
* **Why?** This isolates the probabilistic nature of LLMs from the rendering engine, ensuring stable, testable, and modular application logic.

### 2. The Perception Layer (Hybrid Computer Vision)

To manipulate an image, the system first needs pixel-level semantic understanding.I use a **Hybrid CV Approach**:

* **Custom Semantic Segmentation (ONNX):** I trained a custom semantic segmentation model (U-Net architecture) from scratch. To achieve real-time CPU performance, the PyTorch model was exported to **ONNX format**, optimizing the inference speed dramatically. This handles broad regions like `background`, `skin`, and `hair`.
* **MediaPipe Face Mesh:** For sub-millimeter precision on delicate areas (like lips and under-eye regions), the system dynamically routes the task to Google's MediaPipe Tasks API.
* **Result:** A flawless binary mask map () where every pixel's class is strictly defined.

### 3. Deterministic Editing Engine (Masked Transformation)

Once the mask and JSON intent are received, the rendering engine takes over.

* **Mechanism:** Transformations are strictly mathematical and deterministic.
* *Example (Smoothing):* `result = (1 - alpha) * original + alpha * blurred`


* **Why?** Pixels outside the mask remain 100% untouched. The same input always produces the exact same output. No generative artifacts, no loss of identity.

## 🚀 How It Works (The Pipeline)

1. **Input:** User uploads an image and types a prompt (e.g., *"Make my lips dark red and hide my under-eye bags"*).
2. **LLM Routing:** The system pings the LLM, which outputs a structured routing JSON:
```json
{
  "actions": [
    { "region": "lips", "action": "colorize", "color": [0, 0, 150], "strength": 0.8 },
    { "region": "under_eye", "action": "brighten", "strength": 0.5 }
  ]
}

```


3. **Vision Inference:** The CV dispatcher reads the regions. It triggers MediaPipe for the `lips` mask and the custom ONNX model for the `skin` mask.
4. **Render:** The OpenCV-based engine applies the requested mathematical transformations (Multiply blend, Gaussian blur, CLAHE) strictly within the generated masks.
5. **Output:** A fast, natural-looking, high-fidelity edited image presented on a Streamlit UI with a Before/After slider.

## 🛠️ Tech Stack

* **Frontend:** Streamlit
* **Deep Learning / Vision:**  U-Net, ResNet, ONNXRuntime, Google MediaPipe, OpenCV, NumPy
* **Language Model:** OpenRouter API (Gemini / Llama 3) for JSON structured output
* **Environment:** Python 3.13

## 💡 What Makes This Architecture Right?

From an engineering perspective, building an "AI Agent" that does everything end-to-end is error-prone. This architecture respects the principle of separation of concerns:

* **CV Engine is Deterministic:** Highly stable, easy to test, fast.
* **LLM is the Semantic Layer:** High tolerance for natural language variations, drastically reducing UI complexity.
* **Product-Ready:** The modularity allows swapping the LLM or upgrading the Segmentation model independently without breaking the pipeline.

## ⚙️ Installation & Usage

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/ai-makeup-studio.git
cd ai-makeup-studio

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your OpenRouter API Key
# Create a .env file or export it to your environment variables
export OPENROUTER_API_KEY="your-api-key-here"

# 4. Run the Streamlit Application
streamlit run app.py

```


