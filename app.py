import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io


from models.segment_infer import SegmentationInference
from llm.llm_client import parse_prompt_with_llm
from app.main import run_pipeline  # main.py'deki run_pipeline fonksiyonunu import etmelisin
from utils.face_regions import create_under_eye_mask


st.set_page_config(page_title="AIBrush", layout="wide", page_icon="💄")
ONNX_MODEL_PATH = "models/face_segmentation/face_seg_model.onnx"


# --- CACHING (Modeli her refresh'te tekrar yüklememek için) ---
@st.cache_resource
def load_model():
    return SegmentationInference(ONNX_MODEL_PATH)


seg_engine = load_model()

# --- CSS (Biraz makyaj yapalım arayüze) ---
st.markdown("""
<style>
    .stButton>button {
        width: 100%;
        border-radius: 20px;
        background-color: #FF4B4B;
        color: white;
    }
    .main-header {
        font-size: 2.5rem;
        color: #FF4B4B;
        text-align: center;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# --- ARAYÜZ ---
st.markdown('<p class="main-header"> AI Image Editor ✨</p>', unsafe_allow_html=True)
st.write("Upload your photo, tell what you want, let AI help you!")

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("1. Choose a foto")
    uploaded_file = st.file_uploader("Upload a photo", type=['jpg', 'png', 'jpeg'])

    # Kamera seçeneği
    enable_camera = st.checkbox("Use the camera")
    img_file_buffer = st.camera_input("Take a photo", disabled=not enable_camera) if enable_camera else None

    # Hangisi doluysa onu al
    input_image = uploaded_file if uploaded_file else img_file_buffer

    st.subheader("2. Enter a prompt")
    prompt = st.text_area("How do you want to be seen?",
                          placeholder="E.g.: Color my lips and make my skin smoother.")

    st.markdown("---")
    st.write("Or a Quick Choice:")

    # Hızlı Butonlar (Prompt yazmaya üşenenler için)
    c1, c2 = st.columns(2)
    if c1.button("🌸 Natural Make-Up"):
        prompt = "Smooth my skin and make my lips soft pink"
    if c2.button(" Night make-up"):
        prompt = "Make my lips dark red and brighten my eyes"

    run_btn = st.button("✨ Apply", type="primary")

with col2:
    if input_image is not None:
        # Streamlit file -> OpenCV Image dönüşümü
        image_bytes = input_image.getvalue()
        image = np.array(Image.open(io.BytesIO(image_bytes)))

        # RGB -> BGR (OpenCV formatı)
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Face Detection Kontrolü
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(100, 100))

        if len(faces) > 0:
            face_bbox = faces[0]

            # RUN BUTONUNA BASILINCA
            if run_btn and prompt:
                with st.spinner('AI is thinking... (LLM)'):
                    # 1. LLM Çağrısı
                    llm_response = parse_prompt_with_llm(prompt)

                if llm_response and "actions" in llm_response:
                    with st.spinner('Processing... (ONNX)'):
                        # 2. Segmentasyon
                        mask_small = seg_engine.predict(image_bgr)

                        # 3. Pipeline
                        final_bgr = run_pipeline(
                            image=image_bgr,
                            mask_small=mask_small,
                            actions=llm_response["actions"],
                            face_bbox=face_bbox
                        )

                        # Sonucu Göster (Before/After Slider)
                        from streamlit_image_comparison import image_comparison

                        final_rgb = cv2.cvtColor(final_bgr, cv2.COLOR_BGR2RGB)

                        st.success("Done!")


                        image_comparison(
                            img1=image,
                            img2=final_rgb,
                            label1="Original",
                            label2="AI Enhanced Editing",
                            width=700,
                            starting_position=50,
                            show_labels=True
                        )

                        # İndirme Butonu
                        res_pil = Image.fromarray(final_rgb)
                        buf = io.BytesIO()
                        res_pil.save(buf, format="JPEG")
                        byte_im = buf.getvalue()

                        st.download_button(
                            label="📥 Download the result",
                            data=byte_im,
                            file_name="ai_makeup_result.jpg",
                            mime="image/jpeg"
                        )

                        # LLM'in ne anladığını debug olarak göster (opsiyonel)
                        with st.expander("what did AI do? (Debug)"):
                            st.json(llm_response)

                else:
                    st.error("LLM couldn't understand the prompt. Please try again.")

            elif not run_btn:
                # Henüz çalışmadıysa sadece orijinali göster
                st.image(image, caption="Uploaded Images", use_container_width=True)

        else:
            st.warning("⚠️ Couldn't detect a face. Please upload a better photo.")

    else:
        # Boş durum
        st.info("👈 Upload a photo from left in order to start.")