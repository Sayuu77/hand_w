import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_drawable_canvas import st_canvas

# ==============================
# CONFIGURACIÓN BÁSICA
# ==============================
st.set_page_config(
    page_title='Digit Recognition',
    page_icon='🎯',
    layout='centered'
)

# ==============================
# ESTILOS
# ==============================
st.markdown("""
<style>
    body, .stApp {
        background-color:#111827;
        color:white;
    }
    .title {
        font-size: 2.3rem;
        font-weight:600;
        text-align:center;
        margin-bottom:0.3rem;
        color:white;
    }
    .subtitle {
        font-size:1rem;
        text-align:center;
        color:#9CA3AF;
        margin-bottom:1.5rem;
    }
    .canvas-box {
        background:#000;
        padding:1rem;
        border-radius:10px;
        display:flex;
        justify-content:center;
    }
    .stButton>button {
        width:100%;
        background-color:#4F46E5;
        color:white;
        border:none;
        padding:0.8rem;
        border-radius:8px;
        font-size:1rem;
        font-weight:600;
        cursor:pointer;
    }
    .stButton>button:hover {
        background-color:#4338CA;
    }
    .result {
        font-size:4rem;
        font-weight:700;
        text-align:center;
        color:#4F46E5;
        margin-top:1rem;
    }
</style>
""", unsafe_allow_html=True)

# ==============================
# CARGAR MODELO
# ==============================
@st.cache_resource
def load_model():
    try:
        return tf.keras.models.load_model("model/handwritten.h5")
    except:
        st.error("No se pudo cargar el modelo.")
        return None

model = load_model()

# ==============================
# INTERFAZ
# ==============================
st.markdown('<div class="title">Digit Recognition</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Dibuja un dígito (0-9) y deja que la IA lo reconozca</div>', unsafe_allow_html=True)

# SLIDER DE GROSOR
stroke_width = st.slider("Grosor del pincel", 5, 30, 15)

# CANVAS
st.markdown('<div class="canvas-box">', unsafe_allow_html=True)
canvas_result = st_canvas(
    fill_color="rgba(0, 0, 0, 0)",
    stroke_width=stroke_width,
    stroke_color="#FFFFFF",
    background_color="#000000",
    width=300,
    height=300,
    drawing_mode="freedraw",
    key="canvas",
    display_toolbar=True  # → incluye lápiz, borrar, rehacer, deshacer, limpiar
)
st.markdown('</div>', unsafe_allow_html=True)

# BOTÓN
predict_btn = st.button("🔍 Predecir Dígito")

# ==============================
# PREDICCIÓN
# ==============================
def predict(image, model):
    image = ImageOps.grayscale(image)
    image = image.resize((28, 28))
    image = np.array(image, dtype="float32") / 255.0
    image = image.reshape(1, 28, 28, 1)
    preds = model.predict(image, verbose=0)
    return np.argmax(preds), np.max(preds), preds[0]

if predict_btn and canvas_result.image_data is not None:
    input_image = Image.fromarray(canvas_result.image_data.astype("uint8"), "RGBA")
    digit, conf, raw = predict(input_image, model)

    st.markdown(f'<div class="result">{digit}</div>', unsafe_allow_html=True)
    st.write(f"**Confianza:** {conf*100:.2f}%")

    with st.expander("Ver probabilidades"):
        df = pd.DataFrame({"Dígito": range(10), "Probabilidad": raw}).sort_values("Probabilidad", ascending=False)
        st.dataframe(df, use_container_width=True)

elif predict_btn:
    st.warning("Dibuja antes de predecir 😉")
