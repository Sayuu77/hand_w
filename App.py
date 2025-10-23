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
# ESTILOS MINIMALISTAS
# ==============================
st.markdown("""
<style>
    .stApp {
        background: #0f0f0f;
        color: #ffffff;
    }
    
    /* Centrar todo el contenido */
    .main .block-container {
        max-width: 500px;
        padding: 2rem 1rem;
    }
    
    .title {
        font-size: 2.5rem;
        font-weight: 300;
        text-align: center;
        margin-bottom: 0.5rem;
        color: #ffffff;
        letter-spacing: -0.5px;
    }
    
    .subtitle {
        font-size: 1rem;
        text-align: center;
        color: #888;
        margin-bottom: 2rem;
        font-weight: 300;
    }
    
    .canvas-wrapper {
        display: flex;
        justify-content: center;
        align-items: center;
        background: #000000;
        border-radius: 12px;
        padding: 0;
        margin: 0 auto 2rem auto;
        width: 400px;
        height: 400px;
        border: 1px solid #333;
    }
    
    .stButton>button {
        width: 100%;
        background: #ffffff;
        color: #000000;
        border: none;
        padding: 1rem 2rem;
        border-radius: 8px;
        font-size: 1rem;
        font-weight: 500;
        cursor: pointer;
        transition: all 0.2s ease;
        margin: 1rem 0;
    }
    
    .stButton>button:hover {
        background: #f0f0f0;
        transform: translateY(-1px);
    }
    
    .result {
        font-size: 5rem;
        font-weight: 200;
        text-align: center;
        color: #ffffff;
        margin: 1rem 0;
        letter-spacing: -2px;
    }
    
    .confidence {
        text-align: center;
        color: #888;
        font-size: 0.9rem;
        margin-bottom: 1rem;
    }
    
    .slider-container {
        margin: 1.5rem 0;
    }
    
    /* Personalizar el slider */
    .stSlider {
        margin: 1rem 0;
    }
    
    .stSlider > div > div {
        background: #333;
    }
    
    .stSlider > div > div > div {
        background: #ffffff;
    }
    
    /* Ocultar elementos innecesarios */
    .st-emotion-cache-1kyxreq {
        justify-content: center;
    }
    
    /* Centrar el expander */
    .st-expander {
        margin: 1rem 0;
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
# INTERFAZ CENTRADA
# ==============================
st.markdown('<div class="title">Digit Recognition</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Draw a digit and let AI recognize it</div>', unsafe_allow_html=True)

# SLIDER DE GROSOR - Centrado
stroke_width = st.slider(
    "Brush thickness", 
    5, 30, 15,
    help="Adjust the stroke width"
)


# CANVAS - Centrado y del mismo tamaño que el contenedor negro
st.markdown('<div class="canvas-wrapper">', unsafe_allow_html=True)
canvas_result = st_canvas(
    fill_color="rgba(0, 0, 0, 0)",
    stroke_width=stroke_width,
    stroke_color="#FFFFFF",
    background_color="#000000",
    width=400,  # Mismo ancho que el contenedor
    height=400, # Mismo alto que el contenedor
    drawing_mode="freedraw",
    key="canvas",
    display_toolbar=True
)
st.markdown('</div>', unsafe_allow_html=True)

# BOTÓN CENTRADO
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    predict_btn = st.button("🔍 Predict Digit", use_container_width=True)

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

    # Mostrar resultado centrado
    st.markdown(f'<div class="result">{digit}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="confidence">Confidence: {conf*100:.1f}%</div>', unsafe_allow_html=True)

    # Probabilidades en expander centrado
    with st.expander("View probabilities"):
        df = pd.DataFrame({
            "Digit": range(10), 
            "Probability": raw
        }).sort_values("Probability", ascending=False)
        
        # Estilizar el dataframe
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Digit": st.column_config.NumberColumn("Digit", format="%d"),
                "Probability": st.column_config.ProgressColumn(
                    "Probability",
                    format="%.3f",
                    min_value=0,
                    max_value=1,
                )
            }
        )

elif predict_btn:
    st.warning("Please draw a digit first ✏️")

# ==============================
# FOOTER MINIMALISTA
# ==============================
st.markdown("""
<div style="text-align: center; color: #444; margin-top: 3rem; padding: 1rem; font-size: 0.8rem;">
    AI-powered digit recognition • Built with Streamlit & TensorFlow
</div>
""", unsafe_allow_html=True)

