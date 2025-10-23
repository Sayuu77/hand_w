import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from streamlit_drawable_canvas import st_canvas

# Configuración de la página
st.set_page_config(
    page_title='Reconocimiento de Dígitos Manuscritos',
    page_icon='🎯',
    layout='wide',
    initial_sidebar_state='expanded'
)

# Tema ultra minimalista y elegante
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 50%, #16213e 100%);
        color: #e2e8f0;
    }
    
    /* Ocultar elementos de Streamlit */
    .stApp > header {
        background-color: transparent;
    }
    
    .stApp > div:first-child {
        background: linear-gradient(90deg, #6366f1, #8b5cf6);
    }
    
    .main-title {
        font-size: 3rem;
        text-align: center;
        background: linear-gradient(45deg, #f0f4ff, #c7d2fe, #a5b4fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
        font-weight: 300;
        letter-spacing: -1px;
        font-family: 'Inter', sans-serif;
    }
    
    .subtitle {
        text-align: center;
        color: #94a3b8;
        margin-bottom: 3rem;
        font-size: 1.1rem;
        font-weight: 300;
        letter-spacing: 0.5px;
    }
    
    .canvas-container {
        background: rgba(15, 23, 42, 0.7);
        backdrop-filter: blur(20px);
        border-radius: 24px;
        padding: 2rem;
        margin: 0 auto;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 
            0 8px 32px rgba(0, 0, 0, 0.3),
            inset 0 1px 0 rgba(255, 255, 255, 0.1);
        max-width: 500px;
    }
    
    .result-card {
        background: rgba(15, 23, 42, 0.7);
        backdrop-filter: blur(20px);
        border-radius: 24px;
        padding: 2.5rem;
        margin: 2rem auto;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 
            0 8px 32px rgba(0, 0, 0, 0.3),
            inset 0 1px 0 rgba(255, 255, 255, 0.1);
        text-align: center;
        max-width: 400px;
    }
    
    .predict-button {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 16px !important;
        padding: 1rem 2rem !important;
        font-weight: 500 !important;
        font-size: 1.1rem !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        width: 100% !important;
        margin: 1rem 0 !important;
        box-shadow: 0 4px 20px rgba(99, 102, 241, 0.3);
    }
    
    .predict-button:hover {
        transform: translateY(-3px) !important;
        box-shadow: 
            0 8px 30px rgba(99, 102, 241, 0.4),
            0 2px 8px rgba(99, 102, 241, 0.2) !important;
    }
    
    .sidebar-content {
        background: rgba(15, 23, 42, 0.7);
        backdrop-filter: blur(20px);
        border-radius: 20px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .digit-display {
        font-size: 8rem;
        text-align: center;
        background: linear-gradient(135deg, #f0f4ff, #c7d2fe, #a5b4fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 200;
        margin: 1rem 0;
        font-family: 'Inter', sans-serif;
    }
    
    .confidence-ring {
        width: 120px;
        height: 120px;
        border-radius: 50%;
        background: conic-gradient(#10b981 0%, #334155 0%);
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 2rem auto;
        position: relative;
        box-shadow: 0 4px 20px rgba(16, 185, 129, 0.3);
    }
    
    .confidence-inner {
        width: 90px;
        height: 90px;
        background: #0f172a;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.2rem;
        font-weight: 600;
        color: #10b981;
    }
    
    .canvas-center {
        display: flex;
        justify-content: center;
        align-items: center;
        padding: 1rem;
    }
    
    .stSlider > div > div {
        color: #6366f1;
    }
    
    .feature-text {
        color: #cbd5e1;
        font-size: 0.9rem;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .github-link {
        background: linear-gradient(135deg, #1f2937 0%, #374151 100%);
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        margin-top: 1rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# Función para predecir dígito
@st.cache_resource
def load_model():
    """Cargar el modelo una sola vez usando cache"""
    try:
        model = tf.keras.models.load_model("model/handwritten.h5")
        return model
    except Exception as e:
        st.error(f"Error cargando el modelo: {e}")
        return None

def predictDigit(image, model):
    """Predecir dígito y retornar probabilidades"""
    try:
        image = ImageOps.grayscale(image)
        img = image.resize((28, 28))
        img = np.array(img, dtype='float32')
        img = img / 255.0
        
        # Preparar imagen para el modelo
        img = img.reshape((1, 28, 28, 1))
        
        # Hacer predicción
        predictions = model.predict(img, verbose=0)
        predicted_digit = np.argmax(predictions[0])
        confidence = np.max(predictions[0])
        
        return predicted_digit, confidence, predictions[0]
    except Exception as e:
        st.error(f"Error en predicción: {e}")
        return None, None, None

# Título principal ultra minimalista
st.markdown('<div class="main-title">Digit Recognition</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Dibuja un dígito y descubre la magia de la IA</div>', unsafe_allow_html=True)

# Cargar modelo
model = load_model()

# Layout principal centrado
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    # Contenedor del canvas elegante
    st.markdown('<div class="canvas-container">', unsafe_allow_html=True)
    
    # Configuración minimalista
    drawing_mode = "freedraw"
    
    # Slider elegante
    stroke_width = st.slider(
        '🎨 Grosor del pincel',
        min_value=5,
        max_value=25,
        value=15,
        help="Ajusta el grosor del trazo"
    )
    
    # Canvas de dibujo centrado
    st.markdown('<div class="canvas-center">', unsafe_allow_html=True)
    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 0.0)",
        stroke_width=stroke_width,
        stroke_color="#FFFFFF",
        background_color="#000000",
        height=300,
        width=300,
        drawing_mode=drawing_mode,
        key="canvas",
        display_toolbar=True
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Botón de predicción elegante
    predict_btn = st.button('🚀 **Predecir Dígito**', key='predict', use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Mostrar resultados
if predict_btn and canvas_result.image_data is not None:
    with st.spinner("🔍 Analizando tu dibujo..."):
        # Procesar imagen y predecir
        input_numpy_array = np.array(canvas_result.image_data)
        input_image = Image.fromarray(input_numpy_array.astype('uint8'), 'RGBA')
        
        if model is not None:
            predicted_digit, confidence, all_predictions = predictDigit(input_image, model)
            
            if predicted_digit is not None:
                # Mostrar resultado en tarjeta elegante
                st.markdown('<div class="result-card">', unsafe_allow_html=True)
                
                # Dígito predicho
                st.markdown(f'<div class="digit-display">{predicted_digit}</div>', unsafe_allow_html=True)
                
                # Anillo de confianza
                confidence_percent = int(confidence * 100)
                st.markdown(f'''
                <div class="confidence-ring" style="background: conic-gradient(#10b981 {confidence_percent}%, #334155 {confidence_percent}%);">
                    <div class="confidence-inner">
                        {confidence_percent}%
                    </div>
                </div>
                ''', unsafe_allow_html=True)
                
                # Mensaje de confianza
                if confidence > 0.8:
                    st.success("✨ **Alta precisión** - ¡Excelente dibujo!")
                elif confidence > 0.5:
                    st.info("📊 **Buena precisión** - Resultado confiable")
                else:
                    st.warning("🤔 **Baja confianza** - Intenta dibujar más claro")
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Probabilidades en expander elegante
                with st.expander("📈 Ver detalles de la predicción", expanded=False):
                    # Crear dataframe de probabilidades
                    prob_df = pd.DataFrame({
                        'Dígito': range(10),
                        'Probabilidad': all_predictions
                    }).sort_values('Probabilidad', ascending=False)
                    
                    # Mostrar top 3 predicciones
                    st.markdown("**Top 3 predicciones:**")
                    for i, (_, row) in enumerate(prob_df.head(3).iterrows()):
                        prob_percent = row['Probabilidad'] * 100
                        is_main = i == 0
                        color = "#10b981" if is_main else "#6b7280"
                        emoji = "🎯" if is_main else "📊"
                        
                        st.markdown(f"""
                        {emoji} **Dígito {int(row['Dígito'])}**: `{prob_percent:.1f}%`
                        """)
                
elif predict_btn:
    st.warning("🎨 **Por favor dibuja un dígito en el canvas antes de predecir**")

# Sidebar ultra minimalista
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h2 style="background: linear-gradient(135deg, #f0f4ff, #c7d2fe); 
                  -webkit-background-clip: text; 
                  -webkit-text-fill-color: transparent;
                  font-weight: 300;
                  margin: 0;">
            🤖 Acerca de
        </h2>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
    st.markdown("### 🧠 Tecnología")
    st.markdown("""
    <div class="feature-text">
    Red Neuronal Convolucional entrenada con el dataset MNIST
    </div>
    """, unsafe_allow_html=True)
    
    # Métricas elegantes
    col_metric1, col_metric2 = st.columns(2)
    with col_metric1:
        st.metric("Precisión", ">98%")
    with col_metric2:
        st.metric("Dataset", "60K imágenes")
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
    st.markdown("### 💡 Consejos")
    st.markdown("""
    <div class="feature-text">
    • Dibuja en el centro<br>
    • Usa trazos definidos<br>
    • Ocupa todo el espacio<br>
    • Evita dibujar muy pequeño
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Créditos elegantes
    st.markdown('<div class="github-link">', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center;">
        <div style="color: #94a3b8; font-size: 0.9rem; margin-bottom: 0.5rem;">
            Desarrollado por Vinay Uniyal
        </div>
        <a href="https://github.com/Vinay2022/Handwritten-Digit-Recognition" 
           target="_blank" 
           style="color: #6366f1; text-decoration: none; font-size: 0.8rem;">
           ⭐ Ver en GitHub
        </a>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Footer minimalista
st.markdown("""
<div style="text-align: center; color: #475569; padding: 3rem; font-size: 0.8rem;">
    <p>Hecho con ❤️ usando Streamlit & TensorFlow</p>
</div>
""", unsafe_allow_html=True)
