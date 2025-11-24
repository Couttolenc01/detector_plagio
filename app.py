import streamlit as st
import joblib
import os
import numpy as np

# --------------------------
# Cargar modelos
# --------------------------
@st.cache_resource
def load_models():
    archivos = ["encoder_plagio.pkl", "umbrales_plagio.pkl"]
    for archivo in archivos:
        if not os.path.exists(archivo):
            st.error(f"Archivo no encontrado: {archivo}")
            st.info("Ejecuta: `python detector_simple_bert.py` para generar los modelos.")
            st.stop()
    
    encoder = joblib.load("encoder_plagio.pkl")
    thresholds = joblib.load("umbrales_plagio.pkl")
    return encoder, thresholds

# Cargar el modelo (intento de carga)
try:
    encoder, thresholds = load_models()
    modelo_cargado = True
except Exception as e:
    st.error(f"Error al cargar los modelos: {e}")
    modelo_cargado = False

# --------------------------
# Funciones de predicción
# --------------------------

def compute_similarity(emb1, emb2):
    """Calcula la similitud coseno entre dos embeddings"""
    return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

def classify_by_threshold(similarity, thresholds):
    """Clasifica según umbrales definidos"""
    if similarity >= thresholds['cut']:
        return 'cut'
    elif similarity >= thresholds['light']:
        return 'light'
    elif similarity >= thresholds['heavy']:
        return 'heavy'
    else:
        return 'non'

def calcular_confianza(similarity, thresholds, predicted):
    """Calcula la confianza de la predicción"""
    if predicted == 'cut':
        dist_from_threshold = similarity - thresholds['cut']
        confidence = min(1.0, 0.7 + dist_from_threshold * 10)
    elif predicted == 'light':
        mid = (thresholds['light'] + thresholds['cut']) / 2
        dist = abs(similarity - mid)
        confidence = max(0.5, 1.0 - dist * 5)
    elif predicted == 'heavy':
        mid = (thresholds['heavy'] + thresholds['light']) / 2
        dist = abs(similarity - mid)
        confidence = max(0.5, 1.0 - dist * 5)
    else:  # non
        dist_from_threshold = thresholds['heavy'] - similarity
        confidence = min(1.0, 0.7 + dist_from_threshold * 10)
    
    return max(0.3, min(1.0, confidence))

def predecir_plagio(original, sospechoso):
    # Generar embeddings con BERT
    emb1 = encoder.encode([original])[0]
    emb2 = encoder.encode([sospechoso])[0]
    
    # Calcular similitud coseno
    similarity = compute_similarity(emb1, emb2)
    
    # Clasificar con umbrales
    prediction = classify_by_threshold(similarity, thresholds)
    
    # Calcular confianza
    confidence = calcular_confianza(similarity, thresholds, prediction)
    
    return prediction, similarity, confidence, thresholds

# --------------------------
# Interfaz de usuario con Streamlit
# --------------------------

st.set_page_config(page_title="Detector de Plagio", page_icon="🧠")
st.title("Detector de Plagio – Modelo Clough")

st.markdown("""
Este sistema utiliza embeddings de BERT para detectar similitudes semánticas y plagio entre dos textos.  
**Categorías:**
- **non**: No hay plagio.
- **light**: Paráfrasis ligera.
- **heavy**: Paráfrasis fuerte.
- **cut**: Copia directa con cambios mínimos.
""")

# Verificación de modelos cargados
if not modelo_cargado:
    st.warning("Por favor, entrena el modelo primero ejecutando: `python train_clasificador_clough.py`")
    st.stop()

# Mostrar clases del modelo
with st.expander("Información del modelo"):
    st.write(f"Clases entrenadas: {['non', 'light', 'heavy', 'cut']}")

col1, col2 = st.columns(2)

with col1:
    texto_original = st.text_area("Texto original", height=250)

with col2:
    texto_sospechoso = st.text_area("Texto sospechoso", height=250)

if st.button("Analizar plagio"):
    if not texto_original.strip() or not texto_sospechoso.strip():
        st.error("Por favor ingresa ambos textos.")
    else:
        with st.spinner("Analizando..."):
            prediccion, similitud, confianza, umbrales = predecir_plagio(
                texto_original, texto_sospechoso
            )

        st.divider()
        st.subheader("Resultado del Análisis")

        # Mostrar similitud semántica (lo más importante)
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                "Similitud Semántica (BERT)",
                f"{similitud*100:.1f}%",
                help="Similitud coseno entre embeddings de RoBERTa Large"
            )
        with col2:
            st.metric(
                "Confianza",
                f"{confianza*100:.0f}%",
                help="Qué tan segura está la predicción"
            )

        etiquetas = {
            "non": ("No hay plagio", "success"),
            "light": ("Plagio ligero (paráfrasis leve)", "warning"),
            "heavy": ("Plagio fuerte (paráfrasis intensa)", "warning"),
            "cut": ("Copia directa", "error")
        }
        
        st.markdown("### Veredicto")
        label_text, color = etiquetas.get(prediccion, (prediccion, "info"))
        
        if color == "success":
            st.success(f"{label_text}")
        elif color == "warning":
            st.warning(f"{label_text}")
        elif color == "error":
            st.error(f"{label_text}")
        
        # Mostrar umbrales usados
        with st.expander("Detalles técnicos"):
            st.write("Método: Similitud coseno de embeddings RoBERTa Large (1024 dimensiones)")
            st.write(f"Similitud medida: {similitud:.4f}")
            
            st.write("\nUmbrales calibrados:")
            st.write(f"- Cut (copia): ≥ {umbrales['cut']:.4f}")
            st.write(f"- Light (paráfrasis leve): ≥ {umbrales['light']:.4f}")
            st.write(f"- Heavy (paráfrasis fuerte): ≥ {umbrales['heavy']:.4f}")
            st.write(f"- Non (sin plagio): < {umbrales['heavy']:.4f}")
            
            # Visualización de la similitud con los umbrales
            st.write("\nPosición de tu texto:")
            progress_val = (similitud - 0.5) / 0.5  # Normalizar entre 0.5 y 1.0
            progress_val = max(0, min(1, progress_val))  # Asegurarse de que esté entre 0 y 1
            st.progress(progress_val * 100, text=f"Similitud: {similitud:.4f}")
        
        # Advertencia si confianza es baja
        if confianza < 0.6:
            st.info("La confianza es moderada. La similitud está en una zona ambigua entre categorías.")
