import streamlit as st
import joblib

# --------------------------
# Cargar modelos
# --------------------------
@st.cache_resource
def load_models():
    encoder = joblib.load("encoder_plagio.pkl")
    clf = joblib.load("modelo_plagio.pkl")
    return encoder, clf

encoder, clf = load_models()

# --------------------------
# Predicción
# --------------------------
def predecir_plagio(original, sospechoso):
    entrada = original + " " + sospechoso
    emb = encoder.encode([entrada])
    pred = clf.predict(emb)[0]
    return pred

# --------------------------
# Interfaz
# --------------------------
st.set_page_config(page_title="Detector de Plagio", page_icon="🧠")
st.title("🧠 Detector de Plagio – Modelo Clough Adaptado")

col1, col2 = st.columns(2)

with col1:
    texto_original = st.text_area("Texto original", height=250)

with col2:
    texto_sospechoso = st.text_area("Texto sospechoso", height=250)

if st.button("🔍 Analizar plagio"):
    if not texto_original.strip() or not texto_sospechoso.strip():
        st.error("Ingresa ambos textos.")
    else:
        with st.spinner("Analizando..."):
            nivel = predecir_plagio(texto_original, texto_sospechoso)

        st.subheader("Resultado")

        etiquetas = {
            "non": "🟢 No hay plagio",
            "light": "🟡 Plagio ligero",
            "heavy": "🟠 Plagio fuerte",
            "cut": "🔴 Copia directa / recorte"
        }

        st.write(f"**Nivel estimado:** {etiquetas.get(nivel, nivel)}")
