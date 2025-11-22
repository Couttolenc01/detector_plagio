import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import joblib

# --------------------------
# Cargar modelos
# --------------------------
@st.cache_resource
def load_models():
    encoder = SentenceTransformer("sentence-transformers/all-roberta-large-v1")
    clf = joblib.load("modelo_plagio.pkl")
    le = joblib.load("label_encoder.pkl")
    return encoder, clf, le

encoder, clf, le = load_models()

# --------------------------
# Predicción de plagio
# --------------------------
def predecir_plagio(texto_a, texto_b):
    emb_a = encoder.encode([texto_a])[0]
    emb_b = encoder.encode([texto_b])[0]

    sim = cosine_similarity([emb_a], [emb_b])[0][0]
    porcentaje = sim * 100

    len_ratio = len(texto_a) / max(len(texto_b), 1)

    X = [[sim, len_ratio]]
    clase_idx = clf.predict(X)[0]
    etiqueta = le.inverse_transform([clase_idx])[0]

    return porcentaje, etiqueta

# --------------------------
# Interfaz
# --------------------------
st.set_page_config(page_title="Detector de Plagio", page_icon="🧠")
st.title("🧠 Detector de Plagio")

col1, col2 = st.columns(2)

with col1:
    texto_original = st.text_area("Texto original", height=250)

with col2:
    texto_plagiado = st.text_area("Texto sospechoso", height=250)

if st.button("🔍 Analizar plagio"):
    if not texto_original.strip() or not texto_plagiado.strip():
        st.error("Ingresa ambos textos.")
    else:
        with st.spinner("Analizando..."):
            porcentaje, nivel = predecir_plagio(texto_original, texto_plagiado)

        st.subheader("Resultados")
        st.metric("Similitud semántica", f"{porcentaje:.2f}%")
        st.write(f"**Nivel estimado:** {nivel}")