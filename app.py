import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import joblib

# ============================================================
# Cargar modelos
# ============================================================
@st.cache_resource
def load_models():
    encoder = SentenceTransformer("sentence-transformers/all-roberta-large-v1")
    clf = joblib.load("modelo_plagio.pkl")
    le = joblib.load("label_encoder.pkl")
    return encoder, clf, le

encoder, clf, le = load_models()

# ============================================================
# Funciones auxiliares 
# ============================================================
def limpiar_texto(txt: str) -> str:
    return " ".join(str(txt).lower().split())

def obtener_palabras(txt: str):
    txt = limpiar_texto(txt)
    return txt.split()

def jaccard_palabras(texto_a: str, texto_b: str) -> float:
    palabras_a = set(obtener_palabras(texto_a))
    palabras_b = set(obtener_palabras(texto_b))
    if not palabras_a or not palabras_b:
        return 0.0
    interseccion = len(palabras_a & palabras_b)
    union = len(palabras_a | palabras_b)
    return interseccion / union

# ============================================================
# Predicción de plagio usando 5 FEATURES 
# ============================================================
def predecir_plagio(texto_a, texto_b):
    # 1. Embeddings
    emb_a = encoder.encode([texto_a])[0]
    emb_b = encoder.encode([texto_b])[0]

    # 2. Similitud coseno
    sim = cosine_similarity([emb_a], [emb_b])[0][0]
    porcentaje = sim * 100

    # 3. Ratio de longitud
    len_a = len(texto_a)
    len_b = len(texto_b) if len(texto_b) > 0 else 1
    len_ratio = len_a / len_b

    # 4. Diferencia de caracteres
    diff_len_chars = abs(len_a - len_b)

    # 5. Diferencia de palabras
    palabras_a = obtener_palabras(texto_a)
    palabras_b = obtener_palabras(texto_b)
    diff_len_words = abs(len(palabras_a) - len(palabras_b))

    # 6. Jaccard de palabras
    jacc = jaccard_palabras(texto_a, texto_b)

    # Vector final de 5 features (misma posición que en train)
    X = [[sim, len_ratio, diff_len_chars, diff_len_words, jacc]]

    # Predicción del modelo entrenado
    clase_idx = clf.predict(X)[0]
    etiqueta = le.inverse_transform([clase_idx])[0]

    return porcentaje, etiqueta

# ============================================================
# Interfaz Streamlit
# ============================================================
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