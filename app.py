import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------------------------
#  Cargar modelo SOLO una vez (se cachea)
# -------------------------------------------------
@st.cache_resource
def load_model():
    # Puedes usar el mismo que ya usaste:
    return SentenceTransformer("sentence-transformers/all-roberta-large-v1")
    # Si quieres algo más ligero:
    # return SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

model = load_model()

# -------------------------------------------------
#  Función para calcular similitud
# -------------------------------------------------
def calcular_similitud(texto_a: str, texto_b: str) -> float:
    emb_a = model.encode([texto_a])[0]
    emb_b = model.encode([texto_b])[0]
    sim = cosine_similarity([emb_a], [emb_b])[0][0]
    return sim

def clasificar_plagio(sim: float) -> str:
    if sim >= 0.85:
        return "Plagio ALTO"
    elif sim >= 0.70:
        return "Plagio MODERADO"
    elif sim >= 0.55:
        return "Plagio LEVE"
    else:
        return "No se detecta plagio"

# -------------------------------------------------
#  Interfaz Streamlit
# -------------------------------------------------
st.set_page_config(page_title="Detector de Plagio", page_icon="🧠")

st.title("🧠 Detector de Plagio con Embeddings Semánticos")
st.write(
    "Ingresa un **texto original** y un **texto sospechoso**. "
    "El sistema calcula la similitud semántica usando un modelo tipo BERT/RoBERTa."
)

col1, col2 = st.columns(2)

with col1:
    texto_original = st.text_area(
        "Texto original",
        height=250,
        placeholder="Pega aquí el texto original...",
    )

with col2:
    texto_plagiado = st.text_area(
        "Texto sospechoso",
        height=250,
        placeholder="Pega aquí el texto que quieres analizar...",
    )

if st.button("🔍 Analizar plagio"):
    if not texto_original.strip() or not texto_plagiado.strip():
        st.error("Por favor ingresa ambos textos.")
    else:
        with st.spinner("Calculando similitud..."):
            sim = calcular_similitud(texto_original, texto_plagiado)
            porcentaje = sim * 100
            nivel = clasificar_plagio(sim)

        st.subheader("Resultados")
        st.metric("Similitud", f"{porcentaje:.2f}%")
        st.write(f"**Nivel estimado:** {nivel}")

        # Explicación rápida
        st.caption(
            "Umbrales usados: ≥ 0.85 plagio alto, 0.70–0.85 plagio moderado, "
            "0.55–0.70 plagio leve, < 0.55 no plagio."
        )