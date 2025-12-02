import streamlit as st
import joblib
import os
import numpy as np

# ================== CONFIGURACIÓN BÁSICA ==================
st.set_page_config(
    page_title="Detector de Plagio",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ----- Estilos globales -----
st.markdown(
    """
    <style>
        .main {
            background-color: #ffffff;
        }
        /* Centrar y limitar ancho del contenido */
        .block-container {
            max-width: 1100px;
            padding-top: 2.5rem;
            padding-bottom: 2.5rem;
        }
        textarea {
            border-radius: 18px !important;
            background-color: #f5f5f7 !important;
            border: 1px solid #e3e3ea !important;
        }
        /* Botón principal tipo material */
        .stButton>button {
            background-color: #2962ff;
            color: white;
            border-radius: 999px;
            border: none;
            padding: 0.65rem 2.4rem;
            font-weight: 600;
            font-size: 0.95rem;
            box-shadow: 0 8px 18px rgba(41,98,255,0.35);
        }
        .stButton>button:hover {
            background-color: #1849c6;
            box-shadow: 0 10px 22px rgba(41,98,255,0.4);
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Orden fijo de features 
FEATURE_KEYS = [
    "sim_coseno",
    "jaccard_words",
    "jaccard_bigrams",
    "overlap_coef",
    "len_ratio",
    "jaccard_char_bigrams",
    "vocab_ratio",
]

# ================== CARGA DEL MODELO ==================
@st.cache_resource
def load_model():
    model_path = "modelo_plagio_rf.pkl"

    if not os.path.exists(model_path):
        st.error("No se encontró el modelo entrenado (modelo_plagio_rf.pkl).")
        st.info(
            "Primero entrena el modelo en tu entorno local "
            "y vuelve a ejecutar esta aplicación."
        )
        st.stop()

    model_package = joblib.load(model_path)
    return model_package


with st.spinner("Cargando modelo..."):
    model_package = load_model()

encoder = model_package["encoder"]        # RoBERTa
classifier = model_package["classifier"]  # VotingClassifier (LogReg + RF)
label_encoder = model_package["label_encoder"]  # para decodificar etiquetas


# ================== FUNCIONES DE FEATURES ==================
def compute_similarity(emb1, emb2):
    """Similitud coseno entre dos embeddings."""
    return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))


def extract_features(texto1, texto2, emb1, emb2):
    """
    Extrae el mismo conjunto de features que se usó en el entrenamiento (7 features).
    """
    texto1 = str(texto1)
    texto2 = str(texto2)

    words1 = texto1.lower().split()
    words2 = texto2.lower().split()

    set1 = set(words1)
    set2 = set(words2)

    sim_coseno = compute_similarity(emb1, emb2)

    jaccard_words = len(set1 & set2) / len(set1 | set2) if (set1 | set2) else 0.0

    bigrams1 = set(zip(words1[:-1], words1[1:])) if len(words1) > 1 else set()
    bigrams2 = set(zip(words2[:-1], words2[1:])) if len(words2) > 1 else set()
    jaccard_bigrams = (
        len(bigrams1 & bigrams2) / len(bigrams1 | bigrams2)
        if (bigrams1 | bigrams2) else 0.0
    )

    overlap_coef = (
        len(set1 & set2) / min(len(set1), len(set2))
        if min(len(set1), len(set2)) else 0.0
    )

    len_ratio = (
        min(len(texto1), len(texto2)) / max(len(texto1), len(texto2))
        if max(len(texto1), len(texto2)) else 0.0
    )

    char_bigrams1 = set(texto1[i : i + 2] for i in range(len(texto1) - 1))
    char_bigrams2 = set(texto2[i : i + 2] for i in range(len(texto2) - 1))
    jaccard_char_bigrams = (
        len(char_bigrams1 & char_bigrams2)
        / len(char_bigrams1 | char_bigrams2)
        if (char_bigrams1 | char_bigrams2) else 0.0
    )

    vocab_ratio = (
        min(len(set1), len(set2)) / max(len(set1), len(set2))
        if max(len(set1), len(set2)) else 0.0
    )

    return {
        "sim_coseno": sim_coseno,
        "jaccard_words": jaccard_words,
        "jaccard_bigrams": jaccard_bigrams,
        "overlap_coef": overlap_coef,
        "len_ratio": len_ratio,
        "jaccard_char_bigrams": jaccard_char_bigrams,
        "vocab_ratio": vocab_ratio,
    }


def predecir_plagio(texto_original, texto_sospechoso):
    """Devuelve etiqueta de plagio (string) y % de similitud semántica."""
    emb1 = encoder.encode([texto_original])[0]
    emb2 = encoder.encode([texto_sospechoso])[0]

    features = extract_features(texto_original, texto_sospechoso, emb1, emb2)
    sim_coseno = features["sim_coseno"]

    X = np.array([[features[key] for key in FEATURE_KEYS]], dtype=float)

    pred_encoded = classifier.predict(X)[0]
    etiqueta = label_encoder.inverse_transform([pred_encoded])[0]

    return etiqueta, sim_coseno * 100.0


# ================== DESCRIPCIÓN DE NIVELES ==================
def describir_nivel(etiqueta, sim_porcentaje):
    if etiqueta == "non":
        return (
            "Los textos no presentan coincidencias relevantes. "
            "La similitud detectada puede deberse únicamente al tema general."
        )
    if etiqueta == "light":
        return (
            "Los textos comparten ideas o estructuras parciales, con cambios evidentes. "
            "Podría tratarse de paráfrasis o inspiración fuerte."
        )
    return (
        "Los textos son altamente similares en contenido y redacción. "
        "Existe una fuerte probabilidad de copia directa o plagio sustancial."
    )


# ================== INTERFAZ ==================

st.markdown(
    """
    <h1 style="font-size: 2.6rem; margin-bottom: 0.3rem;">
        Detector de plagio
    </h1>
    <p style="font-size: 1.05rem; color: #4b4b55; max-width: 720px;">
        Compara dos fragmentos de texto y estima el nivel de similitud semántica
        utilizando <strong>embeddings RoBERTa</strong> y un ensamble de modelos
        <strong>Logistic Regression + Random Forest</strong>.
    </p>
    """,
    unsafe_allow_html=True,
)

st.markdown("<hr style='margin: 1.8rem 0 1.6rem 0;'/>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("<h3 style='margin-bottom: 0.5rem;'>Texto original</h3>", unsafe_allow_html=True)
    texto_original = st.text_area(
        "",
        height=260,
        placeholder="Pega aquí el texto base…",
        label_visibility="collapsed",
    )

with col2:
    st.markdown("<h3 style='margin-bottom: 0.5rem;'>Texto sospechoso</h3>", unsafe_allow_html=True)
    texto_sospechoso = st.text_area(
        "",
        height=260,
        placeholder="Pega aquí el texto que quieres analizar…",
        label_visibility="collapsed",
    )

st.write("")
analizar = st.button("Analizar plagio")

if analizar:
    if not texto_original.strip() or not texto_sospechoso.strip():
        st.error("Por favor ingresa ambos textos.")
    else:
        with st.spinner("Analizando…"):
            etiqueta, sim_porcentaje = predecir_plagio(texto_original, texto_sospechoso)

        if etiqueta == "non":
            titulo_nivel = "No hay indicios claros de plagio"
        elif etiqueta == "light":
            titulo_nivel = "Plagio leve"
        else:
            titulo_nivel = "Plagio alto"

        descripcion = describir_nivel(etiqueta, sim_porcentaje)

        st.markdown("<hr style='margin: 2rem 0 1.4rem 0;'/>", unsafe_allow_html=True)

        st.markdown(
            f"""
            <div style="margin-bottom: 0.4rem; font-size: 1.3rem; font-weight: 700;">
                {titulo_nivel}
            </div>
            <div style="font-size: 1.05rem; margin-bottom: 0.6rem;">
                Similitud semántica estimada:
                <strong>{sim_porcentaje:.2f}%</strong>
            </div>
            <div style="font-size: 0.98rem; color: #4b4b55; max-width: 800px;">
                {descripcion}
            </div>
            """,
            unsafe_allow_html=True,
        )