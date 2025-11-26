"""
App de Detección de Plagio
Solo inferencia - NO entrena
"""
import streamlit as st
import joblib
import os
import numpy as np

st.set_page_config(
    page_title="Detector de Plagio",
    layout="centered"
)

# ========== CARGAR MODELO (NO ENTRENA) ==========
@st.cache_resource
def load_model():
    model_path = "modelo_plagio_rf.pkl"
    
    if not os.path.exists(model_path):
        st.error(" Modelo no encontrado")
        st.info("""
        **Pasos para entrenar el modelo:**
        
        1. Fusiona los datasets:
           ```bash
           python merge_datasets_clean.py
           ```
        
        2. Entrena el modelo:
           ```bash
           python train_clasificador_optimizado.py
           ```
        
        3. Recarga esta página
        """)
        st.stop()
    
    model_package = joblib.load(model_path)
    return model_package


with st.spinner("Cargando modelo..."):
    model_package = load_model()

encoder = model_package['encoder']
classifier = model_package['classifier']


# ========== FUNCIONES DE PREDICCIÓN ==========
def compute_similarity(emb1, emb2):
    return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

def extract_features(texto1, texto2, emb1, emb2):
    words1 = texto1.lower().split()
    words2 = texto2.lower().split()

    set1 = set(words1)
    set2 = set(words2)

    sim_coseno = compute_similarity(emb1, emb2)

    # Métricas mínimas que necesita el modelo (NO se muestran)
    jaccard_words = len(set1 & set2) / len(set1 | set2) if set1 | set2 else 0
    bigrams1 = set(zip(words1[:-1], words1[1:])) if len(words1) > 1 else set()
    bigrams2 = set(zip(words2[:-1], words2[1:])) if len(words2) > 1 else set()
    jaccard_bigrams = len(bigrams1 & bigrams2) / len(bigrams1 | bigrams2) if bigrams1 | bigrams2 else 0
    overlap_coef = len(set1 & set2) / min(len(set1), len(set2)) if min(len(set1), len(set2)) else 0
    len_ratio = min(len(texto1), len(texto2)) / max(len(texto1), len(texto2)) if max(len(texto1), len(texto2)) else 0
    char_bigrams1 = set(texto1[i:i+2] for i in range(len(texto1)-1))
    char_bigrams2 = set(texto2[i:i+2] for i in range(len(texto2)-1))
    jaccard_char_bigrams = len(char_bigrams1 & char_bigrams2) / len(char_bigrams1 | char_bigrams2) if char_bigrams1 | char_bigrams2 else 0
    vocab_ratio = min(len(set1), len(set2)) / max(len(set1), len(set2)) if max(len(set1), len(set2)) else 0

    return {
        'sim_coseno': sim_coseno,
        'jaccard_words': jaccard_words,
        'jaccard_bigrams': jaccard_bigrams,
        'overlap_coef': overlap_coef,
        'len_ratio': len_ratio,
        'jaccard_char_bigrams': jaccard_char_bigrams,
        'vocab_ratio': vocab_ratio
    }

def predecir_plagio(original, sospechoso):
    emb1 = encoder.encode([original])[0]
    emb2 = encoder.encode([sospechoso])[0]

    features = extract_features(original, sospechoso, emb1, emb2)
    X = np.array([list(features.values())])

    prediction = classifier.predict(X)[0]
    return prediction


# ========== UI SIMPLE ==========
st.title(" Detector de Plagio")
st.caption("BERT + Random Forest")

st.divider()

col1, col2 = st.columns(2)

with col1:
    texto_original = st.text_area(
        "Texto Original",
        height=200,
        placeholder="Ingresa el texto base..."
    )

with col2:
    texto_sospechoso = st.text_area(
        "Texto a Verificar",
        height=200,
        placeholder="Ingresa el texto sospechoso..."
    )

if st.button(" Analizar", type="primary", use_container_width=True):
    if not texto_original.strip() or not texto_sospechoso.strip():
        st.error("Debes ingresar ambos textos")
    else:
        with st.spinner("Analizando..."):
            prediccion = predecir_plagio(texto_original, texto_sospechoso)

        st.divider()

        if prediccion == "non":
            st.success("✅ NO HAY PLAGIO")
        elif prediccion == "light":
            st.warning("🟡 PLAGIO LIGERO")
        else:
            st.error("🔴 PLAGIO ALTO")

st.divider()
st.caption("Detector de Plagio — Team 2")
