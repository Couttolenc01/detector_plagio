import streamlit as st
import joblib
import os
import numpy as np
import re

st.set_page_config(
    page_title="Detector de Plagio",
    page_icon="🔍",
    layout="centered"
)

# Cargar modelo
@st.cache_resource
def load_model():
    if not os.path.exists("modelo_plagio_rf.pkl"):
        st.error("Modelo no encontrado. Ejecuta primero: python train_clasificador_rf.py")
        st.stop()
    return joblib.load("modelo_plagio_rf.pkl")

model_package = load_model()
encoder = model_package['encoder']
classifier = model_package['classifier']
model_info = model_package.get('model_info', {})

# Funciones auxiliares
def compute_similarity(emb1, emb2):
    return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

def extract_features(texto1, texto2, emb1, emb2):
    sim_coseno = compute_similarity(emb1, emb2)
    
    len1, len2 = len(texto1), len(texto2)
    len_ratio = min(len1, len2) / max(len1, len2) if max(len1, len2) > 0 else 0
    len_diff_rel = abs(len1 - len2) / max(len1, len2) if max(len1, len2) > 0 else 0
    
    words1 = texto1.lower().split()
    words2 = texto2.lower().split()
    set1 = set(words1)
    set2 = set(words2)
    
    common_words = set1.intersection(set2)
    union_words = set1.union(set2)
    
    jaccard_words = len(common_words) / len(union_words) if union_words else 0
    overlap_coef = len(common_words) / min(len(set1), len(set2)) if min(len(set1), len(set2)) > 0 else 0
    
    bigrams1 = set(zip(words1[:-1], words1[1:])) if len(words1) > 1 else set()
    bigrams2 = set(zip(words2[:-1], words2[1:])) if len(words2) > 1 else set()
    jaccard_bigrams = len(bigrams1.intersection(bigrams2)) / len(bigrams1.union(bigrams2)) if bigrams1.union(bigrams2) else 0
    
    trigrams1 = set(zip(words1[:-2], words1[1:-1], words1[2:])) if len(words1) > 2 else set()
    trigrams2 = set(zip(words2[:-2], words2[1:-1], words2[2:])) if len(words2) > 2 else set()
    jaccard_trigrams = len(trigrams1.intersection(trigrams2)) / len(trigrams1.union(trigrams2)) if trigrams1.union(trigrams2) else 0
    
    char_bigrams1 = set([texto1[i:i+2] for i in range(len(texto1)-1)])
    char_bigrams2 = set([texto2[i:i+2] for i in range(len(texto2)-1)])
    jaccard_char_bigrams = len(char_bigrams1.intersection(char_bigrams2)) / len(char_bigrams1.union(char_bigrams2)) if char_bigrams1.union(char_bigrams2) else 0
    
    num_sentences1 = len(re.split(r'[.!?]+', texto1))
    num_sentences2 = len(re.split(r'[.!?]+', texto2))
    sentence_ratio = min(num_sentences1, num_sentences2) / max(num_sentences1, num_sentences2) if max(num_sentences1, num_sentences2) > 0 else 0
    
    vocab_size1 = len(set1)
    vocab_size2 = len(set2)
    vocab_ratio = min(vocab_size1, vocab_size2) / max(vocab_size1, vocab_size2) if max(vocab_size1, vocab_size2) > 0 else 0
    
    ttr1 = vocab_size1 / len(words1) if words1 else 0
    ttr2 = vocab_size2 / len(words2) if words2 else 0
    ttr_diff = abs(ttr1 - ttr2)
    
    return {
        'sim_coseno': sim_coseno,
        'len_ratio': len_ratio,
        'len_diff_rel': len_diff_rel,
        'jaccard_words': jaccard_words,
        'overlap_coef': overlap_coef,
        'jaccard_bigrams': jaccard_bigrams,
        'jaccard_trigrams': jaccard_trigrams,
        'jaccard_char_bigrams': jaccard_char_bigrams,
        'sentence_ratio': sentence_ratio,
        'vocab_ratio': vocab_ratio,
        'ttr_diff': ttr_diff,
    }

def predecir_plagio(original, sospechoso):
    emb1 = encoder.encode([original])[0]
    emb2 = encoder.encode([sospechoso])[0]
    
    features = extract_features(original, sospechoso, emb1, emb2)
    X = np.array([list(features.values())])
    
    prediction = classifier.predict(X)[0]
    probas = classifier.predict_proba(X)[0]
    
    class_names = classifier.classes_
    pred_idx = np.where(class_names == prediction)[0][0]
    confidence = probas[pred_idx]
    
    return prediction, confidence, features, probas, class_names

# Interfaz
st.title("Detector de Plagio")
st.caption("BERT + Random Forest | non · light · cut")

# Info del modelo (colapsable)
with st.expander("Información del modelo"):
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Modelo:** RoBERTa Large")
        st.write(f"**Dimensiones:** {model_info.get('encoder_dim', 1024)}")
    with col2:
        st.write(f"**Accuracy:** {model_info.get('accuracy_test', 0):.2%}")
        st.write(f"**F1-Score:** {model_info.get('f1_test', 0):.2%}")

st.divider()

# Entrada de textos
col1, col2 = st.columns(2)

with col1:
    texto_original = st.text_area(
        "Texto Original",
        height=200,
        placeholder="Texto de referencia..."
    )

with col2:
    texto_sospechoso = st.text_area(
        "Texto a Verificar",
        height=200,
        placeholder="Texto para analizar..."
    )

# Botón análisis
if st.button("Analizar", type="primary", use_container_width=True):
    if not texto_original.strip() or not texto_sospechoso.strip():
        st.error("Ingresa ambos textos para continuar")
    else:
        with st.spinner("Analizando..."):
            prediccion, confianza, features, probas, class_names = predecir_plagio(
                texto_original, texto_sospechoso
            )

        st.divider()
        
        # Resultado principal
        col1, col2 = st.columns([3, 1])
        
        with col1:
            if prediccion == "non":
                st.success("NO HAY PLAGIO")
                st.caption("Los textos son suficientemente diferentes")
            elif prediccion == "light":
                st.warning("PLAGIO LIGERO")
                st.caption("Paráfrasis superficial detectada")
            else:
                st.error("COPIA DIRECTA")
                st.caption("Alta similitud detectada")
        
        with col2:
            st.metric("Confianza", f"{confianza*100:.1f}%")
        
        # Probabilidades
        st.subheader("Probabilidades")
        cols = st.columns(len(class_names))
        for i, clase in enumerate(class_names):
            with cols[i]:
                st.metric(clase.upper(), f"{probas[i]*100:.1f}%")
                st.progress(probas[i])
        
        # Métricas detalladas
        st.subheader("Análisis Detallado")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Similitud BERT", f"{features['sim_coseno']:.3f}")
            st.metric("Jaccard Palabras", f"{features['jaccard_words']:.3f}")
            st.metric("Jaccard Bigramas", f"{features['jaccard_bigrams']:.3f}")
            st.metric("Jaccard Trigramas", f"{features['jaccard_trigrams']:.3f}")
        
        with col2:
            st.metric("Overlap Coef.", f"{features['overlap_coef']:.3f}")
            st.metric("Jaccard Char", f"{features['jaccard_char_bigrams']:.3f}")
            st.metric("Ratio Longitud", f"{features['len_ratio']:.3f}")
            st.metric("Diff Longitud", f"{features['len_diff_rel']:.3f}")
        
        with col3:
            st.metric("Ratio Oraciones", f"{features['sentence_ratio']:.3f}")
            st.metric("Ratio Vocabulario", f"{features['vocab_ratio']:.3f}")
            st.metric("Diff TTR", f"{features['ttr_diff']:.3f}")
        
        # Interpretación
        with st.expander("Interpretación"):
            if prediccion == "non":
                st.info("Los textos presentan diferencias significativas. No se detectó plagio.")
            elif prediccion == "light":
                st.warning("Los textos comparten estructura e ideas con reformulación superficial. Considerar citar fuentes y reescribir con mayor originalidad.")
            else:
                st.error("Los textos son prácticamente idénticos. Esto constituye plagio. Debe reescribirse completamente con palabras propias y citar las fuentes.")

st.divider()
st.caption("Detector de Plagio | BERT + Machine Learning")