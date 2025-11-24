import streamlit as st
import joblib
import os
import numpy as np
import re

# --------------------------
# Configuración de página
# --------------------------
st.set_page_config(
    page_title="Detector de Plagio BERT",
    page_icon="🤖",
    layout="wide"
)

# --------------------------
# Cargar modelo
# --------------------------
@st.cache_resource
def load_model():
    if not os.path.exists("modelo_plagio_rf.pkl"):
        st.error("❌ Modelo no encontrado: modelo_plagio_rf.pkl")
        st.info("💡 Ejecuta: `python train_clasificador_rf.py` para generar el modelo.")
        st.stop()
    
    model_package = joblib.load("modelo_plagio_rf.pkl")
    return model_package

try:
    model_package = load_model()
    encoder = model_package['encoder']
    classifier = model_package['classifier']
    feature_names = model_package['feature_names']
    classes = model_package.get('classes', ['non', 'light', 'cut'])
    model_info = model_package.get('model_info', {})
    modelo_cargado = True
except Exception as e:
    st.error(f"❌ Error al cargar el modelo: {e}")
    modelo_cargado = False

# --------------------------
# Funciones auxiliares
# --------------------------

def compute_similarity(emb1, emb2):
    """Calcula similitud coseno entre embeddings BERT"""
    return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

def extract_features(texto1, texto2, emb1, emb2):
    """Extrae todas las características del par de textos"""
    
    # 1. Similitud semántica BERT
    sim_coseno = compute_similarity(emb1, emb2)
    
    # 2. Longitud
    len1, len2 = len(texto1), len(texto2)
    len_ratio = min(len1, len2) / max(len1, len2) if max(len1, len2) > 0 else 0
    len_diff_rel = abs(len1 - len2) / max(len1, len2) if max(len1, len2) > 0 else 0
    
    # 3. Léxicas
    words1 = texto1.lower().split()
    words2 = texto2.lower().split()
    set1 = set(words1)
    set2 = set(words2)
    
    common_words = set1.intersection(set2)
    union_words = set1.union(set2)
    
    jaccard_words = len(common_words) / len(union_words) if union_words else 0
    overlap_coef = len(common_words) / min(len(set1), len(set2)) if min(len(set1), len(set2)) > 0 else 0
    
    # 4. N-gramas
    bigrams1 = set(zip(words1[:-1], words1[1:])) if len(words1) > 1 else set()
    bigrams2 = set(zip(words2[:-1], words2[1:])) if len(words2) > 1 else set()
    jaccard_bigrams = len(bigrams1.intersection(bigrams2)) / len(bigrams1.union(bigrams2)) if bigrams1.union(bigrams2) else 0
    
    trigrams1 = set(zip(words1[:-2], words1[1:-1], words1[2:])) if len(words1) > 2 else set()
    trigrams2 = set(zip(words2[:-2], words2[1:-1], words2[2:])) if len(words2) > 2 else set()
    jaccard_trigrams = len(trigrams1.intersection(trigrams2)) / len(trigrams1.union(trigrams2)) if trigrams1.union(trigrams2) else 0
    
    # 5. Caracteres
    char_bigrams1 = set([texto1[i:i+2] for i in range(len(texto1)-1)])
    char_bigrams2 = set([texto2[i:i+2] for i in range(len(texto2)-1)])
    jaccard_char_bigrams = len(char_bigrams1.intersection(char_bigrams2)) / len(char_bigrams1.union(char_bigrams2)) if char_bigrams1.union(char_bigrams2) else 0
    
    # 6. Estructurales
    num_sentences1 = len(re.split(r'[.!?]+', texto1))
    num_sentences2 = len(re.split(r'[.!?]+', texto2))
    sentence_ratio = min(num_sentences1, num_sentences2) / max(num_sentences1, num_sentences2) if max(num_sentences1, num_sentences2) > 0 else 0
    
    vocab_size1 = len(set1)
    vocab_size2 = len(set2)
    vocab_ratio = min(vocab_size1, vocab_size2) / max(vocab_size1, vocab_size2) if max(vocab_size1, vocab_size2) > 0 else 0
    
    ttr1 = vocab_size1 / len(words1) if words1 else 0
    ttr2 = vocab_size2 / len(words2) if words2 else 0
    ttr_diff = abs(ttr1 - ttr2)
    
    features = {
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
    
    return features

def predecir_plagio(original, sospechoso):
    """Predice plagio usando BERT + Random Forest"""
    
    # Generar embeddings con BERT
    emb1 = encoder.encode([original])[0]
    emb2 = encoder.encode([sospechoso])[0]
    
    # Extraer características
    features = extract_features(original, sospechoso, emb1, emb2)
    
    # Preparar vector
    X = np.array([list(features.values())])
    
    # Predecir
    prediction = classifier.predict(X)[0]
    probas = classifier.predict_proba(X)[0]
    
    # Obtener confianza
    class_names = classifier.classes_
    pred_idx = np.where(class_names == prediction)[0][0]
    confidence = probas[pred_idx]
    
    return prediction, confidence, features, probas, class_names

# --------------------------
# Interfaz principal
# --------------------------

st.title("🤖 Detector de Plagio – BERT + Machine Learning")

st.markdown("""
Sistema profesional de detección de plagio usando **BERT (RoBERTa Large)** combinado con **Random Forest**.  
Analiza similitud semántica profunda y características léxicas/estructurales.
""")

# Categorías
col1, col2, col3 = st.columns(3)
with col1:
    st.success("**🟢 NON**  \nNo hay plagio  \nTextos diferentes")
with col2:
    st.warning("**🟡 LIGHT**  \nParáfrasis ligera  \nReformulación superficial")
with col3:
    st.error("**🔴 CUT**  \nCopia directa  \nCambios mínimos")

st.divider()

if not modelo_cargado:
    st.warning("⚠️ Modelo no cargado. Entrena primero con: `python train_clasificador_rf.py`")
    st.stop()

# Info del modelo
with st.expander("ℹ️ Información del modelo BERT"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Arquitectura:**")
        st.write(f"- Encoder: RoBERTa Large")
        st.write(f"- Dimensiones: {model_info.get('encoder_dim', 1024)}")
        st.write(f"- Clasificador: Random Forest")
        st.write(f"- Árboles: {model_info.get('n_estimators', 200)}")
    
    with col2:
        st.write("**Performance:**")
        if 'accuracy_test' in model_info:
            st.write(f"- Accuracy (Test): {model_info['accuracy_test']:.1%}")
        if 'accuracy_full' in model_info:
            st.write(f"- Accuracy (Full): {model_info['accuracy_full']:.1%}")
        st.write(f"- Features: {len(feature_names)}")
        st.write(f"- Clases: {', '.join(classes)}")

st.divider()

# Entrada de textos
st.subheader("📝 Ingresa los textos a comparar")

col1, col2 = st.columns(2)

with col1:
    texto_original = st.text_area(
        "📄 Texto Original",
        height=300,
        placeholder="Pega aquí el texto original...",
        help="El texto de referencia contra el cual se comparará"
    )

with col2:
    texto_sospechoso = st.text_area(
        "🔍 Texto a Verificar",
        height=300,
        placeholder="Pega aquí el texto a verificar...",
        help="El texto que quieres analizar para detectar plagio"
    )

# Botón de análisis
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    analizar_btn = st.button("🔍 Analizar Plagio", type="primary", use_container_width=True)

if analizar_btn:
    if not texto_original.strip() or not texto_sospechoso.strip():
        st.error("❌ Por favor ingresa ambos textos para realizar el análisis.")
    else:
        with st.spinner("🧮 Analizando con BERT y Random Forest..."):
            prediccion, confianza, features, probas, class_names = predecir_plagio(
                texto_original, texto_sospechoso
            )

        st.divider()
        st.subheader("📊 Resultado del Análisis")

        # Resultado principal
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            etiquetas = {
                "non": ("✅ NO HAY PLAGIO", "success", "Los textos son suficientemente diferentes."),
                "light": ("⚠️ PLAGIO LIGERO", "warning", "Paráfrasis superficial detectada."),
                "cut": ("🚨 COPIA DIRECTA", "error", "Alta similitud - posible plagio.")
            }
            
            label_text, color, descripcion = etiquetas.get(prediccion, (prediccion, "info", ""))
            
            if color == "success":
                st.success(f"### {label_text}")
            elif color == "warning":
                st.warning(f"### {label_text}")
            elif color == "error":
                st.error(f"### {label_text}")
            
            st.caption(descripcion)
        
        with col2:
            st.metric(
                "Confianza",
                f"{confianza*100:.1f}%",
                help="Probabilidad de la predicción"
            )
        
        with col3:
            # Emoji según resultado
            emoji_map = {"non": "✅", "light": "⚠️", "cut": "🚨"}
            st.markdown(f"<div style='text-align: center; font-size: 80px;'>{emoji_map.get(prediccion, '❓')}</div>", unsafe_allow_html=True)
        
        # Distribución de probabilidades
        st.markdown("### 📊 Distribución de Probabilidades")
        
        prob_cols = st.columns(len(class_names))
        for i, clase in enumerate(class_names):
            with prob_cols[i]:
                emoji = {"non": "🟢", "light": "🟡", "cut": "🔴"}
                prob = probas[i] * 100
                
                # Color según probabilidad
                if prob >= 50:
                    color = "success" if clase == "non" else ("error" if clase == "cut" else "warning")
                else:
                    color = "secondary"
                
                st.metric(
                    f"{emoji.get(clase, '•')} {clase.upper()}",
                    f"{prob:.1f}%"
                )
                st.progress(probas[i])
        
        # Características principales
        st.markdown("### 🔍 Análisis Detallado")
        
        tab1, tab2, tab3 = st.tabs(["🧠 Similitud Semántica", "📝 Características Léxicas", "📏 Características Estructurales"])
        
        with tab1:
            st.markdown("#### Similitud BERT (RoBERTa Large)")
            sim = features['sim_coseno']
            st.progress(sim, text=f"Similitud Coseno: {sim:.3f}")
            
            if sim >= 0.8:
                st.error("🚨 Similitud muy alta - Los textos son casi idénticos semánticamente")
            elif sim >= 0.6:
                st.warning("⚠️ Similitud considerable - Paráfrasis o contenido relacionado")
            elif sim >= 0.4:
                st.info("ℹ️ Similitud moderada - Algunos temas en común")
            else:
                st.success("✅ Similitud baja - Textos diferentes")
            
            st.caption("Mide la similitud semántica profunda usando embeddings de 1024 dimensiones")
        
        with tab2:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Jaccard (palabras)", f"{features['jaccard_words']:.3f}")
                st.caption("Palabras en común")
                
            with col2:
                st.metric("Jaccard (bigramas)", f"{features['jaccard_bigrams']:.3f}")
                st.caption("Pares de palabras")
                
            with col3:
                st.metric("Jaccard (trigramas)", f"{features['jaccard_trigrams']:.3f}")
                st.caption("Tríos de palabras")
            
            st.markdown("---")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Overlap Coefficient", f"{features['overlap_coef']:.3f}")
            with col2:
                st.metric("Jaccard (char bigrams)", f"{features['jaccard_char_bigrams']:.3f}")
        
        with tab3:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Ratio Longitud", f"{features['len_ratio']:.3f}")
                st.caption("Similitud en extensión")
                
            with col2:
                st.metric("Ratio Oraciones", f"{features['sentence_ratio']:.3f}")
                st.caption("Estructura similar")
                
            with col3:
                st.metric("Ratio Vocabulario", f"{features['vocab_ratio']:.3f}")
                st.caption("Riqueza léxica")
            
            st.markdown("---")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Dif. Longitud (rel)", f"{features['len_diff_rel']:.3f}")
            with col2:
                st.metric("Dif. TTR", f"{features['ttr_diff']:.3f}")
                st.caption("Type-Token Ratio")
        
        # Interpretación
        with st.expander("💡 Interpretación del Resultado"):
            if prediccion == "non":
                st.info("""
                **✅ No se detectó plagio.**
                
                Los textos presentan diferencias significativas tanto en contenido semántico como en características léxicas.
                La similitud BERT y las métricas de n-gramas están por debajo de los umbrales de plagio.
                
                **Conclusión:** Los documentos son suficientemente diferentes.
                """)
            elif prediccion == "light":
                st.warning("""
                **⚠️ Plagio ligero detectado.**
                
                Los textos comparten ideas y estructuras similares con reformulación superficial.
                Se detectó paráfrasis que mantiene el contenido original con cambios menores.
                
                **Recomendación:** Verificar fuentes y añadir citas apropiadas. Considerar reescribir con mayor originalidad.
                """)
            else:  # cut
                st.error("""
                **🚨 Copia directa detectada.**
                
                Los textos son prácticamente idénticos o tienen cambios muy mínimos.
                Alta similitud semántica y léxica indica copia sustancial del contenido.
                
                **ADVERTENCIA:** Esto constituye plagio académico. Se debe reescribir completamente con palabras propias y citar correctamente las fuentes.
                """)
        
        # Advertencias
        if confianza < 0.6:
            st.info("ℹ️ **Nota:** La confianza del modelo es moderada. El caso está en una zona fronteriza entre categorías. Se recomienda revisión manual.")

# Footer
st.divider()
st.caption("🤖 Powered by BERT (RoBERTa Large) + Random Forest | Desarrollado con Streamlit")

# Sidebar con info adicional
with st.sidebar:
    st.header("ℹ️ Acerca del Sistema")
    
    st.markdown("""
    ### 🤖 Tecnología BERT
    
    Este detector utiliza **RoBERTa Large**, una versión optimizada de BERT con:
    
    - 🧠 **355M parámetros**
    - 📊 **1024 dimensiones** por embedding
    - 🌐 **Comprensión semántica profunda**
    
    ### 🌲 Random Forest
    
    Clasificador ensemble que combina:
    - 200 árboles de decisión
    - 11 características diferentes
    - Balanceo de clases automático
    
    ### 📏 Características Analizadas
    
    1. **Semánticas:** Similitud BERT
    2. **Léxicas:** Jaccard (palabras, n-gramas, caracteres)
    3. **Estructurales:** Longitud, oraciones, vocabulario
    
    ### 🎯 Categorías
    
    - **NON:** Sin plagio
    - **LIGHT:** Paráfrasis ligera
    - **CUT:** Copia directa
    """)
    
    st.divider()
    
    st.markdown("""
    ### 💡 Consejos de Uso
    
    - Ingresa textos de al menos 100 caracteres
    - Resultados más precisos con textos más largos
    - La confianza >60% es confiable
    - Revisa manualmente casos fronterizos
    """)