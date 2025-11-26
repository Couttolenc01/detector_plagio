"""
Detector de Plagio OPTIMIZADO: 3 Versiones
- MINIMAL: Solo 3 features críticas (más rápido)
- BALANCED: 5 features importantes (recomendado)
- FULL: Todas las features (máxima precisión)
"""
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import joblib
import re

DATASET = "dataset_combined_clean.csv"
MODEL_NAME = "sentence-transformers/all-roberta-large-v1"
#MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
#MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"



# Selecciona la versión aquí:
VERSION = "FULL"  # Opciones: "MINIMAL", "BALANCED", "FULL"

def compute_similarity(emb1, emb2):
    return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

def extract_features_minimal(texto1, texto2, emb1, emb2):
    """Solo las 3 features más importantes - Rápido y efectivo"""
    sim_coseno = compute_similarity(emb1, emb2)
    
    words1 = texto1.lower().split()
    words2 = texto2.lower().split()
    set1 = set(words1)
    set2 = set(words2)
    
    common_words = set1.intersection(set2)
    union_words = set1.union(set2)
    jaccard_words = len(common_words) / len(union_words) if union_words else 0
    
    bigrams1 = set(zip(words1[:-1], words1[1:])) if len(words1) > 1 else set()
    bigrams2 = set(zip(words2[:-1], words2[1:])) if len(words2) > 1 else set()
    jaccard_bigrams = len(bigrams1.intersection(bigrams2)) / len(bigrams1.union(bigrams2)) if bigrams1.union(bigrams2) else 0
    
    return {
        'sim_coseno': sim_coseno,
        'jaccard_words': jaccard_words,
        'jaccard_bigrams': jaccard_bigrams
    }

def extract_features_balanced(texto1, texto2, emb1, emb2):
    """5 features balanceadas - Óptimo costo/beneficio"""
    features = extract_features_minimal(texto1, texto2, emb1, emb2)
    
    words1 = texto1.lower().split()
    words2 = texto2.lower().split()
    set1 = set(words1)
    set2 = set(words2)
    
    # Agregar overlap coefficient
    common_words = set1.intersection(set2)
    overlap_coef = len(common_words) / min(len(set1), len(set2)) if min(len(set1), len(set2)) > 0 else 0
    
    # Agregar length ratio
    len1, len2 = len(texto1), len(texto2)
    len_ratio = min(len1, len2) / max(len1, len2) if max(len1, len2) > 0 else 0
    
    features['overlap_coef'] = overlap_coef
    features['len_ratio'] = len_ratio
    
    return features

def extract_features_full(texto1, texto2, emb1, emb2):
    """Todas las features - Máxima precisión"""
    features = extract_features_balanced(texto1, texto2, emb1, emb2)
    
    words1 = texto1.lower().split()
    words2 = texto2.lower().split()
    set1 = set(words1)
    set2 = set(words2)
    
    len1, len2 = len(texto1), len(texto2)
    
    # Features adicionales
    len_diff_rel = abs(len1 - len2) / max(len1, len2) if max(len1, len2) > 0 else 0
    
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
    
    features['len_diff_rel'] = len_diff_rel
    features['jaccard_trigrams'] = jaccard_trigrams
    features['jaccard_char_bigrams'] = jaccard_char_bigrams
    features['sentence_ratio'] = sentence_ratio
    features['vocab_ratio'] = vocab_ratio
    features['ttr_diff'] = ttr_diff
    
    return features

def main():
    print("\n" + "="*70)
    print(f"DETECTOR DE PLAGIO - VERSION: {VERSION}")
    print("="*70 + "\n")
    
    # Seleccionar función de features
    if VERSION == "MINIMAL":
        extract_features = extract_features_minimal
        print("Features: 3 (sim_coseno, jaccard_words, jaccard_bigrams)")
    elif VERSION == "BALANCED":
        extract_features = extract_features_balanced
        print("Features: 5 (críticas + overlap_coef + len_ratio)")
    else:
        extract_features = extract_features_full
        print("Features: 11 (todas)")
    
    print("\nCargando dataset...")
    df = pd.read_csv(DATASET)
    print(f"Dataset cargado: {len(df)} pares")
    
    categorias_esperadas = {'non', 'light', 'cut'}
    df = df[df['label'].isin(categorias_esperadas)]
    
    print(f"\nDistribución:")
    print(df['label'].value_counts().sort_index())
    
    print(f"\nCargando BERT: {MODEL_NAME}")
    encoder = SentenceTransformer(MODEL_NAME)
    
    print("\nGenerando embeddings BERT...")
    emb1_list = encoder.encode(df["texto1"].tolist(), show_progress_bar=True, batch_size=16)
    emb2_list = encoder.encode(df["texto2"].tolist(), show_progress_bar=True, batch_size=16)
    
    print("\nExtrayendo features...")
    features_list = []
    for i in range(len(df)):
        if i % 50 == 0:
            print(f"Procesando {i}/{len(df)}...", end='\r')
        features = extract_features(
            df.iloc[i]['texto1'],
            df.iloc[i]['texto2'],
            emb1_list[i],
            emb2_list[i]
        )
        features_list.append(features)
    print(f"Features extraídas: {len(features_list)}")
    
    df_features = pd.DataFrame(features_list)
    X = df_features.values
    y = df['label'].values
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTrain: {len(X_train)} | Test: {len(X_test)}")
    
    print("\nEntrenando Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        class_weight='balanced',
        n_jobs=-1
    )
    
    rf.fit(X_train, y_train)
    
    print("\n" + "="*70)
    print("RESULTADOS EN TEST")
    print("="*70)
    
    y_pred_test = rf.predict(X_test)
    accuracy_test = (y_pred_test == y_test).mean()
    f1_test = f1_score(y_test, y_pred_test, average='weighted')
    
    print(f"\nAccuracy: {accuracy_test:.4f}")
    print(f"F1-Score: {f1_test:.4f}\n")
    
    print("Classification Report:")
    print(classification_report(y_test, y_pred_test, zero_division=0))
    
    print("Matriz de Confusión:")
    cm = confusion_matrix(y_test, y_pred_test, labels=['non', 'light', 'cut'])
    cm_df = pd.DataFrame(cm, index=['non', 'light', 'cut'], columns=['non', 'light', 'cut'])
    print(cm_df)
    
    print("\n" + "="*70)
    print("FEATURE IMPORTANCE")
    print("="*70 + "\n")
    
    feature_importance = pd.DataFrame({
        'feature': df_features.columns,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(feature_importance.to_string(index=False))
    
    print("\n" + "="*70)
    print("GUARDANDO MODELO")
    print("="*70 + "\n")
    
    model_package = {
        'encoder': encoder,
        'classifier': rf,
        'feature_names': df_features.columns.tolist(),
        'classes': ['non', 'light', 'cut'],
        'version': VERSION,
        'model_info': {
            'encoder_name': MODEL_NAME,
            'encoder_dim': encoder.get_sentence_embedding_dimension(),
            'n_features': len(df_features.columns),
            'accuracy_test': float(accuracy_test),
            'f1_test': float(f1_test),
            'version': VERSION
        }
    }
    
    joblib.dump(model_package, f"modelo_plagio_{VERSION.lower()}.pkl")
    print(f"Modelo guardado: modelo_plagio_{VERSION.lower()}.pkl")
    
    print("\n" + "="*70)
    print("RESUMEN")
    print("="*70)
    print(f"Versión: {VERSION}")
    print(f"Features: {len(df_features.columns)}")
    print(f"Accuracy: {accuracy_test:.4f}")
    print(f"F1-Score: {f1_test:.4f}")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()