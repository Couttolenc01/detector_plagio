import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import joblib
import time

# CONFIGURACIÓN
DATASET = "dataset_combined_clean.csv"
MODEL_NAME = "sentence-transformers/all-roberta-large-v1"

def compute_similarity(emb1, emb2):
    """Similitud coseno entre embeddings"""
    return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

def extract_features(texto1, texto2, emb1, emb2):
    """
    Features  
    
    - sim_coseno: Similitud semántica BERT
    - jaccard_words: Palabras en común
    - jaccard_bigrams: Bigramas en común
    - overlap_coef: Coeficiente de overlap
    - len_ratio: Ratio de longitud
    - jaccard_char_bigrams: Similitud a nivel carácter
    - vocab_ratio: Ratio de vocabulario
    """
    # Similitud BERT
    sim_coseno = compute_similarity(emb1, emb2)
    
    # Preprocesar textos
    words1 = texto1.lower().split()
    words2 = texto2.lower().split()
    set1 = set(words1)
    set2 = set(words2)
    
    #  Jaccard de palabras
    common_words = set1.intersection(set2)
    union_words = set1.union(set2)
    jaccard_words = len(common_words) / len(union_words) if union_words else 0
    
    #  Jaccard de bigramas
    bigrams1 = set(zip(words1[:-1], words1[1:])) if len(words1) > 1 else set()
    bigrams2 = set(zip(words2[:-1], words2[1:])) if len(words2) > 1 else set()
    jaccard_bigrams = len(bigrams1.intersection(bigrams2)) / len(bigrams1.union(bigrams2)) if bigrams1.union(bigrams2) else 0
    
    #  Overlap coefficient
    overlap_coef = len(common_words) / min(len(set1), len(set2)) if min(len(set1), len(set2)) > 0 else 0
    
    #  Length ratio
    len1, len2 = len(texto1), len(texto2)
    len_ratio = min(len1, len2) / max(len1, len2) if max(len1, len2) > 0 else 0
    
    #  Jaccard de char bigrams
    char_bigrams1 = set([texto1[i:i+2] for i in range(len(texto1)-1)])
    char_bigrams2 = set([texto2[i:i+2] for i in range(len(texto2)-1)])
    jaccard_char_bigrams = len(char_bigrams1.intersection(char_bigrams2)) / len(char_bigrams1.union(char_bigrams2)) if char_bigrams1.union(char_bigrams2) else 0
    
    #  Vocab ratio
    vocab_size1 = len(set1)
    vocab_size2 = len(set2)
    vocab_ratio = min(vocab_size1, vocab_size2) / max(vocab_size1, vocab_size2) if max(vocab_size1, vocab_size2) > 0 else 0
    
    return {
        'sim_coseno': sim_coseno,
        'jaccard_words': jaccard_words,
        'jaccard_bigrams': jaccard_bigrams,
        'overlap_coef': overlap_coef,
        'len_ratio': len_ratio,
        'jaccard_char_bigrams': jaccard_char_bigrams,
        'vocab_ratio': vocab_ratio
    }

def main():
    print("\n" + "+"*50)
    print("DETECTOR DE PLAGIO - BERT + RF")
    print("+"*50 + "\n")
        
    #  Cargar dataset
    df = pd.read_csv(DATASET)
    print(f"   Dataset: {len(df)} pares")
    
    categorias_esperadas = {'non', 'light', 'cut'}
    if not set(df['label'].unique()).issubset(categorias_esperadas):
        print("  Etiquetas inesperadas:", set(df['label'].unique()) - categorias_esperadas)
        df = df[df['label'].isin(categorias_esperadas)]
    
    print(f"\n Distribución:")
    dist = df['label'].value_counts().sort_index()
    for label in dist.index:
        print(f"   {label:6s}: {dist[label]:4d} ({dist[label]/len(df)*100:5.1f}%)")
    
    #  Cargar BERT
    encoder = SentenceTransformer(MODEL_NAME)
    
    #  Generar embeddings
    emb1_list = encoder.encode(df["texto1"].tolist(), show_progress_bar=True, batch_size=32)
    emb2_list = encoder.encode(df["texto2"].tolist(), show_progress_bar=True, batch_size=32)
    
    #  Extraer features
    features_list = []
    for i in range(len(df)):
        if i % 100 == 0:
            print(f"   Procesando {i}/{len(df)}...", end='\r')
        features = extract_features(
            df.iloc[i]['texto1'],
            df.iloc[i]['texto2'],
            emb1_list[i],
            emb2_list[i]
        )
        features_list.append(features)
    
    #  Preparar datos
    df_features = pd.DataFrame(features_list)
    X = df_features.values
    y = df['label'].values
    
    print(f"\n Dimensión final:")
    print(f"   Features: {X.shape[1]}")
    print(f"   Samples:  {X.shape[0]}")
    
    #  Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\n Train and Test Split:")
    print(f"   Train: {len(X_train)} ({len(X_train)/len(X)*100:.0f}%)")
    print(f"   Test:  {len(X_test)} ({len(X_test)/len(X)*100:.0f}%)")
    
    #  Entrenar Random Forest
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        class_weight='balanced',  # Importante si hay desbalance
        n_jobs=-1
    )
    
    rf.fit(X_train, y_train)
    
    #  Evaluar en test
    print("\n" + "+"*50)
    print("RESULTADOS EN TEST SET")
    print("+"*50)
    
    y_pred_test = rf.predict(X_test)
    accuracy_test = (y_pred_test == y_test).mean()
    f1_test = f1_score(y_test, y_pred_test, average='weighted')
    
    print(f"\n Métricas:")
    print(f"   Accuracy: {accuracy_test:.4f} ({accuracy_test*100:.2f}%)")
    print(f"   F1-Score: {f1_test:.4f} ({f1_test*100:.2f}%)")
    
    # matriz de confusion
    print(" Matriz de Confusión:")
    cm = confusion_matrix(y_test, y_pred_test, labels=['non', 'light', 'cut'])
    cm_df = pd.DataFrame(cm, index=['non', 'light', 'cut'], columns=['non', 'light', 'cut'])
    print(cm_df)
    
    #  cross-validation 
    print("\n" + "+"*50)
    print("VALIDACIÓN CRUZADA (5-fold)")
    print("+"*50 + "\n")
    
    cv_scores = cross_val_score(rf, X_train, y_train, cv=5, scoring='f1_weighted')
    print(f"   F1 por fold: {cv_scores}")
    print(f"   Media: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    #  Guardar modelo
    
    model_package = {
        'encoder': encoder,
        'classifier': rf,
        'feature_names': df_features.columns.tolist(),
        'classes': ['non', 'light', 'cut'],
        'model_info': {
            'encoder_name': MODEL_NAME,
            'encoder_dim': encoder.get_sentence_embedding_dimension(),
            'n_features': len(df_features.columns),
            'accuracy_test': float(accuracy_test),
            'f1_test': float(f1_test),
            'cv_f1_mean': float(cv_scores.mean()),
            'cv_f1_std': float(cv_scores.std())
        }
    }
    
    model_file = "modelo_plagio_rf.pkl"
    joblib.dump(model_package, model_file)
    
    # 12. Tiempo total
    print("\n" + "+"*50)
    print("RESUMEN FINAL")
    print("+"*50)
    print(f"   Modelo:       BERT + Random Forest")
    print(f"   BERT:         {MODEL_NAME}")
    print(f"   Features:     {len(df_features.columns)}")
    print(f"   Train size:   {len(X_train)}")
    print(f"   Test size:    {len(X_test)}")
    print(f"   Accuracy:     {accuracy_test:.4f}")
    print(f"   F1-Score:     {f1_test:.4f}")
    print(f"   CV F1:        {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print("="*70 + "\n")
    print(" ¡Entrenamiento completado!")

if __name__ == "__main__":
    import os
    main()