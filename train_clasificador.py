"""
Detector de Plagio con Random Forest + BERT (RoBERTa Large)
3 CATEGORÍAS: non, light, cut
Features: Embeddings BERT + Léxicas + Estructurales
"""
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import re

DATASET = "dataset_clough.csv"
MODEL_NAME = "sentence-transformers/all-roberta-large-v1"  # BERT/RoBERTa Large

def compute_similarity(emb1, emb2):
    """Similitud coseno entre embeddings BERT"""
    return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

def extract_features(texto1, texto2, emb1, emb2):
    """Extrae características múltiples de un par de textos"""
    
    # 1. Similitud semántica (BERT)
    sim_coseno = compute_similarity(emb1, emb2)
    
    # 2. Características de longitud
    len1, len2 = len(texto1), len(texto2)
    len_ratio = min(len1, len2) / max(len1, len2) if max(len1, len2) > 0 else 0
    len_diff_abs = abs(len1 - len2)
    len_diff_rel = len_diff_abs / max(len1, len2) if max(len1, len2) > 0 else 0
    
    # 3. Características léxicas (palabras)
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
    common_bigrams = bigrams1.intersection(bigrams2)
    union_bigrams = bigrams1.union(bigrams2)
    jaccard_bigrams = len(common_bigrams) / len(union_bigrams) if union_bigrams else 0
    
    trigrams1 = set(zip(words1[:-2], words1[1:-1], words1[2:])) if len(words1) > 2 else set()
    trigrams2 = set(zip(words2[:-2], words2[1:-1], words2[2:])) if len(words2) > 2 else set()
    common_trigrams = trigrams1.intersection(trigrams2)
    union_trigrams = trigrams1.union(trigrams2)
    jaccard_trigrams = len(common_trigrams) / len(union_trigrams) if union_trigrams else 0
    
    # 5. Caracteres
    char_bigrams1 = set([texto1[i:i+2] for i in range(len(texto1)-1)])
    char_bigrams2 = set([texto2[i:i+2] for i in range(len(texto2)-1)])
    common_char_bigrams = char_bigrams1.intersection(char_bigrams2)
    union_char_bigrams = char_bigrams1.union(char_bigrams2)
    jaccard_char_bigrams = len(common_char_bigrams) / len(union_char_bigrams) if union_char_bigrams else 0
    
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

def main():
    print("="*70)
    print("🤖 DETECTOR DE PLAGIO: BERT + RANDOM FOREST (3 CATEGORÍAS)")
    print("   Modelo: RoBERTa Large (1024 dim)")
    print("   Categorías: non, light, cut")
    print("="*70)
    
    # Cargar dataset
    print("\n📁 Cargando dataset...")
    df = pd.read_csv(DATASET)
    print(f"✅ {len(df)} pares cargados")
    
    # Verificar categorías
    categorias = df['label'].unique()
    print(f"\n📊 Categorías encontradas: {sorted(categorias)}")
    
    # Validar que solo haya 3 categorías
    categorias_esperadas = {'non', 'light', 'cut'}
    if set(categorias) != categorias_esperadas:
        print(f"\n⚠️  ADVERTENCIA: Se esperaban {categorias_esperadas}")
        print(f"    Pero se encontraron: {set(categorias)}")
        df = df[df['label'].isin(categorias_esperadas)]
        print(f"    Filtrado a {len(df)} pares con categorías válidas")
    
    print(f"\n📊 Distribución:")
    distribucion = df['label'].value_counts().sort_index()
    for label, count in distribucion.items():
        porcentaje = count/len(df)*100
        barra = "█" * int(porcentaje / 2)
        print(f"   {label:6s}: {count:3d} pares ({porcentaje:5.1f}%) {barra}")
    
    # Cargar BERT
    print(f"\n🤖 Cargando {MODEL_NAME}...")
    encoder = SentenceTransformer(MODEL_NAME)
    print(f"✅ Encoder BERT cargado: {encoder.get_sentence_embedding_dimension()} dimensiones")
    
    # Generar embeddings BERT
    print("\n🧮 Generando embeddings BERT...")
    print("   (Esto puede tomar varios minutos...)")
    emb1_list = encoder.encode(df["texto1"].tolist(), show_progress_bar=True, batch_size=16)
    emb2_list = encoder.encode(df["texto2"].tolist(), show_progress_bar=True, batch_size=16)
    print(f"✅ Embeddings generados: {emb1_list.shape}")
    
    # Extraer características
    print("\n🔍 Extrayendo características adicionales...")
    features_list = []
    
    for i in range(len(df)):
        if i % 20 == 0:
            print(f"   Procesando {i}/{len(df)}...", end='\r')
        
        features = extract_features(
            df.iloc[i]['texto1'],
            df.iloc[i]['texto2'],
            emb1_list[i],
            emb2_list[i]
        )
        features_list.append(features)
    
    print(f"   ✅ {len(features_list)} vectores completos extraídos          ")
    
    # Crear DataFrame de features
    df_features = pd.DataFrame(features_list)
    
    print("\n📊 Estadísticas de características:")
    print(f"{'Característica':<25} {'Media':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
    print("-" * 70)
    for col in df_features.columns:
        print(f"{col:<25} {df_features[col].mean():>8.3f} {df_features[col].std():>8.3f} {df_features[col].min():>8.3f} {df_features[col].max():>8.3f}")
    
    # Análisis por categoría
    print("\n" + "="*70)
    print("📊 PERFIL DE CADA CATEGORÍA")
    print("="*70)
    
    for label in sorted(df['label'].unique()):
        mask = df['label'] == label
        print(f"\n🏷️  {label.upper()} (n={mask.sum()}):")
        print(f"   Similitud BERT:     {df_features[mask]['sim_coseno'].mean():.3f} ± {df_features[mask]['sim_coseno'].std():.3f}")
        print(f"   Jaccard palabras:   {df_features[mask]['jaccard_words'].mean():.3f} ± {df_features[mask]['jaccard_words'].std():.3f}")
        print(f"   Jaccard bigramas:   {df_features[mask]['jaccard_bigrams'].mean():.3f} ± {df_features[mask]['jaccard_bigrams'].std():.3f}")
        print(f"   Ratio longitud:     {df_features[mask]['len_ratio'].mean():.3f} ± {df_features[mask]['len_ratio'].std():.3f}")
    
    # Preparar datos
    X = df_features.values
    y = df['label'].values
    
    # Split train/test estratificado
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print("\n" + "="*70)
    print("📊 SPLIT DE DATOS (80/20)")
    print("="*70)
    print(f"   Train: {len(X_train)} ejemplos")
    print(f"   Test:  {len(X_test)} ejemplos")
    
    print("\n   Distribución en Train:")
    train_dist = pd.Series(y_train).value_counts().sort_index()
    for label, count in train_dist.items():
        print(f"      {label:6s}: {count:3d}")
    
    print("\n   Distribución en Test:")
    test_dist = pd.Series(y_test).value_counts().sort_index()
    for label, count in test_dist.items():
        print(f"      {label:6s}: {count:3d}")
    
    # Entrenar Random Forest
    print("\n" + "="*70)
    print("🌲 ENTRENANDO RANDOM FOREST")
    print("="*70)
    
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        class_weight='balanced',
        n_jobs=-1,
        verbose=0
    )
    
    print("   Parámetros:")
    print(f"      - Árboles: 200")
    print(f"      - Max depth: 10")
    print(f"      - Class weight: balanced")
    
    rf.fit(X_train, y_train)
    print("\n✅ Modelo Random Forest entrenado")
    
    # Cross-validation
    print("\n📈 Validación Cruzada (5-fold)...")
    cv_scores = cross_val_score(rf, X_train, y_train, cv=5, scoring='accuracy')
    print(f"   Scores por fold: {[f'{s:.1%}' for s in cv_scores]}")
    print(f"   Accuracy CV:     {cv_scores.mean():.1%} ± {cv_scores.std():.1%}")
    
    # Evaluar en test
    print("\n" + "="*70)
    print("🎯 EVALUACIÓN EN TEST SET")
    print("="*70)
    
    y_pred_test = rf.predict(X_test)
    accuracy_test = (y_pred_test == y_test).mean()
    
    print(f"\n🏆 Accuracy en Test: {accuracy_test:.1%}")
    
    print("\n📋 Classification Report (Test):")
    print(classification_report(y_test, y_pred_test, zero_division=0))
    
    print("\n📊 Matriz de Confusión (Test):")
    cm = confusion_matrix(y_test, y_pred_test, labels=['non', 'light', 'cut'])
    cm_df = pd.DataFrame(
        cm,
        index=['non', 'light', 'cut'],
        columns=['non', 'light', 'cut']
    )
    print(cm_df)
    
    print("\n🎯 Accuracy por clase (Test):")
    for i, label in enumerate(['non', 'light', 'cut']):
        if cm[i, :].sum() > 0:
            acc = cm[i, i] / cm[i, :].sum()
            correct = cm[i, i]
            total = cm[i, :].sum()
            barra = "█" * int(acc * 20)
            print(f"   {label:6s}: {correct:2d}/{total:2d} = {acc*100:5.1f}% {barra}")
    
    # Feature importance
    print("\n" + "="*70)
    print("🔍 IMPORTANCIA DE CARACTERÍSTICAS")
    print("="*70)
    
    feature_importance = pd.DataFrame({
        'feature': df_features.columns,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n📊 Top características más importantes:")
    for idx, row in feature_importance.iterrows():
        bar_length = int(row['importance'] * 50)
        print(f"   {row['feature']:25s}: {row['importance']:.4f} {'█' * bar_length}")
    
    # Evaluar en dataset completo
    print("\n" + "="*70)
    print("📊 EVALUACIÓN EN DATASET COMPLETO")
    print("="*70)
    
    y_pred_all = rf.predict(X)
    accuracy_all = (y_pred_all == y).mean()
    
    print(f"\n🏆 Accuracy total: {accuracy_all:.1%}")
    
    cm_all = confusion_matrix(y, y_pred_all, labels=['non', 'light', 'cut'])
    cm_all_df = pd.DataFrame(
        cm_all,
        index=['non', 'light', 'cut'],
        columns=['non', 'light', 'cut']
    )
    print("\n📊 Matriz de Confusión (Completo):")
    print(cm_all_df)
    
    print("\n🎯 Accuracy por clase (Completo):")
    for i, label in enumerate(['non', 'light', 'cut']):
        if cm_all[i, :].sum() > 0:
            acc = cm_all[i, i] / cm_all[i, :].sum()
            correct = cm_all[i, i]
            total = cm_all[i, :].sum()
            barra = "█" * int(acc * 20)
            print(f"   {label:6s}: {correct:3d}/{total:3d} = {acc*100:5.1f}% {barra}")
    
    # Análisis de errores
    print("\n" + "="*70)
    print("🔍 ANÁLISIS DE ERRORES")
    print("="*70)
    
    errors_mask = y_pred_all != y
    num_errors = errors_mask.sum()
    
    if num_errors > 0:
        error_rate = num_errors/len(y)*100
        print(f"\n❌ {num_errors} errores de {len(y)} ({error_rate:.1f}%)")
        
        errors_df = df[errors_mask].copy()
        errors_df['predicted'] = y_pred_all[errors_mask]
        errors_features = df_features[errors_mask]
        
        print("\n🔎 Primeros 5 errores:")
        for i, (idx, row) in enumerate(errors_df.head(5).iterrows()):
            feat = errors_features.iloc[i]
            print(f"\n  Error {i+1}:")
            print(f"    Real: {row['label']:5s} → Predicho: {row['predicted']:5s}")
            print(f"    Task: {row['task']}")
            print(f"    Archivo: {row['file_plag']}")
            print(f"    Similitud BERT: {feat['sim_coseno']:.3f}")
            print(f"    Jaccard words:  {feat['jaccard_words']:.3f}")
        
        # Confusiones
        print("\n📊 Matriz de confusiones:")
        confusion_pairs = {}
        for real, pred in zip(y[errors_mask], y_pred_all[errors_mask]):
            key = f"{real} → {pred}"
            confusion_pairs[key] = confusion_pairs.get(key, 0) + 1
        
        for pair, count in sorted(confusion_pairs.items(), key=lambda x: x[1], reverse=True):
            porcentaje = count/num_errors*100
            print(f"   {pair:12s}: {count:2d} veces ({porcentaje:5.1f}% de errores)")
    else:
        print("\n✅ ¡CLASIFICACIÓN PERFECTA! Sin errores.")
    
    # Guardar modelo
    print("\n" + "="*70)
    print("💾 GUARDANDO MODELO")
    print("="*70)
    
    model_package = {
        'encoder': encoder,
        'classifier': rf,
        'feature_names': df_features.columns.tolist(),
        'classes': ['non', 'light', 'cut'],
        'model_info': {
            'encoder_name': MODEL_NAME,
            'encoder_dim': encoder.get_sentence_embedding_dimension(),
            'n_features': len(df_features.columns),
            'n_estimators': 200,
            'accuracy_test': float(accuracy_test),
            'accuracy_full': float(accuracy_all)
        }
    }
    
    joblib.dump(model_package, "modelo_plagio_rf.pkl")
    
    print(f"\n✅ Modelo guardado: modelo_plagio_rf.pkl")
    
    print("\n" + "="*70)
    print("📊 RESUMEN FINAL")
    print("="*70)
    print(f"   Modelo:            BERT (RoBERTa Large) + Random Forest")
    print(f"   Dimensiones BERT:  {encoder.get_sentence_embedding_dimension()}")
    print(f"   Features totales:  {len(df_features.columns)}")
    print(f"   Categorías:        {', '.join(['non', 'light', 'cut'])}")
    print(f"   Accuracy Test:     {accuracy_test:.1%}")
    print(f"   Accuracy Completo: {accuracy_all:.1%}")
    print(f"   Árboles RF:        200")
    print("\n🎉 ¡Entrenamiento completado exitosamente!")

if __name__ == "__main__":
    main()
    