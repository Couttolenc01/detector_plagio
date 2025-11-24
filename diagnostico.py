"""
Script para diagnosticar problemas en el dataset de plagio
"""
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import joblib

def compute_similarity(emb1, emb2):
    """Calcula similitud coseno"""
    return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

def analyze_text_pair(texto1, texto2):
    """Analiza características de un par de textos"""
    len1, len2 = len(texto1), len(texto2)
    words1 = set(texto1.lower().split())
    words2 = set(texto2.lower().split())
    
    # Palabras en común
    common_words = words1.intersection(words2)
    jaccard = len(common_words) / len(words1.union(words2)) if words1.union(words2) else 0
    
    # Diferencia de longitud
    len_ratio = min(len1, len2) / max(len1, len2) if max(len1, len2) > 0 else 0
    
    return {
        'len1': len1,
        'len2': len2,
        'len_ratio': len_ratio,
        'words1': len(words1),
        'words2': len(words2),
        'common_words': len(common_words),
        'jaccard': jaccard
    }

def main():
    print("="*70)
    print("🔍 DIAGNÓSTICO DEL DATASET")
    print("="*70)
    
    # Cargar dataset
    df = pd.read_csv("data/dataset_clough.csv")
    print(f"\n✅ {len(df)} pares cargados")
    
    # Verificar distribución básica
    print("\n📊 Distribución de etiquetas:")
    print(df['label'].value_counts())
    
    # Analizar algunos ejemplos por clase
    print("\n" + "="*70)
    print("📋 EJEMPLOS POR CLASE")
    print("="*70)
    
    for label in ['non', 'light', 'heavy', 'cut']:
        print(f"\n{'='*70}")
        print(f"CLASE: {label.upper()}")
        print('='*70)
        
        ejemplos = df[df['label'] == label].head(3)
        
        for idx, row in ejemplos.iterrows():
            print(f"\nEjemplo {idx + 1}:")
            print(f"Task: {row['task']}")
            print(f"Archivo original: {row['file_orig']}")
            print(f"Archivo plagiado: {row['file_plag']}")
            
            # Mostrar primeros 200 chars
            print(f"\nOriginal ({len(row['texto1'])} chars):")
            print(row['texto1'][:200] + "...")
            
            print(f"\nPlagiado ({len(row['texto2'])} chars):")
            print(row['texto2'][:200] + "...")
            
            # Análisis básico
            stats = analyze_text_pair(row['texto1'], row['texto2'])
            print(f"\nEstadísticas:")
            print(f"  Ratio longitud: {stats['len_ratio']:.2%}")
            print(f"  Jaccard (palabras): {stats['jaccard']:.2%}")
            print(f"  Palabras en común: {stats['common_words']}/{stats['words1']}")
    
    # Cargar embeddings si existen
    print("\n" + "="*70)
    print("🧮 ANÁLISIS DE SIMILITUDES")
    print("="*70)
    
    try:
        encoder = joblib.load("encoder_plagio.pkl")
        print("✅ Encoder cargado")
        
        print("\nCalculando similitudes para 10 ejemplos por clase...")
        
        for label in ['non', 'light', 'heavy', 'cut']:
            ejemplos = df[df['label'] == label].head(10)
            sims = []
            
            for _, row in ejemplos.iterrows():
                emb1 = encoder.encode([row['texto1']])[0]
                emb2 = encoder.encode([row['texto2']])[0]
                sim = compute_similarity(emb1, emb2)
                sims.append(sim)
            
            print(f"\n{label.upper()}:")
            print(f"  Similitudes: {[f'{s:.3f}' for s in sims]}")
            print(f"  Media: {np.mean(sims):.3f} ± {np.std(sims):.3f}")
            print(f"  Min-Max: [{min(sims):.3f}, {max(sims):.3f}]")
    
    except FileNotFoundError:
        print("⚠️  Encoder no encontrado. Ejecuta train_clasificador.py primero.")
    
    # Verificar si hay problemas obvios
    print("\n" + "="*70)
    print("⚠️  VERIFICACIÓN DE PROBLEMAS")
    print("="*70)
    
    # Verificar que task+orig sea consistente
    print("\n1. Verificando consistencia de archivos originales...")
    tasks_check = df.groupby('task')['file_orig'].nunique()
    if (tasks_check > 1).any():
        print("   ❌ ERROR: Una misma tarea tiene múltiples archivos originales!")
        print(tasks_check[tasks_check > 1])
    else:
        print("   ✅ Cada tarea tiene un único archivo original")
    
    # Verificar duplicados
    print("\n2. Verificando duplicados...")
    duplicados = df.duplicated(subset=['texto1', 'texto2']).sum()
    if duplicados > 0:
        print(f"   ⚠️  {duplicados} pares duplicados encontrados")
    else:
        print("   ✅ Sin pares duplicados")
    
    # Verificar textos vacíos
    print("\n3. Verificando textos vacíos...")
    vacios1 = (df['texto1'].str.strip() == '').sum()
    vacios2 = (df['texto2'].str.strip() == '').sum()
    if vacios1 > 0 or vacios2 > 0:
        print(f"   ❌ {vacios1} textos originales vacíos, {vacios2} textos plagiados vacíos")
    else:
        print("   ✅ No hay textos vacíos")
    
    # Analizar longitudes por clase
    print("\n4. Analizando longitudes por clase...")
    for label in ['non', 'light', 'heavy', 'cut']:
        subset = df[df['label'] == label]
        len1_mean = subset['texto1'].str.len().mean()
        len2_mean = subset['texto2'].str.len().mean()
        ratio = len2_mean / len1_mean if len1_mean > 0 else 0
        
        print(f"   {label:6s}: orig={len1_mean:.0f} chars, plag={len2_mean:.0f} chars, ratio={ratio:.2f}")
    
    print("\n" + "="*70)
    print("RECOMENDACIONES")
    print("="*70)
    print("""
Las clases 'non', 'light', 'heavy' y 'cut' están muy sobrelapadas en similitud.
Esto sugiere que:

1. El dataset Clough puede ser inherentemente difícil de clasificar solo con similitud
2. Puede haber inconsistencias en el etiquetado original
3. Se necesitan más features además de similitud coseno

SOLUCIONES PROPUESTAS:
- Usar un clasificador Random Forest con múltiples características
- Considerar características léxicas (Jaccard, n-gramas, etc.)
- Entrenar un modelo supervisado en lugar de usar solo umbrales
    """)

if __name__ == "__main__":
    main()