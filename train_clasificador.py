"""
Detector de Plagio SIMPLE con RoBERTa Large
Usa solo similitud coseno de embeddings BERT (más robusto con pocos datos)
"""
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics import classification_report, confusion_matrix
import joblib

DATASET = "data/dataset_clough.csv"
MODEL_NAME = "sentence-transformers/all-roberta-large-v1"

def compute_similarity(emb1, emb2):
    """Calcula similitud coseno entre embeddings"""
    return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

def find_optimal_thresholds(df, emb1_list, emb2_list):
    """Encuentra umbrales óptimos analizando el dataset"""
    
    similarities = []
    labels = []
    
    for i in range(len(df)):
        sim = compute_similarity(emb1_list[i], emb2_list[i])
        similarities.append(sim)
        labels.append(df.iloc[i]['label'])
    
    df_sim = pd.DataFrame({
        'similarity': similarities,
        'label': labels
    })
    
    # Calcular estadísticas por clase
    stats = df_sim.groupby('label')['similarity'].describe()
    
    print("\n📊 ESTADÍSTICAS DE SIMILITUD POR CLASE:")
    print(stats[['mean', 'min', 'max']])
    
    # Usar medianas para definir umbrales
    medians = df_sim.groupby('label')['similarity'].median().sort_values(ascending=False)
    
    print("\n📊 MEDIANAS POR CLASE:")
    for label, median in medians.items():
        print(f"  {label}: {median:.4f}")
    
    # Definir umbrales entre clases
    cut_threshold = medians['cut'] - 0.02  # Un poco más permisivo
    light_threshold = (medians['light'] + medians['heavy']) / 2
    heavy_threshold = (medians['heavy'] + medians['non']) / 2
    
    thresholds = {
        'cut': float(cut_threshold),
        'light': float(light_threshold),
        'heavy': float(heavy_threshold)
    }
    
    print(f"\n🎯 UMBRALES CALCULADOS:")
    print(f"  cut:   >= {thresholds['cut']:.4f}")
    print(f"  light: >= {thresholds['light']:.4f}")
    print(f"  heavy: >= {thresholds['heavy']:.4f}")
    print(f"  non:   <  {thresholds['heavy']:.4f}")
    
    return thresholds, df_sim

def classify_by_threshold(similarity, thresholds):
    """Clasifica según similitud y umbrales"""
    if similarity >= thresholds['cut']:
        return 'cut'
    elif similarity >= thresholds['light']:
        return 'light'
    elif similarity >= thresholds['heavy']:
        return 'heavy'
    else:
        return 'non'

def main():
    print("="*70)
    print("🧠 DETECTOR DE PLAGIO SIMPLE CON ROBERTA (BERT)")
    print("   Método: Solo similitud coseno (sin ML)")
    print("="*70)
    
    print("\n📁 Cargando dataset...")
    df = pd.read_csv(DATASET)
    
    print(f"✅ {len(df)} pares cargados")
    print(f"\n📊 Distribución:")
    print(df['label'].value_counts())
    
    # Cargar RoBERTa Large
    print(f"\n🤖 Cargando {MODEL_NAME}...")
    encoder = SentenceTransformer(MODEL_NAME)
    print(f"✅ Modelo cargado: {encoder.get_sentence_embedding_dimension()} dimensiones")
    
    # Generar embeddings
    print("\n🧮 Generando embeddings con RoBERTa...")
    emb1_list = encoder.encode(df["texto1"].tolist(), show_progress_bar=True, batch_size=16)
    emb2_list = encoder.encode(df["texto2"].tolist(), show_progress_bar=True, batch_size=16)
    
    # Encontrar umbrales óptimos
    print("\n" + "="*70)
    print("🔍 ANALIZANDO SIMILITUDES")
    print("="*70)
    
    thresholds, df_sim = find_optimal_thresholds(df, emb1_list, emb2_list)
    
    # Evaluar con umbrales
    print("\n" + "="*70)
    print("📈 EVALUACIÓN CON UMBRALES")
    print("="*70)
    
    predictions = []
    for sim in df_sim['similarity']:
        pred = classify_by_threshold(sim, thresholds)
        predictions.append(pred)
    
    y_true = df_sim['label'].values
    y_pred = np.array(predictions)
    
    accuracy = (y_pred == y_true).mean()
    
    print(f"\n🎯 Accuracy total: {accuracy:.1%}")
    
    print("\n📋 Classification Report:")
    print(classification_report(y_true, y_pred))
    
    print("\n📊 Matriz de Confusión:")
    cm = confusion_matrix(y_true, y_pred, labels=['non', 'light', 'heavy', 'cut'])
    cm_df = pd.DataFrame(
        cm,
        index=['non', 'light', 'heavy', 'cut'],
        columns=['non', 'light', 'heavy', 'cut']
    )
    print(cm_df)
    
    # Accuracy por clase
    print("\n🎯 Accuracy por clase:")
    for i, label in enumerate(['non', 'light', 'heavy', 'cut']):
        if cm[i, :].sum() > 0:
            acc = cm[i, i] / cm[i, :].sum()
            correct = cm[i, i]
            total = cm[i, :].sum()
            print(f"  {label:6} {correct:2}/{total:2} = {acc*100:5.1f}%")
    
    # Mostrar ejemplos de errores
    print("\n" + "="*70)
    print("🔍 ANÁLISIS DE ERRORES")
    print("="*70)
    
    errors = df_sim[y_pred != y_true].copy()
    errors['predicted'] = y_pred[y_pred != y_true]
    
    if len(errors) > 0:
        print(f"\n❌ {len(errors)} errores encontrados:")
        for idx, row in errors.head(5).iterrows():
            print(f"\n  Real: {row['label']} | Predicho: {row['predicted']} | Sim: {row['similarity']:.4f}")
    else:
        print("\n✅ ¡Sin errores! Clasificación perfecta.")
    
    # Ajustar umbrales si hay muchos errores
    if accuracy < 0.65:
        print("\n" + "="*70)
        print("🔧 AJUSTANDO UMBRALES (accuracy < 65%)")
        print("="*70)
        
        # Usar percentiles en vez de medianas
        cut_p75 = df_sim[df_sim['label'] == 'cut']['similarity'].quantile(0.25)
        light_p50 = df_sim[df_sim['label'] == 'light']['similarity'].quantile(0.5)
        heavy_p50 = df_sim[df_sim['label'] == 'heavy']['similarity'].quantile(0.5)
        
        thresholds_adjusted = {
            'cut': float(cut_p75),
            'light': float(light_p50),
            'heavy': float(heavy_p50)
        }
        
        print(f"\n🎯 UMBRALES AJUSTADOS:")
        print(f"  cut:   >= {thresholds_adjusted['cut']:.4f}")
        print(f"  light: >= {thresholds_adjusted['light']:.4f}")
        print(f"  heavy: >= {thresholds_adjusted['heavy']:.4f}")
        
        # Re-evaluar
        predictions_adj = [classify_by_threshold(sim, thresholds_adjusted) for sim in df_sim['similarity']]
        accuracy_adj = (np.array(predictions_adj) == y_true).mean()
        
        print(f"\n📊 Nueva accuracy: {accuracy_adj:.1%}")
        
        if accuracy_adj > accuracy:
            print("✅ Mejora detectada! Usando umbrales ajustados.")
            thresholds = thresholds_adjusted
            accuracy = accuracy_adj
        else:
            print("⚠️  Sin mejora. Usando umbrales originales.")
    
    # Guardar modelo
    print("\n" + "="*70)
    print("💾 GUARDANDO MODELO")
    print("="*70)
    
    joblib.dump(encoder, "encoder_plagio.pkl")
    joblib.dump(thresholds, "umbrales_plagio.pkl")
    
    print(f"\n✅ Modelo guardado:")
    print(f"   - encoder_plagio.pkl (RoBERTa Large)")
    print(f"   - umbrales_plagio.pkl")
    print(f"\n📊 Accuracy final: {accuracy:.1%}")
    print(f"   Método: Similitud coseno + Umbrales calibrados")

if __name__ == "__main__":
    main()