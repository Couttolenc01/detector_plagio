"""
Script para fusionar dataset_clough.csv con dataset_plagio_manual.csv
"""
import pandas as pd

print("="*70)
print("FUSIONANDO DATASETS")
print("="*70 + "\n")

# Cargar datasets
print("Cargando datasets originales...")
df_clough = pd.read_csv("dataset_clough.csv")
df_manual = pd.read_csv("dataset_plagio_manual.csv")

print(f"Dataset Clough: {len(df_clough)} pares")
print(f"Dataset Manual: {len(df_manual)} pares")

# Mapeo de etiquetas
print("\nMapeando etiquetas...")
label_map = {
    'plagio_alto': 'cut',
    'plagio_leve': 'light',
    'no_plagio': 'non'
}

# Renombrar columnas del dataset manual para que coincidan
df_manual_renamed = df_manual.rename(columns={
    'texto_A': 'texto1',
    'texto_B': 'texto2',
    'etiqueta': 'label'
})

# Mapear etiquetas
df_manual_renamed['label'] = df_manual_renamed['label'].map(label_map)

# Verificar que no haya valores nulos después del mapeo
if df_manual_renamed['label'].isna().any():
    print("\nWARNING: Algunas etiquetas no se mapearon correctamente")
    print(df_manual['etiqueta'].unique())
else:
    print("✓ Todas las etiquetas mapeadas correctamente")

# Agregar columnas faltantes del dataset Clough (si existen)
clough_cols = set(df_clough.columns)
manual_cols = set(df_manual_renamed.columns)

missing_cols = clough_cols - manual_cols
if missing_cols:
    print(f"\nColumnas faltantes en dataset manual: {missing_cols}")
    for col in missing_cols:
        if col not in ['texto1', 'texto2', 'label']:
            df_manual_renamed[col] = 'manual_dataset'  # Valor por defecto

# Asegurar mismo orden de columnas
common_cols = ['texto1', 'texto2', 'label']
extra_cols = [col for col in df_clough.columns if col not in common_cols]

df_clough = df_clough[common_cols + extra_cols]
df_manual_renamed = df_manual_renamed[common_cols + [col for col in extra_cols if col in df_manual_renamed.columns]]

# Fusionar datasets
print("\nFusionando datasets...")
df_combined = pd.concat([df_clough, df_manual_renamed], ignore_index=True)

print(f"\n✓ Dataset fusionado: {len(df_combined)} pares totales")

# Verificar distribución
print("\n" + "="*70)
print("DISTRIBUCIÓN FINAL")
print("="*70)
print(df_combined['label'].value_counts().sort_index())

# Calcular estadísticas
dist = df_combined['label'].value_counts()
print("\nPorcentajes:")
for label in sorted(dist.index):
    count = dist[label]
    pct = count / len(df_combined) * 100
    print(f"  {label:6s}: {count:3d} ({pct:5.1f}%)")

# Verificar balance
min_class = dist.min()
max_class = dist.max()
balance_ratio = max_class / min_class if min_class > 0 else float('inf')

print(f"\nRatio de desbalance: {balance_ratio:.2f}:1")
if balance_ratio > 3:
    print("⚠️  Dataset desbalanceado (usar class_weight='balanced')")
else:
    print("✓ Dataset razonablemente balanceado")

# Verificar duplicados
print("\n" + "="*70)
print("VERIFICACIÓN DE DUPLICADOS")
print("="*70)

duplicados_texto1 = df_combined['texto1'].duplicated().sum()
duplicados_texto2 = df_combined['texto2'].duplicated().sum()
duplicados_pares = df_combined[['texto1', 'texto2']].duplicated().sum()

print(f"Duplicados en texto1: {duplicados_texto1}")
print(f"Duplicados en texto2: {duplicados_texto2}")
print(f"Pares duplicados completos: {duplicados_pares}")

if duplicados_pares > 0:
    print(f"\n⚠️  Eliminando {duplicados_pares} pares duplicados...")
    df_combined = df_combined.drop_duplicates(subset=['texto1', 'texto2'], keep='first')
    print(f"✓ Dataset final: {len(df_combined)} pares únicos")

# Guardar dataset combinado
output_file = "dataset_combined.csv"
df_combined.to_csv(output_file, index=False)

print("\n" + "="*70)
print("GUARDADO")
print("="*70)
print(f"✓ Dataset guardado en: {output_file}")
print(f"  Total de pares: {len(df_combined)}")
print(f"  Categorías: {', '.join(sorted(df_combined['label'].unique()))}")

# Crear versión solo con columnas esenciales (más limpia)
df_clean = df_combined[['texto1', 'texto2', 'label']].copy()
clean_file = "dataset_combined_clean.csv"
df_clean.to_csv(clean_file, index=False)
print(f"\n✓ Dataset limpio guardado en: {clean_file}")
print(f"  (Solo columnas: texto1, texto2, label)")

print("\n" + "="*70)
print("RESUMEN")
print("="*70)
print(f"Dataset Original:  {len(df_clough):3d} pares")
print(f"Dataset Manual:    {len(df_manual):3d} pares")
print(f"Dataset Combinado: {len(df_combined):3d} pares")
print(f"Incremento:        {len(df_combined) - len(df_clough):3d} pares (+{((len(df_combined) - len(df_clough))/len(df_clough)*100):.0f}%)")
print("\n¡Listo para entrenar con muchos más datos!")