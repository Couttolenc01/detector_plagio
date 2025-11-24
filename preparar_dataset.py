import os
import pandas as pd

# Rutas
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data2")  # ← Cambié a data2
INFO_FILE = os.path.join(DATA_DIR, "file_information.csv")
OUTPUT_FILE = os.path.join(BASE_DIR, "dataset_clough.csv")

def load_file_text(filename):
    """Carga el contenido de un archivo de texto"""
    path = os.path.join(DATA_DIR, filename)
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read().strip()
    except Exception as e:
        print(f"Error leyendo {filename}: {e}")
        return ""

def main():
    print("="*70)
    print("📁 PREPARANDO DATASET (3 CATEGORÍAS)")
    print("="*70)
    
    print(f"\n📂 Cargando {INFO_FILE}...")
    df_info = pd.read_csv(INFO_FILE)
    
    print(f"✅ {len(df_info)} archivos encontrados")
    print(f"\n📊 Categorías:")
    print(df_info['Category'].value_counts())
    
    # Verificar que solo haya las categorías esperadas
    categorias_validas = {'non', 'light', 'cut', 'orig'}
    categorias_encontradas = set(df_info['Category'].unique())
    
    if not categorias_encontradas.issubset(categorias_validas):
        print(f"\n⚠️  Categorías inesperadas: {categorias_encontradas - categorias_validas}")
    
    # Separar originales
    originals = df_info[df_info['Category'] == 'orig'].copy()
    print(f"\n📄 {len(originals)} documentos originales")
    
    if len(originals) == 0:
        print("\n⚠️  No se encontraron archivos originales (orig)")
        print("    Buscando patrones orig_text*.txt...")
        
        # Buscar archivos que empiecen con orig_
        import glob
        orig_files = glob.glob(os.path.join(DATA_DIR, "orig_*.txt"))
        
        if orig_files:
            print(f"✅ Encontrados {len(orig_files)} archivos orig_*")
            # Crear DataFrame de originales manualmente
            originals_data = []
            for filepath in orig_files:
                filename = os.path.basename(filepath)
                # Extraer task del nombre (orig_text1.txt -> 1)
                task_id = filename.replace("orig_text", "").replace(".txt", "")
                originals_data.append({
                    'File': filename,
                    'Task': task_id,
                    'Category': 'orig'
                })
            originals = pd.DataFrame(originals_data)
            print(f"📋 Tasks identificados: {sorted(originals['Task'].unique())}")
        else:
            print("❌ No se encontraron archivos originales")
            return
    
    dataset_rows = []
    
    for _, orig_row in originals.iterrows():
        task_id = str(orig_row['Task'])
        orig_filename = orig_row['File']
        orig_text = load_file_text(orig_filename)
        
        if not orig_text:
            print(f"  ⚠️  Texto vacío en {orig_filename}")
            continue
        
        # Encontrar todas las versiones de esta tarea
        # Buscar archivos que empiecen con t{task_id}_
        pattern_prefix = f"t{task_id}_"
        
        versiones = df_info[
            (df_info['File'].str.startswith(pattern_prefix)) &
            (df_info['Category'] != 'orig')
        ]
        
        print(f"\n📝 Task {task_id}: {len(versiones)} versiones")
        print(f"   Original: {orig_filename} ({len(orig_text)} chars)")
        
        for _, ver_row in versiones.iterrows():
            ver_filename = ver_row['File']
            ver_text = load_file_text(ver_filename)
            ver_type = ver_row['Category']
            
            if not ver_text:
                print(f"    ⚠️  Texto vacío en {ver_filename}")
                continue
            
            dataset_rows.append({
                "texto1": orig_text,
                "texto2": ver_text,
                "label": ver_type,
                "task": task_id,
                "file_orig": orig_filename,
                "file_plag": ver_filename
            })
            
            print(f"    ✅ {ver_type:5s}: {ver_filename} ({len(ver_text)} chars)")
    
    # Crear DataFrame
    df_dataset = pd.DataFrame(dataset_rows)
    
    print("\n" + "="*70)
    print("📊 DATASET CREADO")
    print("="*70)
    print(f"   Total de pares: {len(df_dataset)}")
    print(f"\n   Distribución por categoría:")
    distribucion = df_dataset['label'].value_counts()
    for label, count in distribucion.items():
        print(f"      {label:6s}: {count:3d} pares")
    
    print(f"\n   Pares por task:")
    pares_por_task = df_dataset.groupby('task').size()
    for task, count in pares_por_task.items():
        print(f"      Task {task}: {count} pares")
    
    # Guardar
    df_dataset.to_csv(OUTPUT_FILE, index=False, encoding='utf-8')
    print(f"\n✅ Dataset guardado en: {OUTPUT_FILE}")
    
    # Mostrar ejemplo
    print("\n" + "="*70)
    print("📋 EJEMPLO DE ENTRADA")
    print("="*70)
    if len(df_dataset) > 0:
        ejemplo = df_dataset.iloc[0]
        print(f"\nTask: {ejemplo['task']}")
        print(f"Categoría: {ejemplo['label']}")
        print(f"Archivo original: {ejemplo['file_orig']}")
        print(f"Archivo comparado: {ejemplo['file_plag']}")
        print(f"\nOriginal (primeros 150 chars):")
        print(f"   {ejemplo['texto1'][:150]}...")
        print(f"\nComparado (primeros 150 chars):")
        print(f"   {ejemplo['texto2'][:150]}...")
    
    # Estadísticas de longitud
    print("\n" + "="*70)
    print("📏 ESTADÍSTICAS DE LONGITUD")
    print("="*70)
    for label in ['non', 'light', 'cut']:
        subset = df_dataset[df_dataset['label'] == label]
        if len(subset) > 0:
            len_orig = subset['texto1'].str.len().mean()
            len_comp = subset['texto2'].str.len().mean()
            ratio = len_comp / len_orig if len_orig > 0 else 0
            print(f"   {label:6s}: orig={len_orig:.0f} chars, comp={len_comp:.0f} chars, ratio={ratio:.2f}")

if __name__ == "__main__":
    main()
    