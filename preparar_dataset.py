import os
import pandas as pd

# Rutas
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
INFO_FILE = os.path.join(DATA_DIR, "file_information.csv")
OUTPUT_FILE = os.path.join(DATA_DIR, "dataset_clough.csv")

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
    print(" Cargando file_information.csv...")
    df_info = pd.read_csv(INFO_FILE)
    
    print(f" {len(df_info)} archivos encontrados")
    print(f" Categorías: {df_info['Category'].value_counts().to_dict()}")
    
    # Separar originales
    originals = df_info[df_info['Category'] == 'orig'].copy()
    print(f"\n {len(originals)} documentos originales (tasks a-e)")
    
    dataset_rows = []
    
    for _, orig_row in originals.iterrows():
        task_id = orig_row['Task']
        orig_filename = orig_row['File']
        orig_text = load_file_text(orig_filename)
        
        if not orig_text:
            print(f"  Texto vacío en {orig_filename}")
            continue
        
        # Encontrar todas las versiones plagiadas de esta tarea
        plagios = df_info[
            (df_info['Task'] == task_id) & 
            (df_info['Category'] != 'orig')
        ]
        
        print(f"\n Task {task_id}: {len(plagios)} casos de plagio")
        
        for _, plag_row in plagios.iterrows():
            plag_filename = plag_row['File']
            plag_text = load_file_text(plag_filename)
            plag_type = plag_row['Category']
            
            if not plag_text:
                print(f"    Texto vacío en {plag_filename}")
                continue
            
            dataset_rows.append({
                "texto1": orig_text,
                "texto2": plag_text,
                "label": plag_type,
                "task": task_id,
                "file_orig": orig_filename,
                "file_plag": plag_filename
            })
    
    # Crear DataFrame
    df_dataset = pd.DataFrame(dataset_rows)
    
    print(f"\n Dataset creado:")
    print(f"   Total pares: {len(df_dataset)}")
    print(f"   Distribución por categoría:")
    print(df_dataset['label'].value_counts())
    
    # Guardar
    df_dataset.to_csv(OUTPUT_FILE, index=False, encoding='utf-8')
    print(f"\n Dataset guardado en: {OUTPUT_FILE}")
    
    # Mostrar ejemplo
    print("\n Ejemplo de entrada:")
    ejemplo = df_dataset.iloc[0]
    print(f"   Original: {ejemplo['texto1'][:100]}...")
    print(f"   Plagiado: {ejemplo['texto2'][:100]}...")
    print(f"   Label: {ejemplo['label']}")

if __name__ == "__main__":
    main()