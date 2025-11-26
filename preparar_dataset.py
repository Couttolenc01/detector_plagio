import os
import pandas as pd
import glob

# Construcción de dataset_clough.csv desde los archivos

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data2")
INFO_FILE = os.path.join(DATA_DIR, "file_information.csv")
OUTPUT_CLOUGH = os.path.join(BASE_DIR, "dataset_clough.csv")


def load_file_text(filename):
    """Carga texto de un archivo"""
    try:
        with open(os.path.join(DATA_DIR, filename), "r", encoding="utf-8", errors="ignore") as f:
            return f.read().strip()
    except:
        return ""


def build_clough_dataset():
    print("Cargando archivo file_information.csv...")
    df_info = pd.read_csv(INFO_FILE)
    print(f"Archivos indexados: {len(df_info)}")

    categorias_validas = {'non', 'light', 'cut', 'orig'}
    categorias_encontradas = set(df_info['Category'].unique())

    if not categorias_encontradas.issubset(categorias_validas):
        print("Advertencia: hay categorías inesperadas en file_information.csv")

    originals = df_info[df_info['Category'] == 'orig'].copy()

    if len(originals) == 0:
        print("No se encontraron archivos 'orig'. Buscando orig_*.txt...")
        orig_files = glob.glob(os.path.join(DATA_DIR, "orig_*.txt"))
        originals_data = []

        for filepath in orig_files:
            filename = os.path.basename(filepath)
            task_id = filename.replace("orig_text", "").replace(".txt", "")
            originals_data.append({"File": filename, "Task": task_id, "Category": "orig"})

        originals = pd.DataFrame(originals_data)

    dataset_rows = []

    for _, orig_row in originals.iterrows():
        task_id = str(orig_row["Task"])
        orig_filename = orig_row["File"]
        orig_text = load_file_text(orig_filename)
        if not orig_text:
            continue

        pattern_prefix = f"t{task_id}_"
        versiones = df_info[
            (df_info["File"].str.startswith(pattern_prefix)) &
            (df_info["Category"] != "orig")
        ]

        for _, ver_row in versiones.iterrows():
            ver_filename = ver_row["File"]
            ver_text = load_file_text(ver_filename)
            if not ver_text:
                continue

            dataset_rows.append({
                "texto1": orig_text,
                "texto2": ver_text,
                "label": ver_row["Category"],
                "task": task_id,
                "file_orig": orig_filename,
                "file_plag": ver_filename
            })

    df_dataset = pd.DataFrame(dataset_rows)
    df_dataset.to_csv(OUTPUT_CLOUGH, index=False, encoding="utf-8")
    print(f"dataset_clough.csv generado: {len(df_dataset)} pares")

    return df_dataset

# Fusión con dataset manual

def merge_datasets():
    print("Cargando dataset_clough.csv...")
    df_clough = pd.read_csv("dataset_clough.csv")

    print("Cargando dataset_plagio_manual.csv...")
    df_manual = pd.read_csv("dataset_plagio_manual.csv")

    print(f"Clough: {len(df_clough)} pares")
    print(f"Manual: {len(df_manual)} pares")

    df_manual = df_manual.rename(columns={
        "texto_A": "texto1",
        "texto_B": "texto2",
        "etiqueta": "label"
    })

    label_map = {
        "plagio_alto": "cut",
        "plagio_leve": "light",
        "no_plagio": "non",
        "cut": "cut",
        "light": "light",
        "non": "non"
    }

    df_manual["label"] = df_manual["label"].map(label_map)
    df_manual = df_manual.dropna(subset=["label"])

    columnas_esenciales = ["texto1", "texto2", "label"]
    df_clough_clean = df_clough[columnas_esenciales].copy()
    df_manual_clean = df_manual[columnas_esenciales].copy()

    df_clough_clean["origen"] = "clough"
    df_manual_clean["origen"] = "manual"

    df_combined = pd.concat([df_clough_clean, df_manual_clean], ignore_index=True)

    antes = len(df_combined)
    df_combined = df_combined.drop_duplicates(subset=["texto1", "texto2"], keep="first")
    despues = len(df_combined)

    print(f"Duplicados eliminados: {antes - despues}")
    print(f"Total combinado final: {despues}")

    df_combined.to_csv("dataset_combined.csv", index=False, encoding="utf-8")
    df_combined[columnas_esenciales].to_csv("dataset_combined_clean.csv", index=False, encoding="utf-8")

    print("Archivos generados:")
    print("dataset_combined.csv")
    print("dataset_combined_clean.csv")

    return df_combined

if __name__ == "__main__":
    build_clough_dataset()
    merge_datasets()
