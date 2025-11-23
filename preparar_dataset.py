import os
import pandas as pd


# --- Rutas seguras basadas en la ubicación del script ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Carpeta donde están tasks/, sources/ y file_information.csv
DATASET_DIR = os.path.join(BASE_DIR, "data")

TASK_DIR = DATASET_DIR
ANSWER_DIR = DATASET_DIR

SOURCE_DIR = os.path.join(DATASET_DIR, "sources")

OUTPUT_FILE = "data/dataset_plagio_manual.csv"


INFO_FILE = os.path.join(DATASET_DIR, "file_information.csv")



def get_column(df, possible_names):
    """Devuelve la primera columna que exista en el CSV."""
    for name in possible_names:
        if name in df.columns:
            return name
    raise ValueError(f"Ninguna de estas columnas existe: {possible_names}")


def load_file_text(filename):
    """Carga texto de un archivo dentro de tasks/"""
    path = os.path.join(TASK_DIR, filename)
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def main():
    df = pd.read_csv(INFO_FILE)

    # Detectar columnas correctas (compatibilidad universal)
    file_col = get_column(df, ["File", "filename", "file"])
    task_col = get_column(df, ["Task", "task"])
    label_col = get_column(df, ["Category", "class", "label", "type"])

    print("Columnas detectadas:")
    print(" - Archivo:", file_col)
    print(" - Task:", task_col)
    print(" - Label:", label_col)

    # Separar originales
    originals = df[df[label_col] == "orig"]

    dataset_rows = []

    for _, orig_row in originals.iterrows():
        task_id = orig_row[task_col]
        orig_filename = orig_row[file_col]
        orig_text = load_file_text(orig_filename)

        # Todos los plagios/variantes de esa tarea
        related = df[(df[task_col] == task_id) & (df[label_col] != "orig")]

        for _, row in related.iterrows():
            plag_filename = row[file_col]
            plag_text = load_file_text(plag_filename)
            plag_type = row[label_col]

            dataset_rows.append({
                "original": orig_text,
                "texto": plag_text,
                "label": plag_type
            })

    os.makedirs("data", exist_ok=True)

    out_df = pd.DataFrame(dataset_rows)
    out_df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")

    print(f"\nDataset generado correctamente: {OUTPUT_FILE}")
    print(f"Total muestras: {len(out_df)}")


if __name__ == "__main__":
    main()
