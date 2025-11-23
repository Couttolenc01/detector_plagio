import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib

DATASET = "data/dataset_plagio_manual.csv"

def main():
    print("Cargando dataset...")
    df = pd.read_csv(DATASET)

    print("Columnas detectadas:", df.columns.tolist())

    # Validar columnas correctas
    if not {"original", "texto", "label"}.issubset(df.columns):
        raise ValueError("El dataset debe tener columnas: original, texto, label")

    # Combinar los textos como entrada
    X_raw = df["original"] + " " + df["texto"]
    y = df["label"]

    print("Cargando encoder...")
    encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    print("Generando embeddings (tarda un poco)...")
    X_embeddings = encoder.encode(X_raw.tolist(), show_progress_bar=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X_embeddings, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = LogisticRegression(max_iter=2000)
    
    print("Entrenando clasificador...")
    clf.fit(X_train, y_train)

    print("\nEvaluación del modelo:\n")
    preds = clf.predict(X_test)
    print(classification_report(y_test, preds))

    print("Guardando modelos...")
    joblib.dump(encoder, "encoder_plagio.pkl")
    joblib.dump(clf, "modelo_plagio.pkl")

    print("Modelo entrenado y guardado correctamente ✔️")

if __name__ == "__main__":
    main()
