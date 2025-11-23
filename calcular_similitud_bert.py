import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
import joblib

DATASET = "dataset_clough.csv"
OUTPUT_EMB = "embeddings_clough.pkl"

def main():
    df = pd.read_csv(DATASET)

    model = SentenceTransformer("sentence-transformers/all-roberta-large-v1")

    emb1 = model.encode(df["texto1"].tolist(), convert_to_numpy=True, show_progress_bar=True)
    emb2 = model.encode(df["texto2"].tolist(), convert_to_numpy=True, show_progress_bar=True)

    data = {
        "embeddings1": emb1,
        "embeddings2": emb2,
        "labels": df["label"].tolist()
    }

    joblib.dump(data, OUTPUT_EMB)
    print("Embeddings generados:", OUTPUT_EMB)

if __name__ == "__main__":
    main()
