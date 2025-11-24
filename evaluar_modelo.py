import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
from sklearn.model_selection import train_test_split
import joblib

# ============================
# 1. Cargar dataset
# ============================
df = pd.read_csv("dataset_plagio_manual.csv")
print("Registros en dataset:", len(df))

# ============================
# 2. Cargar modelo y LabelEncoder
# ============================
modelo = joblib.load("modelo_plagio.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# ============================
# 3. Cargar modelo de embeddings (mismo que en train)
# ============================
encoder = SentenceTransformer("sentence-transformers/all-roberta-large-v1")

# ============================
# 4. Calcular las MISMAS features que en train
# ============================
emb_A = encoder.encode(df["texto_A"].tolist())
emb_B = encoder.encode(df["texto_B"].tolist())

sims = []
for a, b in zip(emb_A, emb_B):
    sims.append(cosine_similarity([a], [b])[0][0])

df["sim_coseno"] = sims
df["len_ratio"] = df["texto_A"].str.len() / df["texto_B"].str.len()

# X e y exactamente igual que en train_clasificador.py
X = df[["sim_coseno", "len_ratio"]].values
y = label_encoder.transform(df["etiqueta"])

# ============================
# 5. Evaluación en test (mismo split que en train)
# ============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

y_pred_test = modelo.predict(X_test)

print("\n=========== MÉTRICAS EN CONJUNTO DE TEST ===========")
print("Accuracy global (test):", accuracy_score(y_test, y_pred_test))
print("\nReporte por etiqueta (test):")
print(classification_report(y_test, y_pred_test, target_names=label_encoder.classes_))
print("Matriz de confusión (test):")
print(confusion_matrix(y_test, y_pred_test))

# ============================
# 6. Evaluación en TODO el dataset (opcional)
# ============================
y_pred_all = modelo.predict(X)

print("\n=========== MÉTRICAS EN TODO EL DATASET ===========")
print("Accuracy global (todo el dataset):", accuracy_score(y, y_pred_all))
print("\nReporte por etiqueta (todo el dataset):")
print(classification_report(y, y_pred_all, target_names=label_encoder.classes_))
print("Matriz de confusión (todo el dataset):")
print(confusion_matrix(y, y_pred_all))