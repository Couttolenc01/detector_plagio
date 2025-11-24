import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import joblib

# -----------------------------
# Funciones 
# -----------------------------
def limpiar_texto(txt: str) -> str:
    return " ".join(str(txt).lower().split())

def obtener_palabras(txt: str):
    txt = limpiar_texto(txt)
    return txt.split()

def jaccard_palabras(texto_a: str, texto_b: str) -> float:
    palabras_a = set(obtener_palabras(texto_a))
    palabras_b = set(obtener_palabras(texto_b))
    if not palabras_a or not palabras_b:
        return 0.0
    interseccion = len(palabras_a & palabras_b)
    union = len(palabras_a | palabras_b)
    return interseccion / union

# -----------------------------
# 1. Cargar dataset
# -----------------------------
df = pd.read_csv("dataset_plagio_manual.csv")
print("Registros en dataset:", len(df))

# -----------------------------
# 2. Cargar modelo y LabelEncoder
# -----------------------------
modelo = joblib.load("modelo_plagio.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# -----------------------------
# 3. Cargar encoder de embeddings
# -----------------------------
encoder = SentenceTransformer("sentence-transformers/all-roberta-large-v1")

# -----------------------------
# 4. Calcular features 
# -----------------------------
emb_A = encoder.encode(df["texto_A"].tolist())
emb_B = encoder.encode(df["texto_B"].tolist())

sim_list = []
len_ratio_list = []
diff_len_chars_list = []
diff_len_words_list = []
jaccard_list = []

for texto_a, texto_b, vec_a, vec_b in zip(df["texto_A"], df["texto_B"], emb_A, emb_B):
    sim = cosine_similarity([vec_a], [vec_b])[0][0]

    len_a = len(str(texto_a))
    len_b = len(str(texto_b)) if len(str(texto_b)) > 0 else 1
    len_ratio = len_a / len_b

    diff_len_chars = abs(len_a - len_b)

    palabras_a = obtener_palabras(texto_a)
    palabras_b = obtener_palabras(texto_b)
    diff_len_words = abs(len(palabras_a) - len(palabras_b))

    jacc = jaccard_palabras(texto_a, texto_b)

    sim_list.append(sim)
    len_ratio_list.append(len_ratio)
    diff_len_chars_list.append(diff_len_chars)
    diff_len_words_list.append(diff_len_words)
    jaccard_list.append(jacc)

df["sim_coseno"] = sim_list
df["len_ratio"] = len_ratio_list
df["diff_len_chars"] = diff_len_chars_list
df["diff_len_words"] = diff_len_words_list
df["jaccard_palabras"] = jaccard_list

feature_cols = [
    "sim_coseno",
    "len_ratio",
    "diff_len_chars",
    "diff_len_words",
    "jaccard_palabras",
]

X = df[feature_cols].values
y = label_encoder.transform(df["etiqueta"])

# -----------------------------
# 5. Evaluación en test
# -----------------------------
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

# -----------------------------
# 6. Evaluación en todo el dataset
# -----------------------------
y_pred_all = modelo.predict(X)

print("\n=========== MÉTRICAS EN TODO EL DATASET ===========")
print("Accuracy global (todo el dataset):", accuracy_score(y, y_pred_all))
print("\nReporte por etiqueta (todo el dataset):")
print(classification_report(y, y_pred_all, target_names=label_encoder.classes_))
print("Matriz de confusión (todo el dataset):")
print(confusion_matrix(y, y_pred_all))