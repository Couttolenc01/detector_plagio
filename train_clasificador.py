import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import joblib

# -----------------------------
# Funciones
# -----------------------------
def limpiar_texto(txt: str) -> str:
    # Minúsculas y quitar espacios extra
    return " ".join(str(txt).lower().split())

def obtener_palabras(txt: str):
    txt = limpiar_texto(txt)
    # Separar por espacios
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
# 2. Cargar modelo de embeddings
# -----------------------------
encoder = SentenceTransformer("sentence-transformers/all-roberta-large-v1")

# -----------------------------
# 3. Calcular embeddings
# -----------------------------
emb_A = encoder.encode(df["texto_A"].tolist())
emb_B = encoder.encode(df["texto_B"].tolist())

# -----------------------------
# 4. Calcular features
# -----------------------------
sim_list = []
len_ratio_list = []
diff_len_chars_list = []
diff_len_words_list = []
jaccard_list = []

for texto_a, texto_b, vec_a, vec_b in zip(df["texto_A"], df["texto_B"], emb_A, emb_B):
    # 4.1 similitud coseno
    sim = cosine_similarity([vec_a], [vec_b])[0][0]

    # 4.2 ratio de longitud
    len_a = len(str(texto_a))
    len_b = len(str(texto_b)) if len(str(texto_b)) > 0 else 1  # evitar división entre 0
    len_ratio = len_a / len_b

    # 4.3 diferencia en número de caracteres
    diff_len_chars = abs(len_a - len_b)

    # 4.4 diferencia en número de palabras
    palabras_a = obtener_palabras(texto_a)
    palabras_b = obtener_palabras(texto_b)
    diff_len_words = abs(len(palabras_a) - len(palabras_b))

    # 4.5 Jaccard de palabras
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

# -----------------------------
# 5. Features y etiquetas
# -----------------------------
feature_cols = [
    "sim_coseno",
    "len_ratio",
    "diff_len_chars",
    "diff_len_words",
    "jaccard_palabras",
]

X = df[feature_cols].values

le = LabelEncoder()
y = le.fit_transform(df["etiqueta"])

# -----------------------------
# 6. Train/test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# -----------------------------
# 7. Definir modelos base
# -----------------------------
log_reg = LogisticRegression(
    max_iter=1000
)

rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    random_state=42
)

# Ensamble: usa ambos modelos para decidir
clf = VotingClassifier(
    estimators=[("logreg", log_reg), ("rf", rf)],
    voting="soft"  # usa probabilidades de ambos
)

# Entrenar ensamble
clf.fit(X_train, y_train)

print("Accuracy en test (VotingClassifier):", clf.score(X_test, y_test))

# -----------------------------
# 8. Guardar modelo y LabelEncoder
# -----------------------------
joblib.dump(clf, "modelo_plagio.pkl")
joblib.dump(le, "label_encoder.pkl")

print("Modelo (VotingClassifier) guardado como modelo_plagio.pkl")
print("LabelEncoder guardado como label_encoder.pkl")
print("Features usadas:", feature_cols)