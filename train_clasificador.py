import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report
)
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
print("\n" + "+"*50)
print("DETECTOR DE PLAGIO - roBERTa + VotingClassifier")
print("+"*50 + "\n")

df = pd.read_csv("dataset_combined_clean.csv")
print(f"   Registros en dataset: {len(df)}")

print("\n Distribución de etiquetas:")
dist = df["etiqueta"].value_counts().sort_index()
for label in dist.index:
    pct = dist[label] / len(df) * 100
    print(f"   {label:12s}: {dist[label]:4d} ({pct:5.1f}%)")

# -----------------------------
# 2. Cargar modelo de embeddings
# -----------------------------
print("\n   Cargando modelo de embeddings roBERTa...")
encoder_name = "sentence-transformers/all-roberta-large-v1"
encoder = SentenceTransformer(encoder_name)
print(f"   Modelo cargado: {encoder_name}")

# -----------------------------
# 3. Calcular embeddings
# -----------------------------
print("\n   Calculando embeddings...")
emb_A = encoder.encode(df["texto_A"].tolist(), show_progress_bar=True)
emb_B = encoder.encode(df["texto_B"].tolist(), show_progress_bar=True)

# -----------------------------
# 4. Calcular features
# -----------------------------
print("\n   Calculando features...")
sim_list = []
len_ratio_list = []
diff_len_chars_list = []
diff_len_words_list = []
jaccard_list = []

for idx, (texto_a, texto_b, vec_a, vec_b) in enumerate(
    zip(df["texto_A"], df["texto_B"], emb_A, emb_B)
):
    if idx % 50 == 0:
        print(f"   Procesando {idx}/{len(df)}...", end="\r")

    # 4.1 similitud coseno
    sim = cosine_similarity([vec_a], [vec_b])[0][0]

    # 4.2 ratio de longitud
    len_a = len(str(texto_a))
    len_b_raw = str(texto_b)
    len_b = len(len_b_raw) if len(len_b_raw) > 0 else 1  # evitar división entre 0
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

print("\n Dimensión final:")
print(f"   Features: {X.shape[1]}")
print(f"   Samples:  {X.shape[0]}")

# -----------------------------
# 6. Train/test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print("\n Train and Test Split:")
print(f"   Train: {len(X_train)} ({len(X_train) / len(X) * 100:.0f}%)")
print(f"   Test:  {len(X_test)} ({len(X_test) / len(X) * 100:.0f}%)")

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

# -----------------------------
# 8. Validación cruzada (en train)
# -----------------------------
print("\n" + "+"*50)
print("VALIDACIÓN CRUZADA (5-fold)")
print("+"*50 + "\n")

cv_scores = cross_val_score(
    clf,
    X_train,
    y_train,
    cv=5,
    scoring="f1_weighted"
)

print(f"   F1 por fold: {cv_scores}")
print(f"   Media: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# -----------------------------
# 9. Entrenar ensamble
# -----------------------------
print("\n   Entrenando VotingClassifier en train...")
clf.fit(X_train, y_train)

# -----------------------------
# 10. Evaluar en test
# -----------------------------
print("\n" + "+"*50)
print("RESULTADOS EN TEST SET")
print("+"*50 + "\n")

y_pred = clf.predict(X_test)

acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average="weighted")

print(" Métricas:")
print(f"   Accuracy: {acc:.4f} ({acc * 100:.2f}%)")
print(f"   F1-Score: {f1:.4f} ({f1 * 100:.2f}%)\n")

# Matriz de confusión y reporte por clase
labels_num = np.unique(y)
labels_str = le.inverse_transform(labels_num)

cm = confusion_matrix(y_test, y_pred, labels=labels_num)
cm_df = pd.DataFrame(cm, index=labels_str, columns=labels_str)

print(" Matriz de Confusión:")
print(cm_df, "\n")

print(" Reporte de clasificación:")
print(classification_report(y_test, y_pred, target_names=labels_str))

# -----------------------------
# 11. Guardar modelo y LabelEncoder
# -----------------------------
joblib.dump(clf, "modelo_plagio.pkl")
joblib.dump(le, "label_encoder.pkl")

print("\n Modelo (VotingClassifier) guardado como modelo_plagio.pkl")
print(" LabelEncoder guardado como label_encoder.pkl")
print(" Features usadas:", feature_cols)

# -----------------------------
# 12. Resumen final
# -----------------------------
print("\n" + "+"*50)
print("RESUMEN FINAL")
print("+"*50)
print(f"   Modelo:       roBERTa + VotingClassifier (LogReg + RF)")
print(f"   roBERTa:         {encoder_name}")
print(f"   Features:     {len(feature_cols)}")
print(f"   Train size:   {len(X_train)}")
print(f"   Test size:    {len(X_test)}")
print(f"   Accuracy:     {acc:.4f}")
print(f"   F1-Score:     {f1:.4f}")
print(f"   CV F1:        {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
print("="*70 + "\n")
print(" ¡Entrenamiento completado!")
