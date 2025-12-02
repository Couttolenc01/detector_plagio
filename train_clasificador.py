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
    classification_report,
)
import joblib

# -----------------------------
# Funciones auxiliares
# -----------------------------
def limpiar_texto(txt: str) -> str:
    return " ".join(str(txt).lower().split())


def obtener_palabras(txt: str):
    txt = limpiar_texto(txt)
    return txt.split()


# -----------------------------
# 1. Cargar dataset
# -----------------------------
print("\n" + "+" * 50)
print("DETECTOR DE PLAGIO - BERT + RF")
print("+" * 50 + "\n")

df = pd.read_csv("dataset_combined_clean.csv", header=1)
print(f"   Dataset: {len(df)} pares")

# Asegurarnos de que las columnas claves existan
esperadas = {"texto1", "texto2", "label"}
faltantes = esperadas - set(df.columns)
if faltantes:
    raise ValueError(f"Faltan columnas en dataset_combined_clean.csv: {faltantes}")

print("\n Distribución de etiquetas:")
dist = df["label"].value_counts().sort_index()
for label in dist.index:
    pct = dist[label] / len(df) * 100
    print(f"   {label:12s}: {dist[label]:4d} ({pct:5.1f}%)")

# -----------------------------
# 2. Cargar modelo de embeddings
# -----------------------------
print("\n   Cargando modelo de embeddings RoBERTa...")
encoder_name = "sentence-transformers/all-roberta-large-v1"
encoder = SentenceTransformer(encoder_name)
print(f"   Modelo cargado: {encoder_name}")

# -----------------------------
# 3. Calcular embeddings
# -----------------------------
print("\n   Calculando embeddings...")
emb_1 = encoder.encode(df["texto1"].tolist(), show_progress_bar=True)
emb_2 = encoder.encode(df["texto2"].tolist(), show_progress_bar=True)

# -----------------------------
# 4. Calcular features (7 features)
# -----------------------------
print("\n   Calculando features...")

sim_list = []
jaccard_words_list = []
jaccard_bigrams_list = []
overlap_coef_list = []
len_ratio_list = []
jaccard_char_bigrams_list = []
vocab_ratio_list = []

for idx, (t1, t2, v1, v2) in enumerate(
    zip(df["texto1"], df["texto2"], emb_1, emb_2)
):
    if idx % 50 == 0:
        print(f"   Procesando {idx}/{len(df)}...", end="\r")

    # palabras
    words1 = obtener_palabras(t1)
    words2 = obtener_palabras(t2)
    set1 = set(words1)
    set2 = set(words2)

    # 1) similitud coseno entre embeddings
    sim_cos = cosine_similarity([v1], [v2])[0][0]

    # 2) Jaccard de palabras
    jacc_words = (
        len(set1 & set2) / len(set1 | set2)
        if (set1 | set2)
        else 0.0
    )

    # 3) Jaccard de bigramas de palabras
    bigrams1 = set(zip(words1[:-1], words1[1:])) if len(words1) > 1 else set()
    bigrams2 = set(zip(words2[:-1], words2[1:])) if len(words2) > 1 else set()
    jacc_bigrams = (
        len(bigrams1 & bigrams2) / len(bigrams1 | bigrams2)
        if (bigrams1 | bigrams2)
        else 0.0
    )

    # 4) Coeficiente de solapamiento
    overlap = (
        len(set1 & set2) / min(len(set1), len(set2))
        if min(len(set1), len(set2))
        else 0.0
    )

    # 5) Ratio de longitud de texto
    len_ratio = (
        min(len(t1), len(t2)) / max(len(t1), len(t2))
        if max(len(t1), len(t2))
        else 0.0
    )

    # 6) Jaccard de bigramas de caracteres
    chars1 = set(t1[i : i + 2] for i in range(len(t1) - 1))
    chars2 = set(t2[i : i + 2] for i in range(len(t2) - 1))
    jacc_char_big = (
        len(chars1 & chars2) / len(chars1 | chars2)
        if (chars1 | chars2)
        else 0.0
    )

    # 7) Ratio de vocabulario
    vocab_ratio = (
        min(len(set1), len(set2)) / max(len(set1), len(set2))
        if max(len(set1), len(set2))
        else 0.0
    )

    sim_list.append(sim_cos)
    jaccard_words_list.append(jacc_words)
    jaccard_bigrams_list.append(jacc_bigrams)
    overlap_coef_list.append(overlap)
    len_ratio_list.append(len_ratio)
    jaccard_char_bigrams_list.append(jacc_char_big)
    vocab_ratio_list.append(vocab_ratio)

df["sim_coseno"] = sim_list
df["jaccard_words"] = jaccard_words_list
df["jaccard_bigrams"] = jaccard_bigrams_list
df["overlap_coef"] = overlap_coef_list
df["len_ratio"] = len_ratio_list
df["jaccard_char_bigrams"] = jaccard_char_bigrams_list
df["vocab_ratio"] = vocab_ratio_list

# -----------------------------
# 5. Features y etiquetas
# -----------------------------
feature_cols = [
    "sim_coseno",
    "jaccard_words",
    "jaccard_bigrams",
    "overlap_coef",
    "len_ratio",
    "jaccard_char_bigrams",
    "vocab_ratio",
]

X = df[feature_cols].values

le = LabelEncoder()
y = le.fit_transform(df["label"])

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
log_reg = LogisticRegression(max_iter=1000)

rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    random_state=42,
)

clf = VotingClassifier(
    estimators=[("logreg", log_reg), ("rf", rf)],
    voting="soft",
)

# -----------------------------
# 8. Validación cruzada
# -----------------------------
print("\n" + "+" * 50)
print("VALIDACIÓN CRUZADA (5-fold)")
print("+" * 50 + "\n")

cv_scores = cross_val_score(
    clf,
    X_train,
    y_train,
    cv=5,
    scoring="f1_weighted",
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
print("\n" + "+" * 50)
print("RESULTADOS EN TEST SET")
print("+" * 50 + "\n")

y_pred = clf.predict(X_test)

acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average="weighted")

print(" Métricas:")
print(f"   Accuracy: {acc:.4f} ({acc * 100:.2f}%)")
print(f"   F1-Score: {f1:.4f} ({f1 * 100:.2f}%)\n")

labels_num = np.unique(y)
labels_str = le.inverse_transform(labels_num)

cm = confusion_matrix(y_test, y_pred, labels=labels_num)
cm_df = pd.DataFrame(cm, index=labels_str, columns=labels_str)

print(" Matriz de Confusión:")
print(cm_df, "\n")

print(" Reporte de clasificación:")
print(classification_report(y_test, y_pred, target_names=labels_str))

# -----------------------------
# 11. Calcular UMBRALES automáticos de similitud
# -----------------------------
print("\n" + "+" * 50)
print("CALCULANDO UMBRALES AUTOMÁTICOS DE SIMILITUD")
print("+" * 50 + "\n")

umbrales = {}
for label in df["label"].unique():
    sims = df.loc[df["label"] == label, "sim_coseno"]
    umbrales[label] = {
        "min": float(sims.min()),
        "p25": float(sims.quantile(0.25)),
        "median": float(sims.median()),
        "p75": float(sims.quantile(0.75)),
        "max": float(sims.max()),
    }
    print(f" {label}: {umbrales[label]}")

# -----------------------------
# 12. Guardar modelo, encoder y umbrales
# -----------------------------
model_package = {
    "encoder": encoder,
    "classifier": clf,
    "label_encoder": le,
    "feature_cols": feature_cols,
    "umbrales": umbrales,
}

joblib.dump(model_package, "modelo_plagio_rf.pkl")

print("\n Modelo completo guardado como modelo_plagio_rf.pkl")
print(" Features usadas:", feature_cols)

print("\n" + "+" * 50)
print("RESUMEN FINAL")
print("+" * 50)
print(f"   Modelo:       RoBERTa + VotingClassifier (LogReg + RF)")
print(f"   RoBERTa:      {encoder_name}")
print(f"   Features:     {len(feature_cols)}")
print(f"   Train size:   {len(X_train)}")
print(f"   Test size:    {len(X_test)}")
print(f"   Accuracy:     {acc:.4f}")
print(f"   F1-Score:     {f1:.4f}")
print(f"   CV F1:        {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
print("=" * 70 + "\n")
print(" ¡Entrenamiento completado!")