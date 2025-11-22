import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import joblib

# 1. Cargar dataset
df = pd.read_csv("dataset_plagio_manual.csv")
print("Registros en dataset:", len(df))

# 2. Cargar modelo de embeddings
encoder = SentenceTransformer("sentence-transformers/all-roberta-large-v1")

# 3. Calcular embeddings y similitud
emb_A = encoder.encode(df["texto_A"].tolist())
emb_B = encoder.encode(df["texto_B"].tolist())

sims = []
for a, b in zip(emb_A, emb_B):
    sims.append(cosine_similarity([a], [b])[0][0])

df["sim_coseno"] = sims

# 4. Feature adicional
df["len_ratio"] = df["texto_A"].str.len() / df["texto_B"].str.len()

# 5. Features y etiquetas
X = df[["sim_coseno", "len_ratio"]]

le = LabelEncoder()
y = le.fit_transform(df["etiqueta"])

# 6. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 7. Entrenar modelo
clf = LogisticRegression()
clf.fit(X_train, y_train)

print("Accuracy en test:", clf.score(X_test, y_test))

# 8. Guardar modelos
joblib.dump(clf, "modelo_plagio.pkl")
joblib.dump(le, "label_encoder.pkl")

print("Modelo guardado como modelo_plagio.pkl")
print("LabelEncoder guardado como label_encoder.pkl")