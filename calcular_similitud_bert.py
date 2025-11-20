import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# 1. Cargar dataset
df = pd.read_csv("dataset_plagio_manual.csv")

# 2. Modelo de embeddings (RoBERTa)
model = SentenceTransformer("sentence-transformers/all-roberta-large-v1")

# 3. Extraer textos
textos_A = df["texto_A"].tolist()
textos_B = df["texto_B"].tolist()

# 4. Obtener embeddings
emb_A = model.encode(textos_A)
emb_B = model.encode(textos_B)

# 5. Calcular similitud coseno
sims = []
for a, b in zip(emb_A, emb_B):
    sims.append(cosine_similarity([a], [b])[0][0])

df["sim_coseno"] = sims

# 6. Guardar resultado
df.to_csv("resultado_similitud.csv", index=False, encoding="utf-8")

print("Similitud calculada. Archivo: resultado_similitud.csv\n")
print(df[["tipo_par", "etiqueta", "sim_coseno"]])
