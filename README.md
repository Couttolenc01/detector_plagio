# 🧠 Detector de Plagio con Embeddings (RoBERTa) + Clasificación Supervisada

Este proyecto implementa un **sistema de detección automática de plagio** que compara dos textos y clasifica su nivel de similitud en tres categorías:

- **cut** → *plagio alto*  
- **light** → *plagio leve*  
- **non** → *no plagio*  

El sistema utiliza **embeddings semánticos basados en RoBERTa**, características léxicas adicionales y un modelo híbrido supervisado (Logistic Regression + Random Forest).  
También incluye una **interfaz web desarrollada con Streamlit** para facilitar su uso.

---

# ⚙️ ¿Cómo funciona el modelo?

El pipeline del detector tiene cuatro etapas principales:

---

## 1️⃣ Generación de embeddings semánticos (RoBERTa Large)

Los textos se transforman en vectores numéricos utilizando:

sentence-transformers/all-roberta-large-v1
---

Los embeddings permiten capturar similitud semántica profunda, detectando:

- paráfrasis  
- plagio estructural  
- similitudes conceptuales aun con palabras distintas  

La similitud entre textos se calcula con **similitud del coseno**.

---

## 2️⃣ Extracción de 7 features adicionales

Además del vector semántico, el modelo calcula **7 características léxicas/estadísticas**:

| Feature | Descripción |
|--------|-------------|
| **sim_coseno** | Similitud semántica entre embeddings |
| **jaccard_words** | Coincidencia entre palabras |
| **jaccard_bigrams** | Coincidencia entre pares de palabras |
| **overlap_coef** | Proporción de vocabulario compartido |
| **len_ratio** | Razón entre longitudes de los textos |
| **jaccard_char_bigrams** | Similitud entre bigramas de caracteres |
| **vocab_ratio** | Comparación entre vocabularios únicos |

Estas features refuerzan la clasificación, ya que los embeddings pueden agrupar textos demasiado similares aunque sean paráfrasis.

---

## 3️⃣ Clasificador supervisado: Logistic Regression + Random Forest

El modelo final es un ensamble mediante:

- **LogisticRegression**  
- **RandomForestClassifier**  

con **votación suave (soft voting)**.

Este enfoque mejora la robustez al clasificar entre plagio leve y alto, categorías que pueden ser difíciles de separar con embeddings únicamente.

El modelo guardado incluye:

modelo_plagio_rf.pkl
	•	encoder RoBERTa
	•	classifier (VotingClassifier)
	•	label_encoder
	•	feature_cols
	•	umbrales estadísticos

---

## 4️⃣ Interfaz web con Streamlit

La aplicación web (`app.py`) permite:

- Ingresar dos textos  
- Calcular similitud semántica  
- Clasificar automáticamente el nivel de plagio  
- Mostrar explicaciones para el usuario  

Ejecutar con:

```bash
streamlit run app.py

El dataset final usado para entrenar al modelo es:

dataset_combined_clean.csv

Contiene 300 pares de textos, divididos en:
	•	90 casos de cut (plagio alto)
	•	90 casos de light (plagio leve)
	•	120 casos de non (no plagio)

¿Cómo se generó?
	•	Se recopilaron textos originales de Internet.
	•	Se generaron variantes usando modelos de IA (ChatGPT / DeepSeek):
	•	plagio alto
	•	plagio leve
	•	no plagio (totalmente distinto)
	•	También se generaron textos completamente creados por IA.

Columnas del dataset:

Columna
Contenido
texto1
Texto original
texto2
Texto sospechoso
label
Clase objetivo (cut/light/non)

🧪 Resultados del modelo

Al entrenar con los 300 pares se obtuvieron los siguientes resultados:
	•	Accuracy: 86.67%
	•	F1-score ponderado: 86.73%
	•	Validación cruzada (5-fold): 0.7775 ± 0.0336

El modelo:
	•	distingue muy bien casos non
	•	confunde ocasionalmente light ↔ cut, lo cual es esperado debido a su cercanía semántica

--- 

🖥️ Cómo ejecutar el proyecto

1️⃣ Clonar el repositorio
git clone <URL_DEL_REPOSITORIO>
cd detector_plagio

2️⃣ Crear y activar el entorno virtual
macOS / Linux:
python3 -m venv venv
source venv/bin/activate

Windows:
python -m venv venv
venv\Scripts\activate

3️⃣ Instalar dependencias
pip install -r requirements.txt

4️⃣ Entrenar el modelo
python train_clasificador.py

Esto genera:
modelo_plagio_rf.pkl

5️⃣ Ejecutar la aplicación web
streamlit run app.py

