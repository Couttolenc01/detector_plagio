# 🧠 Detector de Plagio con Embeddings y Clasificación Supervisada

Este proyecto implementa un **detector de plagio de texto** 
El sistema analiza dos textos y clasifica el nivel de plagio en una de tres categorías:

- `plagio_alto`
- `plagio_leve`
- `no_plagio`  
  *(incluye casos de “ruido”, donde los textos no tienen ninguna relación entre sí)*

El proyecto también incluye una **interfaz web basada en Streamlit** para facilitar su uso.

---

## ⚙️ ¿Cómo funciona el modelo?

El pipeline del sistema tiene cuatro etapas principales:

### 1️⃣ Embeddings semánticos (RoBERTa / BERT)

Cada texto se convierte en un vector numérico utilizando:
SentenceTransformer(“sentence-transformers/all-roberta-large-v1”)

Estos embeddings capturan el **significado** del texto y permiten comparar semánticamente dos oraciones o párrafos completos.

La similitud entre ambos embeddings se calcula usando la **similitud coseno**, que indica qué tan parecidos son los textos a nivel de significado.

---

### 2️⃣ Features utilizadas para la clasificación

A partir de los textos y de sus embeddings, se calculan **5 features**:

1. **sim_coseno**  
   - Similitud entre embeddings.  
   - Mientras más alto, más parecidos en significado.

2. **len_ratio**  
   - Relación entre la longitud del texto A y B.  
   - Útil para detectar cuando un texto es una versión recortada/parafraseada del otro.

3. **diff_len_chars**  
   - Diferencia absoluta en número de caracteres.

4. **diff_len_words**  
   - Diferencia en número de palabras.

5. **jaccard_palabras**  
   - Similaridad entre conjuntos de palabras.  
   - Mide qué tantas palabras comparten.

Estas características juntas hacen que el modelo pueda identificar desde plagio literal hasta paráfrasis.

---

### 3️⃣ Clasificador supervisado

Se utiliza un modelo de **Regresión Logística (LogisticRegression)** para clasificar los pares en:

- `plagio_alto`
- `plagio_leve`
- `no_plagio`

El modelo entrena con los 5 features mencionados y con un conjunto balanceado de ejemplos reales y casos de “ruido”.

Los componentes entrenados se guardan como:

- `modelo_plagio.pkl`
- `label_encoder.pkl`

---

### 4️⃣ Interfaz web con Streamlit

La aplicación (`app.py`) permite:

- Ingresar dos textos.
- Analizar su similitud semántica.
- Mostrar:
  - Porcentaje aproximado de similitud.
  - Clasificación final del nivel de plagio.

Se ejecuta con:
streamlit run app.py

---

## 📚 Dataset utilizado

El dataset se encuentra en:
dataset_plagio_manual.csv

Contiene **120 pares de textos**, distribuidos así:

- **30** casos de `plagio_alto`
- **30** casos de `plagio_leve`
- **60** casos de `no_plagio`

Dentro de los casos `no_plagio` se incluyen también ejemplos de **ruido**:  
pares donde los textos NO tienen relación alguna.  
Esto ayuda a que el modelo sea más robusto y no se confunda frente a textos arbitrarios.

Columnas del dataset:

- `texto_A`
- `texto_B`
- `etiqueta`

---

## 🖥️ Cómo correr el proyecto

### 1️⃣ Clonar el repositorio

```bash
git clone <URL_DEL_REPOSITORIO>
- cd detector_plagio

- python3 -m venv venv
 source venv/bin/activate

- pip install -r requirements.txt

- pip preparar_dataset.py

- python train_clasificador.py

- streamlit run app.py