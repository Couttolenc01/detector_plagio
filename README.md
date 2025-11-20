🧠 Detector de Plagio con Embeddings Semánticos (BERT / RoBERTa)

Este proyecto implementa un sistema de detección de similitud entre textos utilizando embeddings semánticos generados por modelos avanzados como RoBERTa (vía sentence-transformers).

El objetivo es identificar distintos niveles de plagio incluso cuando existe paráfrasis moderada o fuerte, algo que técnicas tradicionales como TF-IDF no pueden lograr.

El sistema permite cargar textos desde archivos .txt, generar un dataset dinámico y calcular similitud entre pares de textos mediante cosine similarity.

⸻

🚀 Funcionalidades principales
	•	Lectura dinámica de múltiples archivos .txt desde una carpeta.
	•	Construcción automática de un dataset a partir de un archivo pares_textos.csv.
	•	Generación de embeddings semánticos usando RoBERTa Large.
	•	Cálculo de similitud con cosine similarity.
	•	Clasificación conceptual de los niveles de plagio:
	•	🔴 Plagio alto (0.85 – 1.00)
	•	🟠 Plagio moderado (0.70 – 0.85)
	•	🟡 Plagio leve (0.55 – 0.70)
	•	🟢 No plagio (0.00 – 0.45)
	•	Exportación de resultados a CSV.

⸻

📦 Estructura del proyecto

detector_plagio/
├── textos/                        # Carpeta con archivos .txt
├── pares_textos.csv               # Define qué archivos se comparan entre sí
├── construir_dataset_desde_archivos.py
├── calcular_similitud_bert.py
├── dataset_manual.py              # (Opcional) Dataset estático para pruebas
├── resultado_similitud_archivos.csv
├── resultado_similitud.csv
├── requirements.txt
└── README.md


⸻

🛠 Instalación y ejecución

1️⃣ Clonar el repositorio

git clone https://github.com/Couttolenc01/detector_plagio.git
cd detector_plagio


⸻

2️⃣ Crear un entorno virtual

macOS / Linux

python3 -m venv venv

Windows

python -m venv venv


⸻

3️⃣ Activar el entorno virtual

macOS / Linux

source venv/bin/activate

Windows (PowerShell)

venv\Scripts\activate


⸻

4️⃣ Instalar dependencias

pip install -r requirements.txt


⸻

5️⃣ (Opcional) Crear un archivo .env si deseas usar API

touch .env

Y dentro escribir:

OPENAI_API_KEY=TU_API_KEY


⸻

6️⃣ Construir dataset desde archivos .txt

Este archivo genera automáticamente dataset_plagio_archivos.csv leyendo tus textos.

python construir_dataset_desde_archivos.py


⸻

7️⃣ Calcular la similitud con embeddings BERT

python calcular_similitud_bert.py

Los resultados aparecerán en:

resultado_similitud_archivos.csv


⸻

🧪 Ejemplo de salida

  tipo_par       etiqueta          sim_coseno
0 literal       plagio_alto        0.9400
1 moderado      plagio_moderado    0.9149
2 fuerte        plagio_leve        0.8508
3 no_rel        no_plagio          0.3720

La similitud se interpreta así:
	•	0.94 → Plagio alto (casi igual)
	•	0.91 → Plagio moderado (paráfrasis leve)
	•	0.85 → Paráfrasis fuerte
	•	0.37 → Textos no relacionados

⸻

🧠 ¿Cómo funciona este sistema?
	1.	Embeddings semánticos: Convertimos cada texto en un vector de alta dimensión usando un modelo pre-entrenado (RoBERTa Large).
	2.	Comparación vectorial: Medimos qué tan similares son los vectores mediante cosine similarity.
	3.	Interpretación: Valores cercanos a 1.0 indican alta similitud; valores cercanos a 0.0 indican que los textos no se parecen.

No se realiza entrenamiento propio: el sistema usa un modelo ya pre-entrenado en millones de pares de oraciones.
