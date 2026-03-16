# 🍷 Wine Intelligence API: Advanced Quality Predictor

Este proyecto implementa una solución de Machine Learning de extremo a extremo (End-to-End) para la predicción de la calidad de vinos basándose en sus características físico-químicas. A diferencia de aproximaciones básicas, este sistema utiliza un algoritmo de **Gradient Boosting** y expone una **REST API** profesional construida con **FastAPI**.

## 🚀 Características Principales
- **Algoritmo Avanzado:** Implementación de `HistGradientBoostingClassifier` para capturar relaciones no lineales complejas entre variables.
- **Inferencia Optimizada:** Uso de `lru_cache` para una gestión eficiente de la memoria y carga de artefactos del modelo.
- **Validación Robusta:** Esquemas de datos con **Pydantic** que aseguran la integridad de las entradas (inputs) a la API.
- **Respuesta Enriquecida:** La API no solo devuelve la calidad, sino también la confianza del modelo (%) y una recomendación de uso personalizada.

---

## 🛠️ Decisiones Técnicas (Justificación)

Para este proyecto, se optó por una arquitectura más robusta que un simple script de prueba:

1.  **Modelo (Boosting vs Bagging):** Se seleccionó **HistGradientBoosting**. A diferencia del Random Forest (que entrena árboles independientes), el Boosting entrena árboles de forma secuencial donde cada uno corrige los errores del anterior. Esto permite una precisión superior en datasets numéricos.
2.  **Escalabilidad:** Se incluyó un `StandardScaler` persistido para asegurar que los datos de entrada en la API reciban el mismo tratamiento estadístico que los datos de entrenamiento.
3.  **Arquitectura Decoupled:** La lógica de inferencia (`wine_engine.py`) está totalmente separada del servidor web (`app_server.py`), facilitando el mantenimiento y futuras integraciones.

---

## 📦 Instalación y Configuración

1. **Clonar o crear la carpeta del proyecto:**
   ```bash
   cd wine_quality_api

2. **Crear y activar un entorno virtual:**
    ```bash
    python -m venv .venv
    # En Windows:
    .venv\Scripts\activate
    # En macOS/Linux:
    source .venv/bin/activate

3. **Instalar las dependencias necesarias:**
    ```bash
    pip install -r requirements.txt

## 🏃‍♂️ Guía de Ejecución

Sigue este orden para asegurar el correcto funcionamiento de la API:

1. **Entrenamiento del Modelo**
Este script descarga el dataset oficial de UCI Machine Learning Repository, procesa los datos, entrena el modelo avanzado y genera los archivos .pkl.
    ```bash
    python model_builder.py

2. **Inicio de la API REST**
Una vez generados los modelos, lanza el servidor:
    ```bash
    python app_server.py

La API estará disponible en: http://localhost:8000

## 📑 Documentación de la API

FastAPI genera documentación automática interactiva. Puedes probar los endpoints directamente desde el navegador en:
http://localhost:8000/docs

Endpoint Principal: POST /predict
Recibe las características químicas del vino y devuelve un análisis completo.

Ejemplo de Body (JSON):

{
  "fixed_acidity": 7.4,
  "volatile_acidity": 0.7,
  "citric_acid": 0.0,
  "residual_sugar": 1.9,
  "chlorides": 0.076,
  "free_sulfur_dioxide": 11.0,
  "total_sulfur_dioxide": 34.0,
  "density": 0.9978,
  "pH": 3.51,
  "sulphates": 0.56,
  "alcohol": 9.4
}

## 🗂️ Estructura del Repositorio
app_server.py: Punto de entrada de FastAPI y configuración del servidor Uvicorn.

model_builder.py: Pipeline de descarga (UCI Repo) y entrenamiento del modelo.

wine_engine.py: Motor de inferencia, esquemas Pydantic y lógica de negocio.

settings.py: Configuración centralizada de rutas y variables de entorno.

saved_models/: Directorio donde se almacenan el modelo y el escalador entrenados.