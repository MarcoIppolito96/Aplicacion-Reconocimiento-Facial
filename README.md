# TP1 - Sistema de Reconocimiento Facial

Plantilla base para desarrollar un sistema completo de detección, alineación, extracción de embeddings e identificación/verificación facial.

> 📄 El enunciado completo del trabajo práctico está en [ENUNCIADO.md](ENUNCIADO.md).

**Alumno:** Marco Ippolito

## 🚀 Cómo correrlo (entrega)

**Modelo elegido:** ArcFace (InsightFace `buffalo_l`) para los embeddings y SCRFD para la detección.
**Umbral:** `SIMILARITY_THRESHOLD=0.243` (similitud coseno), calibrado en validación para FAR = 1e-3 (ver `2-train.ipynb`, sección 4.3).

1. Clonar el repositorio y cambiar a la rama de la entrega:
   ```bash
   git clone https://github.com/MarcoIppolito96/Aplicacion-Reconocimiento-Facial.git
   cd Aplicacion-Reconocimiento-Facial
   git checkout feature/tp-reconocimiento-facial
   ```
2. Descargar los modelos desde el link público (solo lectura):
   **https://drive.google.com/drive/folders/1cw74S4P0jSiEI0Mlo_AYOl8vOoA3yGX9?usp=sharing**
   y copiar **estos dos archivos** en la carpeta `models/` de la raíz del repo:
   ```text
   models/
   ├── det_10g.onnx      # detector SCRFD
   └── w600k_r50.onnx    # reconocedor ArcFace (embedding de 512 dimensiones)
   ```
   > Fuente original de ambos modelos: paquete `buffalo_l` de InsightFace (https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip).
   > En el mismo link está `cnn_propia.onnx`, la CNN entrenada en `2-train.ipynb` (modelo comparado y descartado).
3. Levantar todo con Docker (el `.env.docker.example` ya está configurado):
   ```bash
   docker compose build
   docker compose up -d
   ```
4. Verificar:
   - `http://localhost:8000/health` debe responder `{"status":"ok","model":"w600k_r50.onnx"}`.
   - Frontend: `http://localhost:8080` (pestañas *Registrar identidad* y *Predecir*).
   - Swagger: `http://localhost:8000/docs`.

## Objetivo del backend

Implementar una API asincrónica en Python que permita:

- Registrar identidades (`/insert`)
- Ejecutar inferencia sobre una imagen (`/predict`)
- Consultar el estado del procesamiento asincrónico (`/status/{job_id}`)

La API responde `HTTP 202` con un `job_id` y luego permite consultar el resultado con estado:

```json
{
  "status": "done | inProgress | failed",
  "link": "url | none"
}
```

## Estructura

```text
tuia-face-recognition-app/
├── 1-pre-processing.ipynb       # Parte 1: dataset y preprocesamiento
├── 2-train.ipynb                # Parte 2: entrenamiento y comparación de modelos
├── src/
│   ├── app/
│   │   └── main.py              # App FastAPI + /health
│   ├── frontend/                # UI Gradio provista por la cátedra
│   │   ├── app.py
│   │   └── gradio_ui.py
│   └── lib/
│       ├── api.py               # Endpoints /upload, /insert, /predict, /status, /files
│       ├── config.py            # Configuración vía .env
│       ├── schemas.py
│       ├── face_pipeline.py     # Detección, alineación y preprocesamiento (compartido con las notebooks)
│       ├── services/
│       │   ├── face_service.py  # detect_faces, align_face y extract_embedding_from_face
│       │   └── task_manager.py
│       └── storage/
│           ├── base.py
│           ├── embedding_store.py   # Persistencia en JSON
│           └── pgvector_store.py    # Persistencia en PostgreSQL + pgvector
├── models/                      # Modelo entrenado (.pth / .onnx), no se versiona
├── output/                      # Imágenes subidas y resultados, no se versiona
├── requirements.txt
├── Dockerfile
├── Dockerfile.frontend
├── docker-compose.yml
├── .env.docker.example
└── .env.local.example
```

# Preparando el ambiente local

## Requisitos para trabajar de forma local

- Python 3.12
- Docker

## Configurá tu modelo

Entrená tu modelo y guardalo dentro de la carpeta `models`. El backend admite modelos con extensión **.pth** (PyTorch, se carga con `torch.load` el modelo completo) o **.onnx** (se carga con `onnxruntime`). Si usan otro framework, expórtenlo a **.onnx**.

Recordá actualizar la configuración del `.env` correspondiente (`MODEL_NAME`, `EMBEDDING_DIM`, `FACE_SIZE`, `SIMILARITY_THRESHOLD`) para que apunte a tu modelo.

> El backend no arranca si no existe el archivo `models/<MODEL_NAME>`.

Con docker compose, el entorno reinicia la aplicación y actualiza el código automáticamente al detectar cambios. Lo mismo ocurre en local si usan `--reload`.

El reinicio automático puede no funcionar en algunas versiones de Docker Desktop para *Windows*. En ese caso, vuelvan a correr los comandos del siguiente apartado para actualizar el código dentro de Docker.

## Opción 1 - Corriendo dentro de Docker

### 1. Construí y corré la aplicación

Ajustá el archivo **.env.docker.example** a tus necesidades. Luego corré desde la terminal:

```bash
docker compose build
docker compose up -d
```

## Opción 2 - Configurando el ambiente local

### 1. Instalá uv

Para usuarios de Linux o Mac:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Para usuarios de Windows:

```bash
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

[Documentación de uv](https://docs.astral.sh/uv/getting-started/installation/).

### 2. Configurá un ambiente virtual con Python 3.12

```bash
uv venv --python 3.12 .venv
```

### 3. Activá el ambiente virtual

```bash
source .venv/bin/activate
```

### 4. Instalá las dependencias

```bash
uv pip install -r requirements.txt
```

### 5. Iniciá la base de datos

```bash
docker compose up postgres -d
```

### 6. Iniciá el frontend

```bash
cd src
uvicorn frontend.app:app --port 8080
```

### 7. Iniciá el backend

Ajustá el archivo *.env.local.example* a tus necesidades. Desde la carpeta `src`, en otra terminal:

```bash
cp ../models/<NOMBRE_DE_TU_MODELO> models/
cp ../.env.local.example .env
uvicorn app.main:app --reload --port 8000
```

## Configuración

No hardcodeen parámetros: configúrenlos mediante `.env`.

1. Para la ejecución local, copien `.env.local.example` a `src/.env`.
2. Ajusten las variables del modelo, los paths y el threshold.
3. Configuren el detector: `DETECTOR_MODEL_NAME` (opcional, archivo `.pth`/`.onnx` en `models/`), `DET_SCORE_THRESHOLD`, `DET_INPUT_SIZE`, `MIN_FACE_SIZE` y `ALIGN_MARGIN`. Si su detector viene de una librería, sobrescriban `_load_detector` en `face_service.py`.
4. Opcional (habilitado por defecto): configuren la conexión a PostgreSQL + pgvector (`USE_PGVECTOR`). Con `USE_PGVECTOR=false`, los embeddings se guardan en un archivo JSON (`EMBEDDINGS_PATH`).

## Endpoints

- Backend: `http://localhost:8000` (documentación Swagger en `http://localhost:8000/docs`)
- PostgreSQL/pgvector: `localhost:5432`
- Frontend (provisto por la cátedra): `http://localhost:8080`

## Pipeline

### Provisto por la cátedra

- API asincrónica (`/upload`, `/insert`, `/predict`, `/status`) y manejo de jobs.
- Carga del modelo (`.pth` / `.onnx`) desde `models/`.
- Búsqueda por similitud configurable (`cosine` o `l2`) e identificación (`identify`).
- Manejo de desconocidos con `SIMILARITY_THRESHOLD`.
- Persistencia configurable en JSON o en PostgreSQL + pgvector (`USE_PGVECTOR`).
- Frontend para registrar identidades, predecir y visualizar los resultados (cajas, etiquetas y keypoints).

### Implementado (`src/lib/services/face_service.py` + `src/lib/face_pipeline.py`)

```text
imagen BGR → detect_faces (SCRFD) → align_face (5 landmarks + norm_crop 112×112)
           → extract_embedding_from_face (ArcFace, 512-d, L2) → identify (coseno ≥ 0.243, si no "unknown")
```

1. `detect_faces`: detector SCRFD (`det_10g.onnx`), cargado una sola vez en `_load_detector`. Filtra por `DET_SCORE_THRESHOLD` y `MIN_FACE_SIZE`, recorta las cajas a la imagen con `_clip_xyxy` y devuelve `[]` si no hay rostros.
2. `align_face`: vuelve a detectar en un recorte con margen (`ALIGN_MARGIN`) alrededor de la caja para obtener los 5 landmarks, se queda con la cara de mayor IoU con la caja pedida y alinea con una transformación de similitud a la plantilla de ArcFace (`norm_crop`, `FACE_SIZE`×`FACE_SIZE`). Los keypoints se devuelven **relativos a la caja**, como espera el frontend.
3. `extract_embedding_from_face`: preprocesamiento BGR → RGB, `(x − 127.5) / 127.5`, CHW con dimensión de batch; inferencia con onnxruntime (o PyTorch si el modelo es `.pth`) y normalización L2. Devuelve floats de Python de largo `EMBEDDING_DIM`.

El preprocesamiento vive en `face_pipeline.py` y lo importan **tanto las notebooks como el backend**, para garantizar que la entrada al modelo sea idéntica en entrenamiento e inferencia.

## Modelo y fine-tuning

Detalle completo en `2-train.ipynb`. Modelos: **https://drive.google.com/drive/folders/1cw74S4P0jSiEI0Mlo_AYOl8vOoA3yGX9?usp=sharing**

| | Modelo A: CNN propia (descartado) | Modelo B: ArcFace (**elegido**) |
|---|---|---|
| Arquitectura | 4 bloques (2× Conv 3×3 + BatchNorm + ReLU, MaxPool), Global Average Pooling, Dropout 0,3, Linear + BatchNorm → embedding | ResNet-50 (InsightFace `w600k_r50`) |
| Entrenamiento | Desde cero sobre 85 identidades de Pins (11.094 imágenes) | Preentrenado en WebFace600K (~600.000 identidades); sin fine-tuning |
| Pérdida | Cross-entropy (label smoothing 0,1, pesos por clase); se usa la capa previa al clasificador | ArcFace (margen angular aditivo) |
| Embedding | 128 dimensiones, L2 | 512 dimensiones, L2 |
| Parámetros / tamaño | 1,2 M / 4,8 MB | 43,6 M / 174 MB |
| Inferencia CPU (Colab) | 17 ms/rostro | 380 ms/rostro |

**Hiperparámetros de la CNN:** AdamW (lr 1e-3, weight decay 5e-4), cosine annealing, batch 128, 30 épocas con early stopping (paciencia 5) y augmentation solo en train. Mejor accuracy de validación: 87,9% (época 29), sin overfitting. Un experimento sin augmentation llegó a 100% en train (overfitting) y empeoró en personas nuevas (AUC 0,807 contra 0,868).

**Métricas sobre el test open-set** (20 personas nunca vistas por la CNN):

| Métrica | CNN propia | ArcFace |
|---|---|---|
| AUC (verificación) | 0,868 | ≈ 1,000 |
| EER | 21,5% | 0,25% |
| TAR @ FAR = 1e-3 | 9,4% | 99,6% |
| Top-1 / Top-5 (galería de 15 personas) | 65,8% / 94,7% | 100% / 100% |
| Precision / Recall / F1 macro | 0,75 / 0,41 / 0,48 | 0,996 / 0,999 / 0,998 |
| Desconocidos rechazados | 87,3% | 99,1% |

**Justificación:** ArcFace separa casi perfectamente a personas nuevas, que es el escenario real de la app. La CNN propia generaliza parcialmente (AUC 0,87), pero sus distribuciones de similitud se superponen: con un umbral que evite confundir personas rechaza a la mayoría de las registradas. Las causas son la cantidad de identidades de entrenamiento (85 contra ~600.000), la pérdida softmax sin margen y la menor capacidad. Trade-offs: ArcFace es más pesado y lento, y su licencia es **no comercial / investigación**.

**Umbral:** `0.243` (coseno), el menor umbral con FAR ≤ 1e-3 en los pares de **validación**. En test open-set da FAR 0,02% y TAR 99,1%.

**Análisis de errores:** los falsos negativos de ArcFace son casos extremos (maquillaje de personaje, décadas de diferencia de edad, oclusiones grandes), y sus falsos positivos son parecidos reales con similitud baja (≈ 0,3). La CNN, en cambio, confunde personas por el color de pelo y el tono de la foto, con similitudes altas (≈ 0,8).

## Dataset

Detalle completo en `1-pre-processing.ipynb`.

- **Fuente:** Pins Face Recognition (Kaggle, `hereisburak/pins-face-recognition`), licencia CC0-1.0. Son fotos de celebridades recopiladas de Pinterest; se usan solo con fines académicos y **no se suben al repositorio**.
- **Tamaño:** 105 identidades y 17.534 imágenes (100% `.jpg`, 0 corruptas), entre 86 y 237 por persona (media 167): **desbalance moderado (2,8x)**, compensado con pesos por clase.
- **Variaciones:** pose, edad, caracterizaciones, maquillaje, oclusiones (anteojos, manos) e iluminación; 7,3% en blanco y negro.
- **Reglas de calidad:** sin rostro (6), más de un rostro ambiguo (1), rostro < 40 px (0), borrosas por varianza del Laplaciano < 30 (24), casi-duplicados con similitud ArcFace > 0,95 (609) y posibles etiquetas erróneas detectadas por clustering (6). Quedan **16.888 imágenes**.
- **Splits:** 85 identidades closed-set (80/10/10 estratificado: train 11.094, val 1.387, test 1.387) y 20 identidades **open-set** solo para test (3.020), sin fuga de datos verificada.
- **Sesgos:** mayoría de celebridades occidentales, de piel clara y adultos jóvenes, con fotos profesionales y retocadas: es esperable un peor desempeño en grupos subrepresentados.

## Notas importantes

- Para usar `pgvector`, levanten `postgres` y definan `USE_PGVECTOR=true` en `.env`.
- Si cambian `EMBEDDING_DIM`, la tabla `embeddings` se recrea y se borran los registros previos.
- Los modelos (`*.pth`, `*.onnx`) y los datos no se versionan (ver `.gitignore`). Coloquen el modelo entrenado en `models/` y súbanlo a un link **público de solo lectura** para que los docentes puedan descargarlo.
