# SAM 3 Histology Annotator & Roboflow Pipeline 🚀

Plataforma integral de **segmentación automática**, **edición e inspección fina de anotaciones** y **pipeline de entrenamiento en Roboflow**, utilizando **SAM 3 (Segment Anything Model 3 de Meta AI)**.

Diseñada especialmente para imágenes complejas (como histología o microscopía) que contienen cientos de células o estructuras por imagen, permitiendo segmentarlas automáticamente en segundos, curar/corregir clases o falsos positivos individualmente y exportar o entrenar modelos en Roboflow.

---

## 🌟 Características Principales

### 🧠 1. Segmentación Automática con SAM 3.1 (Meta AI)
- **Multi-Prompt Automático**: Ejecuta automáticamente una suite de prompts visuales universales (*cell or nucleus, elongated dark nucleus, circular tissue structure, object, etc.*) o personalizados basados en ontologías biológicas.
- **Extracción de Polígonos de Alta Precisión**: Convierte las máscaras binarias generadas por SAM 3 en contornos poligonales (formato COCO) simplificados mediante OpenCV (`findContours` + `approxPolyDP`).
- **Ajuste de Umbral en Tiempo Real**: Slider para filtrar la confianza de las detecciones al instante sin re-ejecutar el modelo.

### 🔬 2. Clasificación Zero-Shot con CONCH y Embeddings UNI (MahmoodLab)
- **CONCH (512-d Vision-Language)**: Clasificador histológico zero-shot que analiza los recortes (*crops*) de las células segmentadas por SAM 3 y las asigna automáticamente a las clases biológicas más afines de la ontología.
- **UNI (1024-d ViT-Large)**: Extracción de vectores morfológicos densos de alta dimensionalidad para cada objeto segmentado.

### 📄 3. Extracción de Ontología y CRUD de Imágenes de PDFs
- **Extracción Inteligente**: Extrae texto e imágenes embebidas de papers o libros en PDF vía `pymupdf`.
- **Generación de Ontologías con Gemini**: Diseña estructuras jerárquicas con nombres canónicos y prompts visuales optimizados para segmentación.
- **CRUD Completo de Imágenes Extraídas**:
  - **Visualización en Galería y Lightbox**: Zoom de alta resolución, dimensiones y página de origen.
  - **Edición de Captions**: Actualización y persistencia de pies de figura.
  - **Eliminación y Depuración**: Borrado de esquemas o figuras no deseadas antes de anotar.
  - **Carga de Imágenes**: Posibilidad de adjuntar imágenes histológicas adicionales al conjunto.
- **Segmentación en Lote**: Botón para procesar automáticamente todas las imágenes del PDF con SAM 3.1 + CONCH.

### 🖼️ 4. Soporte Multi-Imagen
- Carga de múltiples imágenes (Drag & Drop o selector).
- Galería con estado en tiempo real (🔴 Pendiente / 🟢 Anotada).
- Preservación independiente del estado de anotación y correcciones por cada imagen.

### ✂️ 5. Edición Fina e Individual de Detecciones
- **Selección Individual y Múltiple**: Click sobre cualquier celda/polígono para seleccionarla, `Shift + Click` para selección múltiple, o `Ctrl + A` para seleccionar todas las detecciones visibles.
- **Reasignación de Clases**: Menú contextual con click derecho o dropdown en la barra superior para mover detecciones individuales o en lote entre clases.
- **Eliminación Rápida**: Tecla `Delete` / `Backspace` o botón en pantalla para borrar falsos positivos o elementos no deseados.

### 🏷️ 6. Gestión Completa de Clases
- **Renombrado Inline**: Edita el nombre genérico (*Clase 1*, *Clase 2*) a nombres semánticos (*Espermatogonia B*, *Célula de Sertoli*, etc.).
- **Selector de Color**: Color picker por clase para ajustar el tono de visualización.
- **Visibilidad Toggle**: Muestra u oculta clases específicas para facilitar el trabajo en zonas muy pobladas.
- **Creación y Eliminación de Clases**: Agrega clases personalizadas vacías o elimina clases enteras.

### 🚀 7. Exportación e Integración con Roboflow
- **Exportación Local COCO JSON**: Descarga directa de anotaciones consolidadas (imágenes, categorías, bboxes y polígonos `segmentation`).
- **Subida Directa a Roboflow**: Envío de imágenes y dataset anotado a tu workspace y proyecto de Roboflow vía API.
- **Disparo de Entrenamiento**: Lanza el entrenamiento de tu modelo (ej. YOLOv8) en Roboflow con un solo click desde la interfaz web.

---

## 🏗️ Estructura del Proyecto

```text
Meta_SAM_V3/
├── backend/
│   ├── main.py                  # API FastAPI para inferencia de SAM 3 (Ultralytics + Meta), CONCH, UNI
│   ├── pathology_models.py      # Módulo de modelos fundacionales (CONCH 512d + UNI 1024d)
│   ├── pdf_ontology.py          # Extracción de PDFs, ontología con Gemini y CRUD de imágenes
│   ├── roboflow_integration.py  # Módulo de conversión COCO, upload y training en Roboflow
│   ├── test_pathology_models.py # Tests unitarios de CONCH, UNI y CRUD de imágenes
│   └── test_pdf_ontology.py     # Tests unitarios de ontología
├── frontend/
│   ├── src/
│   │   └── pages/
│   │       └── index.astro      # Interfaz de usuario interactiva (Astro + Vanilla JS + CSS)
│   ├── package.json
│   └── astro.config.mjs
├── datasets/
│   ├── ontologies/              # Almacenamiento JSON de ontologías generadas
│   └── pdf_images/              # Imágenes y textos extraídos de PDFs
├── segmenter.py                 # Script CLI / Interactivo de segmentación zero-shot con SAM 3 (Ultralytics)
├── sam3.pt                      # Checkpoint de pesos de SAM 3
├── .env                         # Credenciales (Roboflow, Gemini API Key)
├── .env.example                 # Plantilla de variables de entorno
├── package.json                 # Script principal (npm run dev)
└── start.sh                     # Script Bash de lanzamiento simultáneo de servidores
```

---

## 🎯 Uso de Segmentación Semántica Zero-Shot con SAM 3

### 1. Modo Interactivo por Consola
Permite elegir cualquier imagen y definir qué elementos o conceptos segmentar:
```bash
# Ejecutar en modo interactivo (solicita la imagen y los elementos por pantalla):
python segmenter.py

# O pasar parámetros directamente vía CLI:
python segmenter.py --image mi_imagen.jpg --elements "person, glasses, red tie" --conf 0.25
python segmenter.py --image histologia.png --elements "cell nucleus, circular lumen" --conf 0.20
```

### 2. Desde la Aplicación Web
1. Iniciar los servidores con `./start.sh` o `npm run dev`.
2. Abrir el navegador en `http://localhost:4321`.
3. Arrastrar o seleccionar cualquier imagen en la pestaña **🖼️ Imágenes**.
4. En el panel **🎯 Elementos a segmentar**, escribir los conceptos separados por comas o hacer click en los chips de sugerencias.
5. Hacer click en **⚡ Segmentar Elementos** para segmentar instantáneamente con SAM 3 en GPU y editar/exportar los resultados.


## 📐 Arquitectura, Patrones de Diseño y Complejidad

### 🧩 1. Patrones de Diseño Aplicados

- **Singleton / Resource Holder (Carga Única del Modelo)**:
  - En `backend/main.py` ([main.py](file:///run/media/dracero/DiscoMecanico/AIProjects/Meta_SAM_V3/backend/main.py#L56-L75)), el modelo SAM 3 (`build_sam3_image_model`) y su procesador (`Sam3Processor`) se instancian una sola vez durante el evento de inicio (`lifespan`) de FastAPI y se mantienen residentes globalmente en VRAM. Esto evita la penalización severa de recargar el modelo de varios GB en cada petición HTTP.

- **Multi-Prompt Visual Strategy (Estrategia de Prompting en Cascada)**:
  - Definición de una tabla de estrategias de prompts universales (`AUTO_SEGMENT_PROMPTS` en `backend/main.py`) que ejecutan de forma secuencial descripciones geométricas y biológicas (*"cell or nucleus"*, *"elongated dark nucleus"*, *"circular tissue structure"*, etc.) sobre una misma imagen para lograr la partición y segmentación automática multi-clase sin requerir pre-etiquetado manual.

- **State Pattern / In-Memory Store (Frontend)**:
  - En `frontend/src/pages/index.astro`, se implementa una tienda centralizada en memoria mediante JavaScript `Map` (`imageStore`). Almacena los metadatos, archivos base, dimensiones y grupos de detecciones de cada imagen subida, permitiendo cambios reactivos instantáneos de imagen activa, clases y selección sin persistencia redundante en servidor.

- **Factory Method**:
  - `build_coco_json` y `build_multi_image_coco` en `backend/roboflow_integration.py` ([roboflow_integration.py](file:///run/media/dracero/DiscoMecanico/AIProjects/Meta_SAM_V3/backend/roboflow_integration.py#L158-L298)) encapsulan la construcción dinámica de payloads estructurados en formato estándar COCO Dataset (categorías, imágenes, bounding boxes y polígonos `segmentation`), permitiendo exportación individual o en lote.

- **Facade / Adapter Pattern**:
  - El módulo `roboflow_integration.py` actúa como fachada entre la API REST/UI y los servicios remotos de Roboflow. Oculta la complejidad del SDK de Roboflow, gestionando la resolución dinámica de slugs de proyectos, la subida multipart de imágenes con JSON de anotación, el versionado del dataset y el disparo de entrenamientos remotos.

- **Decoupled Vector Layer Overlay (Capa de Renderizado Vectorial)**:
  - En la interfaz web (`index.astro`), la visualización de la imagen base (`<img>`) está desglosada de la capa vectorial SVG (`<svg viewBox>`). Las modificaciones de umbral de confianza, visibilidad de clase o selección múltiple interactúan con el DOM SVG en tiempo real sin requerir renderizados en Canvas rasterizado ni re-descarga de imágenes.

---

### ⏱️ 2. Controles de Complejidad Temporal (Time Complexity)

- **Caché de Embeddings Visuales en SAM 3 (\(O(1)\) para prompts secundarios)**:
  - `processor.set_image(inf_image)` en `backend/main.py` procesa el Vision Transformer (ViT) de SAM 3 sobre la imagen **una única vez**. 
  - Las consultas consecutivas de la suite multi-prompt ejecutan `reset_all_prompts` y `set_text_prompt`, reutilizando las características visuales extraídas previamente. Esto reduce la complejidad temporal de prompts adicionales de \(O(\text{ViT Backbone Heavy Forward})\) a \(O(\text{Text Encoder} + \text{Cross-Attention})\), acelerando la inferencia multi-clase de segundos a milisegundos por prompt.

- **Filtrado de Umbral en Tiempo Real en Cliente (\(O(N)\) local vs. \(O(\text{Inferencia HTTP})\))**:
  - El backend retorna todas las detecciones candidatas a un umbral bajo (0.01). El frontend almacena las \(N\) detecciones en memoria y aplica filtrado instantáneo al mover el slider de confianza mediante iteración lineal simple (\(O(N)\) en JavaScript). Esto elimina la latencia de red y evita re-ejecutar la inferencia del modelo PyTorch.

- **Simplificación Poligonal Ramer-Douglas-Peucker (\(O(P_{\text{denso}}) \to O(P_{\text{aprox}})\))**:
  - Uso de `cv2.approxPolyDP` en `mask_to_polygons` ([main.py](file:///run/media/dracero/DiscoMecanico/AIProjects/Meta_SAM_V3/backend/main.py#L124-L171)) con tolerancia `epsilon=1.5`. Reduce el número de vértices de los contornos extraídos de miles de píxeles contiguos (\(O(P_{\text{denso}})\)) a listas compactas de 10-30 puntos (\(O(P_{\text{aprox}})\)). Esto acelera drásticamente el renderizado SVG en el cliente, la serialización JSON y la velocidad de transferencia HTTP.

- **Estructuras de Datos con Búsqueda en Tiempo Constante (\(O(1)\))**:
  - Uso intensivo de `Set` (`selectedDetections`, `hiddenCategories`) y `Map` (`imageStore`) en el cliente para operaciones de adición, borrado y verificación de visibilidad/selección en \(O(1)\) sin recorridos de arreglos \(O(N)\).
  - Deduplicación de categorías en COCO mediante búsquedas por llave en diccionarios Python (\(O(1)\)).

---

### 💾 3. Controles de Complejidad Espacial y Optimización VRAM (Space Complexity)

- **Redimensionamiento Controlado para Inferencia (`MAX_INFERENCE_DIM = 1440`)**:
  - En `_prepare_image_for_inference` ([main.py](file:///run/media/dracero/DiscoMecanico/AIProjects/Meta_SAM_V3/backend/main.py#L234-L255)), las imágenes gigapíxel o de alta resolución se redimensionan proporcionalmente a un máximo de 1440px antes de pasar a la red neuronal.
  - Mantiene el consumo espacial de memoria VRAM acotado en \(O(\text{MAX\_DIM}^2)\) en lugar de escalar cuadráticamente con resoluciones arbitrarias del usuario (\(O(W \times H)\)), evitando errores de Out-Of-Memory (OOM) en la GPU. Las coordenadas de detecciones se reescalan linealmente (`scale_x`, `scale_y`) a las dimensiones originales sin pérdida de precisión.

- **Gestión de Segmentos de Memoria CUDA (`PYTORCH_CUDA_ALLOC_CONF`)**:
  - Configuración inicial `os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"` para prevenir la fragmentación de memoria en la VRAM de la GPU (ej. NVIDIA RTX 3060) durante múltiples peticiones consecutivas.

- **Inferencia en Precisión Mixta (`bfloat16` + `torch.inference_mode()`)**:
  - **`torch.bfloat16`**: Reduce en un 50% el espacio ocupado por tensores y mapas de activación en VRAM en comparación con `float32`.
  - **`torch.inference_mode()`**: Inhabilitación total del rastreo del grafo de autograd y gradientes de PyTorch, fijando la complejidad espacial del pase forward en estricto \(O(1)\) respecto a la memoria retenida.

- **Liberación Activa de Caché VRAM (`torch.cuda.empty_cache()`)**:
  - Invocación explícita de `torch.cuda.empty_cache()` al finalizar cada endpoint de inferencia (`/api/segment-auto`, `/api/segment-point`) y en el apagado del servidor (`lifespan`), previniendo fugas de memoria en la GPU.

- **Stream en Memoria y Limpieza de Disco**:
  - Procesamiento de archivos de imagen vía `io.BytesIO` sin persistencia temporal redundante en disco durante las llamadas a la API.
  - Creación de directorios aislados con `tempfile.mkdtemp` para la subida a Roboflow, asegurando la eliminación garantizada mediante `shutil.rmtree` en bloques `try...finally`.

---

## 🛠️ Requisitos Previos

1. **Python**: Version 3.10 o 3.12 con soporte PyTorch y CUDA (GPU recomendada para SAM 3).
2. **Node.js**: Version 18 o superior.
3. **Acceso a SAM 3 en Hugging Face**: Debes tener acceso aprobado en [facebook/sam3](https://huggingface.co/facebook/sam3) y haber ejecutado `huggingface-cli login`.
4. **Cuenta en Roboflow**: API Key y un proyecto configurado en Roboflow (tipo *Instance Segmentation* u *Object Detection*).

---

## ⚙️ Configuración e Instalación

### 1. Variables de Entorno
Crea un archivo `.env` en la raíz del proyecto basándote en `.env.example`:

```env
ROBOFLOW_API_KEY=tu_api_key_privada
ROBOFLOW_WORKSPACE=nombre_de_tu_workspace
ROBOFLOW_PROJECT=nombre_de_tu_proyecto
```

### 2. Entorno Python y Dependencias
Asegúrate de tener configurado tu entorno virtual (ej. `.venv`):

```bash
# Activar entorno virtual
source .venv/bin/activate

# Instalar dependencias requeridas
pip install roboflow python-dotenv opencv-python-headless "numpy>=1.26,<2"
```

### 3. Dependencias del Frontend
```bash
cd frontend
npm install
cd ..
```

---

## 🚀 Ejecución del Proyecto

Puedes iniciar el backend y el frontend simultáneamente ejecutando un único comando desde la raíz del proyecto:

```bash
npm run dev
```

o directamente mediante el script bash:

```bash
bash ./start.sh
```

### Puertos por Defecto:
- **Frontend (Astro)**: [http://localhost:4321](http://localhost:4321)
- **Backend API (FastAPI)**: [http://localhost:8000](http://localhost:8000)

---

## 💻 Flujo de Trabajo Sugerido

1. **Cargar Imágenes**: Abre la app en el navegador ([http://localhost:4321](http://localhost:4321)) y arrastra tus fotos de histología o microscopía.
2. **Segmentación**: SAM 3 procesará la imagen automáticamente extrayendo clases y polígonos.
3. **Renombrar Clases**: Ve a la pestaña **Clases** y renombrá *"Clase 1"*, *"Clase 2"* con sus nombres biológicos reales.
4. **Curar Detecciones**:
   - Haz click en detecciones erróneas y presiona `Delete` para borrarlas.
   - Selecciona detecciones y cámbialas de clase mediante el menú desplegable o click derecho.
5. **Exportar / Entrenar**:
   - Ve a la pestaña **Exportar**.
   - Haz click en **Descargar COCO JSON** para guardar la copia local, o
   - Haz click en **Subir a Roboflow** y luego en **Entrenar Modelo** para iniciar el entrenamiento remoto.

---

## 🛡️ Licencia y Créditos
- **Modelo SAM 3**: Desenvolupado por Meta AI.
- **Integración y UI**: Desarrollado para segmentación e inspección histológica de precisión.
