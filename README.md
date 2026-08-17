# SAM 3 Histology Annotator & Roboflow Pipeline 🚀

Plataforma integral de **segmentación automática**, **edición e inspección fina de anotaciones** y **pipeline de entrenamiento en Roboflow**, utilizando **SAM 3 (Segment Anything Model 3 de Meta AI)**.

Diseñada especialmente para imágenes complejas (como histología o microscopía) que contienen cientos de células o estructuras por imagen, permitiendo segmentarlas automáticamente en segundos, curar/corregir clases o falsos positivos individualmente y exportar o entrenar modelos en Roboflow.

---

## 🌟 Características Principales

### 🧠 1. Segmentación Automática con SAM 3 (Meta AI)
- **Multi-Prompt Automático**: Ejecuta automáticamente una suite de prompts visuales universales (*cell or nucleus, elongated dark nucleus, circular tissue structure, object, etc.*) para detectar cientos de instancias por imagen.
- **Extracción de Polígonos de Alta Precisión**: Convierte las máscaras binarias generadas por SAM 3 en contornos poligonales (formato COCO) simplificados mediante OpenCV (`findContours` + `approxPolyDP`).
- **Ajuste de Umbral en Tiempo Real**: Slider para filtrar la confianza de las detecciones al instante sin re-ejecutar el modelo.

### 🖼️ 2. Soporte Multi-Imagen
- Carga de múltiples imágenes (Drag & Drop o selector).
- Galería con estado en tiempo real (🔴 Pendiente / 🟢 Anotada).
- Preservación independiente del estado de anotación y correcciones por cada imagen.

### ✂️ 3. Edición Fina e Individual de Detecciones
- **Selección Individual y Múltiple**: Click sobre cualquier celda/polígono para seleccionarla, `Shift + Click` para selección múltiple, o `Ctrl + A` para seleccionar todas las detecciones visibles.
- **Reasignación de Clases**: Menú contextual con click derecho o dropdown en la barra superior para mover detecciones individuales o en lote entre clases.
- **Eliminación Rápida**: Tecla `Delete` / `Backspace` o botón en pantalla para borrar falsos positivos o elementos no deseados.

### 🏷️ 4. Gestión Completa de Clases
- **Renombrado Inline**: Edita el nombre genérico (*Clase 1*, *Clase 2*) a nombres semánticos (*Espermatogonia B*, *Célula de Sertoli*, etc.).
- **Selector de Color**: Color picker por clase para ajustar el tono de visualización.
- **Visibilidad Toggle**: Muestra u oculta clases específicas para facilitar el trabajo en zonas muy pobladas.
- **Creación y Eliminación de Clases**: Agrega clases personalizadas vacías o elimina clases enteras.

### 🚀 5. Exportación e Integración con Roboflow
- **Exportación Local COCO JSON**: Descarga directa de anotaciones consolidadas (imágenes, categorías, bboxes y polígonos `segmentation`).
- **Subida Directa a Roboflow**: Envío de imágenes y dataset anotado a tu workspace y proyecto de Roboflow vía API.
- **Disparo de Entrenamiento**: Lanza el entrenamiento de tu modelo (ej. YOLOv8) en Roboflow con un solo click desde la interfaz web.

---

## 🏗️ Estructura del Proyecto

```text
Meta_SAM_V3/
├── backend/
│   ├── main.py                  # API FastAPI para inferencia de SAM 3 y endpoints REST
│   └── roboflow_integration.py  # Módulo de conversión COCO, upload y training en Roboflow
├── frontend/
│   ├── src/
│   │   └── pages/
│   │       └── index.astro      # Interfaz de usuario interactiva (Astro + Vanilla JS + CSS)
│   ├── package.json
│   └── astro.config.mjs
├── .env                         # Credenciales y configuración de Roboflow (Ignorado en git)
├── .env.example                 # Plantilla de variables de entorno
├── package.json                 # Script principal (npm run dev)
└── start.sh                     # Script Bash de lanzamiento simultáneo de servidores
```

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
