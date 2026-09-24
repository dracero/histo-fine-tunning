# Plan de Implementación: Respeto Estricto de la Ontología Espacial en el Etiquetado Celular con Gemini Vision

## 1. Contexto y Diagnóstico del Problema

### Síntoma
En el etiquetado celular automatizado con Gemini Vision (`✨ Etiquetar Gemini` y pipeline de autolabeling), el modelo respeta la **ontología textual** (asigna los nombres taxonómicos e histológicos correctos definidos en los PDFs/ontologías, como *Espermatogonia A clara*, *Espermatocito primario*, *Célula de Sertoli*, *Célula de Leydig*), pero **no respeta la ontología espacial**.

### Consecuencia Biológica Inaceptable
- Células de Leydig clasificadas dentro del epitelio germinal del túbulo seminífero (cuando residen exclusivamente en el estroma conectivo intertubular).
- Espermatogonias clasificadas en la luz central del túbulo o en el estroma intersticial (cuando deben estar exclusivamente en la lámina basal).
- Espermatozoides o espermátides tardías clasificadas en la lámina basal (cuando corresponden a la región luminal/adluminal).

### Causas Raíz Identificadas en el Código
1. **Pérdida de Reglas Espaciales en Backend (`backend/main.py`)**:
   - En la línea 1795 se ejecutaba: `spatial_map = ont_doc.get("spatial_map")`.
   - Sin embargo, en el documento de ontología (`datasets/ontologies/arch4.json`), las reglas no están bajo la clave `"spatial_map"`, sino dentro de la lista `"spatial_rules"` y de cada diccionario en `"micro_structures"`.
   - Por tanto, `spatial_map` siempre evaluaba a `None` y ninguna regla de compartimento llegaba al clasificador.
2. **Contexto Celular Ciego al Espacio en Prompting (`backend/gemini_vision.py`)**:
   - Cada célula se reportaba a Gemini como: `Cell #X: in compartment 'unknown'`.
   - El prompt no incluía las reglas negativas de exclusión (`forbidden_in`) ni las posiciones radiales relativas.
   - Los campos `zone`, `parent` y `cytology` de las clases candidatas quedaban vacíos o truncados.
3. **Falta de Detección de Compartimentos y Restricciones Negativas**:
   - El prompt no instruía a Gemini a identificar previamente la compartimentación tisular (túbulo vs intersticio vs luz).
   - No existía prohibición explícita de asignar células de Leydig dentro de los túbulos, ni células germinales en el intersticio.
4. **Ausencia de Validación y Corrección Programática Post-Gemini**:
   - El endpoint `/api/classify-gemini` aceptaba ciegamente la respuesta de Gemini sin someterla a validación topológica con `validate_spatial_rules()`.
5. **Mezcla Macro/Micro en Frontend (`frontend/src/pages/index.astro`)**:
   - Si existían segmentaciones macroestructurales (túbulos, luz, intersticio), el frontend las empaquetaba junto con las células en un solo vector de `detections`, perdiendo la jerarquía espacial contenedor-contenido.

---

## 2. Arquitectura de la Solución

```
+-----------------------------------------------------------------------------+
|                                FRONTEND                                     |
|  - Separa anotaciones macro (túbulos, intersticio) de micro (células)       |
|  - Envía células en 'detections' y macroestructuras en 'macro_annotations'  |
+-----------------------------------------------------------------------------+
                                       |
                                       v
+-----------------------------------------------------------------------------+
|                      BACKEND: /api/classify-gemini                          |
|  1. Carga ontología (arch4.json)                                            |
|  2. derive_spatial_map_and_rules():                                         |
|     - spatial_rules_lookup (reglas completas por clase celular)              |
|     - spatial_map (compartimento -> clases permitidas)                      |
|     - forbidden_map (compartimento -> clases terminantemente prohibidas)    |
|  3. Si hay macro_annotations: calcula contención geométrica con OpenCV     |
|     (cv2.pointPolygonTest) -> asigna 'containing_layer' a cada célula       |
+-----------------------------------------------------------------------------+
                                       |
                                       v
+-----------------------------------------------------------------------------+
|                     GEMINI VISION: CLASIFICACIÓN BATCH                      |
|  - Imagen anotada con contornos y números de célula en contexto completo    |
|  - DIRECTIVA DE ONTOLOGÍA ESPACIAL EN PROMPT:                               |
|    * Restricciones anatómicas absolutas (túbulo vs intersticio vs luz)      |
|    * Lista de exclusiones negativas (forbidden_in)                          |
|  - Salida estructurada JSON exigiendo:                                      |
|    {"cell_index": X, "compartment": "...", "class_key": "...", ...}         |
+-----------------------------------------------------------------------------+
                                       |
                                       v
+-----------------------------------------------------------------------------+
|             POST-PROCESAMIENTO: enforce_spatial_rules_on_detections         |
|  - Valida coherencia entre compartimento detectado y clase asignada         |
|  - Si hay violación biológica (ej: Leydig en túbulo):                       |
|    * Reasigna automáticamente a la clase permitida más afín en ese estrato  |
|    * Marca 'spatial_corrected' = True y documenta la corrección             |
+-----------------------------------------------------------------------------+
                                       |
                                       v
+-----------------------------------------------------------------------------+
|                       RESPUESTA AL FRONTEND / UI                            |
|  - Detecciones celulares 100% compatibles con la ontología espacial         |
|  - Preserva macroestructuras intactas en el visor                           |
|  - Tooltip en hover muestra: "💡 [compartimento] justificación citológica"  |
+-----------------------------------------------------------------------------+
```

---

## 3. Plan Detallado de Modificaciones

### Módulo 1: `backend/pdf_ontology.py`
1. **Nueva función `derive_spatial_map_and_rules(ontology_doc)`**:
   - Parsea exhaustivamente `spatial_rules`, `micro_structures` y `macro_structures` del documento JSON.
   - Genera:
     - `spatial_rules_lookup`: micro_key -> `{compartment, parent_macro, forbidden_in, relative_radial_position, rule_description}`.
     - `spatial_map`: compartimento o macroestructura -> lista de `micro_keys` biológicamente válidas.
     - `forbidden_map`: compartimento o macroestructura -> lista de `micro_keys` prohibidas.
   - Provee fallbacks robustos para testículo e histología general en caso de ontologías parciales.
2. **Nueva función `enforce_spatial_rules_on_detections(...)`**:
   - Comprueba para cada célula si su `class_key` está prohibida en su compartimento (`containing_layer` o `compartment`).
   - Si existe incompatibilidad (ej. `celula_leydig` en `tubulo_seminifero` o `espermatogonia` en `espacio_intersticial` o `luz_tubular`), reasigna automáticamente a una clase permitida acorde al estrato (basal, intermedio, adluminal o luminal) y marca `spatial_corrected = True`.

### Módulo 2: `backend/gemini_vision.py`
1. **Actualización de `classify_cells_batch_gemini()`**:
   - Aceptar `spatial_rules_lookup`, `spatial_map`, `forbidden_map`, y `macro_annotations`.
   - Poblar los metadatos de cada clase en el prompt con `parent`, `zone`, `forbidden_in` y descripción citológica extraída de la ontología.
   - Inyectar el bloque estructurado:
     ```
     ==================================================================
     REGLAS ESTRICTAS DE ONTOLOGÍA ESPACIAL Y DISTRIBUCIÓN TOPOLÓGICA:
     1. ESPACIO INTERSTICIAL (estroma conectivo intertubular):
        - PERMITIDAS: celula_leydig, celula_peritubular.
        - ESTRICTAMENTE PROHIBIDAS: espermatogonia_*, espermatocito_*, espermatide_*, espermatozoide, celula_sertoli.
        * ¡Las células germinales y de Sertoli NUNCA existen en el estroma intertubular!
     2. TÚBULO SEMINÍFERO (epitelio germinal):
        - ESTRICTAMENTE PROHIBIDA: celula_leydig.
        * Estrato Basal: espermatogonia_a_clara, espermatogonia_a_oscura, espermatogonia_b, celula_sertoli, celula_peritubular.
        * Estrato Intermedio: espermatocito_primario, espermatocito_secundario.
        * Estrato Adluminal: espermatide_temprana, espermatide_tardia.
        * Luz Tubular: espermatozoide, espermatide_tardia.
     ==================================================================
     ```
   - Actualizar el formato JSON exigido a Gemini:
     `{"cell_index": 0, "compartment": "tubulo_basal", "class_key": "espermatogonia_a_clara", "confidence": 0.95, "reasoning": "..."}`.
   - Ejecutar `enforce_spatial_rules_on_detections()` tras recibir la respuesta para garantizar 0% de violaciones.

### Módulo 3: `backend/main.py`
1. **Actualización de `/api/classify-gemini`**:
   - Aceptar `macro_annotations: Optional[str] = Form(None)`.
   - Si `detections` contiene macroestructuras (ej. `is_macro=True` o clave en túbulos/intersticio), separarlas automáticamente.
   - Extraer `spatial_map`, `spatial_rules_lookup` y `forbidden_map` mediante `derive_spatial_map_and_rules(ont_doc)`.
   - Si se proporcionan `macro_annotations`, asociar a cada célula su `containing_layer` mediante test de punto en polígono (`cv2.pointPolygonTest`).
   - Invocar `classify_cells_batch_gemini` pasando todas las directivas espaciales.
   - Devolver el reporte con células clasificadas, conteo de correcciones espaciales y macroestructuras intactas.

### Módulo 4: `frontend/src/pages/index.astro`
1. **Actualización de `runGeminiClassification()`**:
   - Discriminar `macroDets` y `cellDets` a partir de `data.groups`.
   - Enviar únicamente `cellDets` en el payload de células a clasificar y adjuntar `macro_annotations` si están disponibles.
   - Al recibir la respuesta del backend, reensamblar los grupos de células sin destruir ni sobreescribir las macroestructuras existentes.
   - Enriquecer el tooltip flotante para mostrar el compartimento anatómico:
     `💡 [compartimento] justificación citológica`.

---

## 4. Plan de Verificación y Control de Calidad

### A. Pruebas Automatizadas (Scripts de Validación)
1. **Test de Derivación de Reglas**:
   - Cargar `arch4.json` y verificar que `derive_spatial_map_and_rules()` extrae las 11 reglas sin pérdida, mapeando correctamente los 4 macrocompartimentos y sus exclusiones.
2. **Test de Corrección de Violaciones**:
   - Inyectar una detección sintética de `celula_leydig` en compartimento `tubulo_seminifero` y verificar su corrección automática a célula tubular.
   - Inyectar una detección de `espermatogonia` en compartimento `espacio_intersticial` y verificar su corrección a `celula_leydig`.
3. **Test de Sintaxis y Compilación**:
   - Ejecutar `python3 -m py_compile` sobre `pdf_ontology.py`, `gemini_vision.py`, `main.py`.

### B. Verificación Manual en Interfaz de Usuario
1. Cargar imagen histológica testicular con ontología activa `arch4`.
2. Segmentar con Cellpose.
3. Hacer clic en `✨ Etiquetar Gemini`.
4. Verificar que:
   - Las células del estroma intertubular quedan etiquetadas como Célula de Leydig.
   - Las células del epitelio germinal quedan distribuidas concéntricamente (espermatogonias en la base, espermatocitos en el medio, espermátides hacia la luz).
   - Ninguna célula de Leydig aparece dentro del túbulo.
   - Al pasar el cursor por cualquier célula, el tooltip refleja su compartimento anatómico.
