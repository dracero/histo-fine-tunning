#!/usr/bin/env python3
"""
Segmentación Semántica Zero-Shot con SAM 3 (Segment Anything Model 3) usando Ultralytics.

Permite segmentar cualquier imagen seleccionando elementos/conceptos en lenguaje natural
sin necesidad de entrenamiento (open-vocabulary zero-shot).

Uso interactivo:
    python segmenter.py

Uso con argumentos CLI:
    python segmenter.py --image mi_imagen.jpg --elements "person, glasses, cell, nucleus" --conf 0.25
"""

import os
import sys
import argparse
import shutil
from typing import List, Optional
import cv2
import matplotlib.pyplot as plt
import numpy as np

try:
    from ultralytics.models.sam import SAM3SemanticPredictor
    from ultralytics.utils.plotting import Annotator, colors
except ImportError:
    print("❌ Error: 'ultralytics' no está instalado. Instalalo ejecutando: uv pip install ultralytics")
    sys.exit(1)


def obtener_modelo_sam3(model_path: str = "sam3.pt") -> str:
    """Verifica la existencia de los pesos de SAM 3 o los descarga desde Hugging Face."""
    if os.path.exists(model_path):
        return model_path

    # Verificar si está en la caché de Hugging Face
    hf_cache_pattern = os.path.expanduser("~/.cache/huggingface/hub/models--facebook--sam3/snapshots")
    if os.path.exists(hf_cache_pattern):
        for root, _, files in os.walk(hf_cache_pattern):
            if "sam3.pt" in files:
                found_path = os.path.join(root, "sam3.pt")
                try:
                    os.symlink(found_path, model_path)
                    print(f"🔗 Enlazado checkpoint desde caché: {found_path} -> {model_path}")
                    return model_path
                except Exception:
                    return found_path

    print(f"⚠️ No se encontró '{model_path}'. Descargando desde Hugging Face (facebook/sam3)...")
    try:
        from huggingface_hub import hf_hub_download
        from getpass import getpass

        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
        if not hf_token:
            hf_token = getpass("Pegá tu token de Hugging Face (con acceso a facebook/sam3): ")

        downloaded_path = hf_hub_download(repo_id="facebook/sam3", filename="sam3.pt", token=hf_token)
        shutil.copy(downloaded_path, model_path)
        print(f"✅ Pesos descargados exitosamente en: {model_path}")
        return model_path
    except Exception as e:
        print(f"❌ Error al descargar pesos de SAM 3: {e}")
        sys.exit(1)


def segmentar_imagen(
    image_path: str,
    elements: List[str],
    model_path: str = "sam3.pt",
    conf: float = 0.25,
    outdir: str = "resultados_sam3",
    show_plot: bool = False,
) -> tuple[np.ndarray, list]:
    """
    Ejecuta la segmentación semántica zero-shot sobre la imagen para los elementos especificados.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"No se encontró la imagen en la ruta: '{image_path}'")

    model_file = obtener_modelo_sam3(model_path)
    os.makedirs(outdir, exist_ok=True)

    print(f"\n🚀 Inicializando SAM3SemanticPredictor con modelo: {model_file}...")
    overrides = {
        "conf": conf,
        "task": "segment",
        "mode": "predict",
        "model": model_file,
        "quantize": 16,  # FP16 para inferencia optimizada en GPU
        "save": False,
    }
    predictor = SAM3SemanticPredictor(overrides=overrides)

    print(f"🖼️ Cargando imagen: {image_path}")
    predictor.set_image(image_path)

    print(f"🔍 Segmentando elementos: {elements}")
    results = predictor(text=elements)

    im_bgr = cv2.imread(image_path)
    if im_bgr is None:
        raise ValueError(f"No se pudo leer la imagen con OpenCV: {image_path}")

    annotator = Annotator(im_bgr.copy(), pil=False, line_width=2)
    total_instancias = 0
    detections_summary = []

    for r in results:
        masks = r.masks
        boxes = getattr(r, "boxes", None)
        names = getattr(r, "names", elements)

        if masks is None or len(masks) == 0:
            continue

        mask_data = masks.data.cpu().numpy()
        num_masks = len(mask_data)
        total_instancias += num_masks

        # Extraer información de clases y cajas
        cls_indices = boxes.cls.cpu().numpy().astype(int) if boxes is not None and hasattr(boxes, "cls") and boxes.cls is not None else [0] * num_masks
        confs = boxes.conf.cpu().numpy() if boxes is not None and hasattr(boxes, "conf") and boxes.conf is not None else [conf] * num_masks
        xyxy = boxes.xyxy.cpu().numpy() if boxes is not None and hasattr(boxes, "xyxy") and boxes.xyxy is not None else None

        for idx, (m, cls_idx, score) in enumerate(zip(mask_data, cls_indices, confs)):
            cls_name = names[cls_idx] if (isinstance(names, list) and cls_idx < len(names)) else (names.get(cls_idx, str(cls_idx)) if isinstance(names, dict) else str(cls_idx))
            color_box = colors(cls_idx, True)

            # Dibujar máscara
            annotator.masks(np.expand_dims(m, 0), [color_box])

            # Dibujar bounding box y etiqueta con el concepto
            if xyxy is not None and idx < len(xyxy):
                box = xyxy[idx]
                label = f"{cls_name} {score:.2f}"
                annotator.box_label(box, label, color=color_box)

            detections_summary.append({
                "concept": cls_name,
                "confidence": float(score),
                "box": xyxy[idx].tolist() if xyxy is not None and idx < len(xyxy) else None
            })

    im_result = annotator.result()

    # Guardar imagen anotada
    nombre_base = os.path.splitext(os.path.basename(image_path))[0]
    out_filename = f"{nombre_base}_segmentado_sam3.png"
    out_filepath = os.path.join(outdir, out_filename)
    cv2.imwrite(out_filepath, im_result)

    print(f"\n✅ Segmentación completada:")
    print(f"   • Instancias totales detectadas: {total_instancias}")
    print(f"   • Imagen guardada en: {out_filepath}")

    if show_plot:
        im_rgb = cv2.cvtColor(im_result, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(12, 10))
        plt.imshow(im_rgb)
        plt.axis("off")
        plt.title(f"SAM 3 Segmentación | Conceptos: {', '.join(elements)} ({total_instancias} detectados)")
        plt.tight_layout()
        plt.show()

    return im_result, detections_summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Segmentación semántica zero-shot con SAM 3 (Ultralytics) por selección de imagen y conceptos."
    )
    parser.add_argument("--image", type=str, default=None, help="Ruta de la imagen a segmentar")
    parser.add_argument(
        "--elements", "--prompt", "--text",
        dest="elements",
        type=str,
        default=None,
        help="Elementos a segmentar separados por comas (ej: 'persona, anteojos' o 'cell, nucleus, lumen')"
    )
    parser.add_argument("--conf", type=float, default=0.25, help="Umbral mínimo de confianza (por defecto: 0.25)")
    parser.add_argument("--model", type=str, default="sam3.pt", help="Ruta al modelo SAM 3 (por defecto: sam3.pt)")
    parser.add_argument("--outdir", type=str, default="resultados_sam3", help="Directorio para guardar los resultados")
    parser.add_argument("--show", action="store_true", help="Mostrar ventana con la imagen resultante")

    args = parser.parse_args()

    # Modo interactivo si no se pasaron argumentos
    image_path = args.image
    if not image_path:
        print("=" * 65)
        print("🎯 SAM 3 — Segmentación Semántica Zero-Shot Interactiva")
        print("=" * 65)
        while not image_path:
            inp = input("👉 Ingrese la ruta de la imagen a segmentar: ").strip().strip("'\"")
            if os.path.exists(inp):
                image_path = inp
            else:
                print(f"❌ El archivo '{inp}' no existe. Intente nuevamente.")

    elements_str = args.elements
    if not elements_str:
        print("\n📝 Indique qué elementos de la imagen desea segmentar.")
        print("   (Puede ingresar uno o varios conceptos separados por comas)")
        print("   Ejemplo dominio general: person, glasses, backpack, dog, car")
        print("   Ejemplo histología/médico: cell nucleus, circular lumen, red blood cell, membrane")
        while not elements_str:
            elements_str = input("👉 Elementos a segmentar: ").strip()

    # Parsear lista de elementos
    elements = [e.strip() for e in elements_str.split(",") if e.strip()]
    if not elements:
        print("❌ No se especificaron elementos válidos.")
        sys.exit(1)

    segmentar_imagen(
        image_path=image_path,
        elements=elements,
        model_path=args.model,
        conf=args.conf,
        outdir=args.outdir,
        show_plot=args.show or ("--image" not in sys.argv),
    )


if __name__ == "__main__":
    main()