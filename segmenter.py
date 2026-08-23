"""
Segmentación de estructuras histológicas de testículo con SAM 3 (Meta).

Requisitos previos:
  - Haber corrido 00_instalar_sam3.sh
  - Tener acceso aprobado al modelo en https://huggingface.co/facebook/sam3
  - Estar autenticado con `huggingface-cli login`

SAM3 NO reconoce automáticamente clases médicas específicas (no sabe qué es
una "espermatogonia B" per se) -- funciona con PROMPTS DE TEXTO genéricos que
describen la forma/apariencia visual. Hay que probar varias frases por
estructura y quedarte con la que mejor funcione en tus imágenes.

Uso:
    python sam3_histologia_testiculo.py --image mi_imagen.png --prompt "cell nucleus"
"""

import argparse
import os
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


# ------------------------------------------------------------------
# Prompts sugeridos por tipo de estructura (ver README para ajustarlos)
# ------------------------------------------------------------------
PROMPTS_SUGERIDOS = {
    "nucleos_celulas": "cell nucleus",
    "tubulo_completo": "circular tissue structure",
    "arteria_pared": "concentric ring structure",
    "lumen_vaso": "empty circular lumen",
    "eritrocitos": "red blood cells",
}


def cargar_modelo() -> Sam3Processor:
    print("Cargando SAM 3.1 (esto puede tardar la primera vez)...")
    model = build_sam3_image_model(version="sam3.1")
    processor = Sam3Processor(model)
    return processor


def segmentar_con_texto(processor: Sam3Processor, ruta_imagen: str, prompt_texto: str, umbral_score: float = 0.3) -> tuple[Image.Image, list, list, list]:
    """Segmenta todas las instancias que coincidan con el prompt de texto."""
    image = Image.open(ruta_imagen).convert("RGB")
    inference_state = processor.set_image(image)

    output = processor.set_text_prompt(state=inference_state, prompt=prompt_texto)

    masks = output["masks"]     # tensor/array de máscaras binarias
    boxes = output["boxes"]     # bounding boxes [x1,y1,x2,y2]
    scores = output["scores"]   # confianza por instancia

    # Filtrar por score mínimo
    keep = [i for i, s in enumerate(scores) if s >= umbral_score]
    masks = [masks[i] for i in keep]
    boxes = [boxes[i] for i in keep]
    scores = [scores[i] for i in keep]

    print(f"  Prompt: '{prompt_texto}' -> {len(masks)} instancias (score >= {umbral_score})")
    return image, masks, boxes, scores


def visualizar_resultado(image: Image.Image, masks: list, boxes: list, scores: list, prompt_texto: str, ruta_salida: str) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    ax.imshow(image)

    rng = np.random.default_rng(42)
    for mask, box, score in zip(masks, boxes, scores):
        color = rng.random(3)
        mask_np = np.array(mask) if not isinstance(mask, np.ndarray) else mask
        colored_mask = np.zeros((*mask_np.shape, 4))
        colored_mask[mask_np > 0] = (*color, 0.45)
        ax.imshow(colored_mask)

        x1, y1, x2, y2 = box
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=1.5, edgecolor=color, facecolor="none"
        )
        ax.add_patch(rect)
        ax.text(x1, y1 - 4, f"{score:.2f}", color=color, fontsize=8, weight="bold")

    ax.set_title(f"Prompt: \"{prompt_texto}\"  |  {len(masks)} instancias detectadas")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(ruta_salida, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Guardado: {ruta_salida}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Ruta a la imagen de histología")
    parser.add_argument(
        "--prompt", default=None,
        help="Prompt de texto (frase corta en inglés). Si se omite, prueba todos los sugeridos."
    )
    parser.add_argument("--umbral", type=float, default=0.3, help="Score mínimo de confianza")
    parser.add_argument("--outdir", default="resultados_sam3", help="Carpeta de salida")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    processor = cargar_modelo()

    nombre_base = os.path.splitext(os.path.basename(args.image))[0]

    prompts_a_probar = (
        {"custom": args.prompt} if args.prompt else PROMPTS_SUGERIDOS
    )

    for nombre_clase, texto_prompt in prompts_a_probar.items():
        image, masks, boxes, scores = segmentar_con_texto(
            processor, args.image, texto_prompt, args.umbral
        )
        if len(masks) == 0:
            print(f"  (sin resultados para '{texto_prompt}', probá otra frase)")
            continue

        ruta_salida = os.path.join(args.outdir, f"{nombre_base}_{nombre_clase}.png")
        visualizar_resultado(image, masks, boxes, scores, texto_prompt, ruta_salida)


if __name__ == "__main__":
    main()