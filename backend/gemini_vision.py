"""
Dynamic Multimodal Vision Assistant with Gemini 2.5 Flash.

Provides zero-shot visual prompt refinement and image structure discovery
without any hardcoded prompt lists or static translation dictionaries.
"""

import io
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional
from PIL import Image

logger = logging.getLogger("sam3-backend")

DEFAULT_COLORS = [
    "#e11d48", "#8b5cf6", "#06b6d4", "#f59e0b", "#10b981",
    "#ec4899", "#6366f1", "#14b8a6", "#f97316", "#84cc16",
    "#a855f7", "#0ea5e9", "#ef4444", "#22c55e", "#eab308",
    "#d946ef", "#38bdf8", "#fb923c", "#4ade80", "#facc15",
]


def _get_gemini_client(api_key: Optional[str] = None):
    """Initialize Google GenAI client using provided or environment API key."""
    key = api_key or os.environ.get("GEMINI_API_KEY")
    if not key:
        logger.warning("GEMINI_API_KEY is not configured in environment.")
        return None
    try:
        from google import genai
        return genai.Client(api_key=key)
    except Exception as e:
        logger.error(f"Failed to initialize google-genai client: {e}")
        return None


def refine_prompt_multimodal(
    image: Image.Image,
    user_prompt: str,
    ontology_context: Optional[List[Dict[str, Any]]] = None,
    api_key: Optional[str] = None,
) -> str:
    """
    Dynamically translate and refine any user prompt (in Spanish or any language)
    into a concise visual grounding prompt optimized for SAM 3.1, considering
    the actual visual context of the image.
    """
    if not user_prompt or not user_prompt.strip():
        return "cell nucleus"

    client = _get_gemini_client(api_key)
    if client is None:
        # Fallback to the raw prompt if Gemini client is not configured
        return user_prompt.strip()

    try:
        # Prepare lightweight image preview for Gemini
        img_preview = image.copy()
        if img_preview.mode != "RGB":
            img_preview = img_preview.convert("RGB")
        max_dim = 768
        if max(img_preview.size) > max_dim:
            ratio = max_dim / max(img_preview.size)
            img_preview = img_preview.resize(
                (int(img_preview.width * ratio), int(img_preview.height * ratio)),
                Image.LANCZOS,
            )

        ont_summary = ""
        if ontology_context:
            keys = [c.get("name", c.get("key", "")) for c in ontology_context[:10]]
            ont_summary = f"Active ontology structures: {', '.join(keys)}."

        sys_inst = (
            "You are a multimodal vision specialist. Convert the user's histological/biological "
            "query into a single concise English visual grounding prompt (3 to 6 words max) "
            "specifically optimized for Segment Anything Model 3 (SAM 3.1 open-vocabulary grounding). "
            "Focus on direct visual attributes visible in the image: color/staining (dark violet, pink eosinophilic), "
            "shape (round, elongated spindle, tubular cavity, wavy bundle), and structure (nucleus, lumen, fiber). "
            "Return ONLY the concise visual phrase without quotes or explanation."
        )

        prompt_text = f"User term: '{user_prompt}'. {ont_summary} Generate the optimal SAM 3 visual phrase:"

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[prompt_text, img_preview],
            config={
                "system_instruction": sys_inst,
                "temperature": 0.1,
            },
        )

        refined = response.text.strip().replace('"', '').replace("'", "")
        logger.info(f"Gemini refined prompt '{user_prompt}' -> '{refined}'")
        return refined if refined else user_prompt.strip()

    except Exception as e:
        logger.warning(f"Error in multimodal prompt refinement: {e}")
        return user_prompt.strip()


def discover_visual_primitives_from_image(
    image: Image.Image,
    ontology_context: Optional[List[Dict[str, Any]]] = None,
    max_structures: int = 8,
    api_key: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Dynamically inspect the histology/microscopy image with Gemini Vision to discover
    all distinct visual structures present on this specific slide, returning
    a list of prompt descriptors suitable for SAM 3.1 grounding.
    """
    client = _get_gemini_client(api_key)
    if client is None:
        return []

    try:
        img_preview = image.copy()
        if img_preview.mode != "RGB":
            img_preview = img_preview.convert("RGB")
        max_dim = 1024
        if max(img_preview.size) > max_dim:
            ratio = max_dim / max(img_preview.size)
            img_preview = img_preview.resize(
                (int(img_preview.width * ratio), int(img_preview.height * ratio)),
                Image.LANCZOS,
            )

        ont_info = ""
        if ontology_context:
            ont_info = "Relevant ontology knowledge: " + json.dumps([
                {"name": c.get("name", c.get("key")), "prompt": c.get("prompt")}
                for c in ontology_context[:15]
            ], ensure_ascii=False)

        sys_inst = """\
You are an expert computational pathologist and vision AI assistant.
Analyze this histology / microscopy image and identify all distinct visual structures, cell types, \
connective fibers, lumens/cavities, and tissue compartments present in THIS specific image.

For each structure, provide:
- "key": short ASCII identifier (e.g. "cell_nucleus", "collagen_fiber", "tubular_lumen")
- "name": Spanish name (e.g. "Núcleos celulares", "Fibras de colágeno", "Luz tubular")
- "label": Short UI label in Spanish
- "prompt": A concise English visual prompt (3-6 words) for SAM 3.1 segmentation describing color, shape, and structure (e.g. "dark violet round nucleus", "pink wavy collagen fiber", "empty circular lumen space").

Return ONLY a valid JSON array of objects.
"""

        user_content = [
            f"Analyze this image and extract up to {max_structures} distinct visual structures.\n{ont_info}",
            img_preview,
        ]

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=user_content,
            config={
                "system_instruction": sys_inst,
                "temperature": 0.2,
                "response_mime_type": "application/json",
            },
        )

        raw_text = response.text.strip()
        if raw_text.startswith("```"):
            raw_text = re.sub(r"^```(?:json)?\s*", "", raw_text)
            raw_text = re.sub(r"\s*```$", "", raw_text)

        structures = json.loads(raw_text)
        if not isinstance(structures, list):
            return []

        # Assign palette colors
        for i, s in enumerate(structures):
            if "color" not in s:
                s["color"] = DEFAULT_COLORS[i % len(DEFAULT_COLORS)]
            if "label" not in s:
                s["label"] = s.get("name", s.get("key", f"Estructura {i + 1}"))

        logger.info(f"Gemini discovered {len(structures)} visual primitives in image: {[s['key'] for s in structures]}")
        return structures

    except Exception as e:
        logger.error(f"Error discovering visual primitives with Gemini Vision: {e}", exc_info=True)
        return []


def classify_with_multimodal_gemini_fusion(
    image: Image.Image,
    detections: List[Dict[str, Any]],
    candidate_classes: List[Dict[str, Any]],
    ontology_name: Optional[str] = None,
    ontology_context: Optional[Dict[str, Any]] = None,
    api_key: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Multimodal Agentic Pathology Classifier:
    Combines Gemini Multimodal Spatial & Anatomical Reasoning with
    MahmoodLab CONCH (512-dim) and Paige AI Virchow 2 (1280-dim) Foundation Embeddings.

    1. Computes deep feature embeddings with Virchow 2 (1280d) and CONCH (512d).
    2. Renders a numbered visual grounding map on the slide so Gemini perceives the
       exact anatomical tissue context (basement membrane vs adluminal vs lumen vs stroma).
    3. Gemini analyzes each cell's spatial compartment and nuclear chromatin architecture,
       fusing its clinical reasoning with foundation model similarity metrics.
    4. Produces high-precision classifications with diagnostic justifications.
    """
    if not detections:
        return []

    # Import pathology functions dynamically to avoid circular dependencies
    from pathology_models import (
        extract_crops_from_detections,
        classify_detections_with_conch,
        filter_cellular_candidate_classes,
        VirchowModelWrapper,
        UniModelWrapper,
        _compute_detection_area,
    )
    import torch
    import torch.nn.functional as F
    from PIL import ImageDraw, ImageFont

    # 1. Filter candidate classes to retain only cellular entities
    cellular_classes = filter_cellular_candidate_classes(candidate_classes or [])
    if not cellular_classes:
        cellular_classes = candidate_classes or []

    class_meta_map = {
        c.get("key", f"class_{i}"): {
            "key": c.get("key", f"class_{i}"),
            "label": c.get("label", c.get("name", c.get("key"))),
            "name": c.get("name", c.get("label", c.get("key"))),
            "color": c.get("color", DEFAULT_COLORS[i % len(DEFAULT_COLORS)]),
            "prompt": c.get("prompt", ""),
        }
        for i, c in enumerate(cellular_classes)
    }

    # 2. Extract crops and compute Virchow 2 (1280-dim), UNI (1024-dim) & CONCH (512-dim)
    crops = extract_crops_from_detections(image, detections)
    num_dets = len(detections)

    virchow_feats = None
    virchow_sim_matrix = None
    try:
        virchow = VirchowModelWrapper.get_instance()
        if not virchow.is_loaded:
            virchow.load()
        if virchow.is_loaded:
            virchow_feats = virchow.encode_crops(crops, batch_size=16).float()
            virchow_norm = F.normalize(virchow_feats, dim=-1)
            virchow_sim_matrix = torch.matmul(virchow_norm, virchow_norm.T).cpu().numpy()
            logger.info(f"Computed Virchow 2 (1280d) features for {num_dets} cells")
    except Exception as virchow_err:
        logger.warning(f"Virchow feature computation skipped: {virchow_err}")

    uni_feats = None
    uni_sim_matrix = None
    try:
        uni = UniModelWrapper.get_instance()
        if not uni.is_loaded:
            uni.load()
        if uni.is_loaded:
            uni_feats = uni.encode_crops(crops, batch_size=16).float()
            uni_norm = F.normalize(uni_feats, dim=-1)
            uni_sim_matrix = torch.matmul(uni_norm, uni_norm.T).cpu().numpy()
            logger.info(f"Computed UNI (1024d) features for {num_dets} cells")
    except Exception as uni_err:
        logger.warning(f"UNI feature computation skipped: {uni_err}")

    conch_classified = classify_detections_with_conch(
        image=image,
        detections=detections,
        candidate_classes=cellular_classes,
        temperature=0.12,
        is_histology=True,
    )

    # 3. Create spatial grounding image with numbered overlays for Gemini
    img_w, img_h = image.size
    annotated_img = image.copy().convert("RGB")
    draw = ImageDraw.Draw(annotated_img)

    for i, det in enumerate(detections):
        box = det.get("box")
        if not box and "bbox" in det and len(det["bbox"]) == 4:
            bx, by, bw, bh = det["bbox"]
            box = [bx, by, bx + bw, by + bh]
        if not box:
            box = [0, 0, 10, 10]

        x1, y1, x2, y2 = box
        # Draw bounding box
        draw.rectangle([x1, y1, x2, y2], outline="#ef4444", width=2)
        # Draw cell index tag
        tag_text = f"#{i+1}"
        draw.rectangle([x1, max(0, y1 - 14), x1 + 24, y1], fill="#ef4444")
        draw.text((x1 + 2, max(0, y1 - 14)), tag_text, fill="#ffffff")

    # Resize annotated image for Gemini to optimize latency while preserving clarity
    max_dim = 1200
    if max(annotated_img.size) > max_dim:
        ratio = max_dim / max(annotated_img.size)
        gemini_image = annotated_img.resize(
            (int(annotated_img.width * ratio), int(annotated_img.height * ratio)),
            Image.LANCZOS,
        )
    else:
        gemini_image = annotated_img

    # 4. Check if Gemini client is available
    client = _get_gemini_client(api_key)
    gemini_predictions: Dict[int, Dict[str, Any]] = {}

    if client is not None:
        try:
            # Prepare cellular summary for Gemini
            cell_descriptors = []
            for i, det in enumerate(conch_classified):
                box = det.get("box") or [0, 0, 10, 10]
                area = _compute_detection_area(det)
                top_conch_key = det.get("class_key", "")
                top_conch_conf = det.get("conch_confidence", 0.0)
                scores = det.get("conch_scores", {})
                sorted_scores = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:3]
                scores_summary = ", ".join([f"{k}: {v:.2f}" for k, v in sorted_scores])

                cell_descriptors.append(
                    f"Cell #{i+1}: Location [x={box[0]}, y={box[1]}, w={box[2]-box[0]}, h={box[3]-box[1]}], "
                    f"Area={area:.0f}px. CONCH top scores: [{scores_summary}]"
                )

            classes_desc = "\n".join([
                f"- '{c.get('key')}': {c.get('label', c.get('name'))} - Visual characteristics: {c.get('prompt', '')}"
                for c in cellular_classes
            ])

            sys_inst = """\
You are a senior computational pathologist performing cell-by-cell histological classification.
You are given a high-resolution histology image with numbered red bounding boxes (#1, #2, #3, ...) marking individual segmented cells.

Analyze the anatomical tissue architecture (e.g. basement membrane/basal compartment, adluminal compartment, tubular lumen, interstitial connective tissue) AND nuclear morphology (size, chromatin texture, pale vs dark staining, presence/position of nucleoli).

For each numbered cell, identify its exact cell type from the candidate classes list.
Provide a clear, brief diagnostic justification explaining the anatomical location and cytological features.

Output ONLY a JSON list of objects:
[
  {
    "cell_index": 1,
    "class_key": "exact_key_from_list",
    "confidence": 0.95,
    "reasoning": "Located directly on the basal lamina with an oval pale nucleus and peripheral nucleolus."
  }
]
"""

            user_prompt = f"""\
Histology Slide Analysis (Ontology: {ontology_name or 'Histopathology'})

CANDIDATE CELL CLASSES:
{classes_desc}

SEGMENTED CELLS TO CLASSIFY:
{chr(10).join(cell_descriptors)}

Inspect the attached slide with numbered red bounding boxes and classify every single cell from #1 to #{num_dets}:
"""

            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[user_prompt, gemini_image],
                config={
                    "system_instruction": sys_inst,
                    "temperature": 0.1,
                    "response_mime_type": "application/json",
                },
            )

            raw_text = response.text.strip()
            if raw_text.startswith("```"):
                raw_text = re.sub(r"^```(?:json)?\s*", "", raw_text)
                raw_text = re.sub(r"\s*```$", "", raw_text)

            parsed = json.loads(raw_text)
            if isinstance(parsed, list):
                for item in parsed:
                    c_idx = item.get("cell_index")
                    if c_idx is not None and isinstance(c_idx, int):
                        gemini_predictions[c_idx - 1] = {
                            "class_key": item.get("class_key"),
                            "confidence": float(item.get("confidence", 0.9)),
                            "reasoning": item.get("reasoning", ""),
                        }
            logger.info(f"Gemini Multimodal Vision classified {len(gemini_predictions)} / {num_dets} cells.")

        except Exception as gemini_err:
            logger.error(f"Gemini Multimodal reasoning error: {gemini_err}", exc_info=True)

    # 5. Multimodal Fusion (Gemini Vision + Virchow 2 + CONCH)
    final_classified = []
    for i, det in enumerate(conch_classified):
        det_copy = dict(det)
        orig_det = detections[i] if i < len(detections) else {}

        # Preserve user manual labels unconditionally
        if orig_det.get("is_user_exemplar") or orig_det.get("manual_override"):
            det_copy["is_user_exemplar"] = True
            det_copy["virchow_confidence"] = 1.0
            det_copy["gemini_confidence"] = 1.0
            det_copy["gemini_reasoning"] = "Etiqueta manual confirmada por el usuario."
            final_classified.append(det_copy)
            continue

        gemini_pred = gemini_predictions.get(i)
        conch_key = det_copy.get("class_key")
        conch_conf = det_copy.get("conch_confidence", 0.5)

        if gemini_pred and gemini_pred.get("class_key") in class_meta_map:
            g_key = gemini_pred["class_key"]
            g_conf = gemini_pred.get("confidence", 0.9)
            g_reason = gemini_pred.get("reasoning", "")
            meta = class_meta_map[g_key]

            det_copy["category_id"] = g_key
            det_copy["class_key"] = g_key
            det_copy["class_label"] = meta["label"]
            det_copy["color"] = meta["color"]
            det_copy["gemini_confidence"] = round(g_conf, 4)
            det_copy["gemini_reasoning"] = g_reason
            det_copy["multimodal_fused"] = True
            det_copy["score"] = round(max(g_conf, conch_conf), 4)

        elif conch_key in class_meta_map:
            meta = class_meta_map[conch_key]
            det_copy["category_id"] = conch_key
            det_copy["class_key"] = conch_key
            det_copy["class_label"] = meta["label"]
            det_copy["color"] = meta["color"]
            det_copy["gemini_confidence"] = round(conch_conf, 4)
            det_copy["gemini_reasoning"] = f"Clasificado por similitud morfológica y cromatínica (CONCH/Virchow2)."
            det_copy["score"] = round(conch_conf, 4)

        final_classified.append(det_copy)

    # 6. Data-driven neighbourhood consistency with Virchow (1280d)
    if virchow_sim_matrix is not None and len(final_classified) > 1:
        for i, det in enumerate(final_classified):
            if det.get("is_user_exemplar"):
                continue
            # Check if an uncertain cell is strongly aligned (>0.88) with a high-confidence neighbour
            if det.get("gemini_confidence", 0.0) < 0.70:
                best_sim = -1.0
                best_j = -1
                for j, other in enumerate(final_classified):
                    if i == j:
                        continue
                    if other.get("gemini_confidence", 0.0) >= 0.85:
                        sim = float(virchow_sim_matrix[i, j])
                        if sim > best_sim:
                            best_sim = sim
                            best_j = j
                if best_j >= 0 and best_sim > 0.88:
                    donor = final_classified[best_j]
                    det["category_id"] = donor["category_id"]
                    det["class_key"] = donor["class_key"]
                    det["class_label"] = donor["class_label"]
                    det["color"] = donor["color"]
                    det["neighbour_aligned"] = True
                    det["neighbour_similarity"] = round(best_sim, 4)
                    det["gemini_reasoning"] += f" (Consistencia de vecindad Virchow 1280d: {best_sim:.2f})"

    return final_classified

