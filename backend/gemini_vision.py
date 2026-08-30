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
    Multimodal Agentic Pathology Classifier with Calibrated Probabilities:
    Combines MahmoodLab CONCH (512-dim) and Paige AI Virchow 2 (1280-dim) Foundation Embeddings
    with high-resolution visual crops passed directly to Gemini Multimodal Vision.

    1. Computes deep feature embeddings with CONCH (512d) and Virchow 2 (1280d).
    2. Extracts high-resolution cellular crops with adaptive contextual margins.
    3. Sends ambiguous/borderline crops directly to Gemini Vision for cytological arbitration.
    4. Performs Bayesian/calibrated fusion so LLM output cannot blindly overwrite confident foundation predictions.
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
    crops = extract_crops_from_detections(image, detections, margin_ratio=0.35, min_size=80)
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

    # Combined dual-foundation morphological similarity matrix (Virchow 2 1280d + UNI 1024d)
    morph_sim_matrix = None
    if virchow_sim_matrix is not None and uni_sim_matrix is not None:
        morph_sim_matrix = 0.5 * virchow_sim_matrix + 0.5 * uni_sim_matrix
    elif virchow_sim_matrix is not None:
        morph_sim_matrix = virchow_sim_matrix
    elif uni_sim_matrix is not None:
        morph_sim_matrix = uni_sim_matrix

    # 3. High-Precision CONCH (512d) zero-shot classification with multi-template ensembling
    conch_classified = classify_detections_with_conch(
        image=image,
        detections=detections,
        candidate_classes=cellular_classes,
        temperature=0.08,
        is_histology=True,
    )

    # 4. Multi-modal HD Crop Arbitration with Gemini
    client = _get_gemini_client(api_key)
    gemini_predictions: Dict[int, Dict[str, Any]] = {}

    if client is not None:
        try:
            # Classes description for prompt
            classes_desc = "\n".join([
                f"- '{c.get('key')}': {c.get('label', c.get('name'))} (Prompt: {c.get('prompt', '')})"
                for c in cellular_classes
            ])

            sys_inst = """\
You are a senior computational pathologist performing cell-by-cell cytological analysis on high-resolution image crops.
For each cell crop, examine nuclear shape, chromatin texture (hyperchromatic vs vesicular), nucleoli presence/position, \
cytoplasmic volume, and histological compartment.

Validate or arbitrate the CONCH foundation model prediction.
Output ONLY a valid JSON list of objects:
[
  {
    "cell_index": 1,
    "class_key": "exact_key_from_list",
    "confidence": 0.88,
    "reasoning": "Vesicular chromatin with prominent peripheral nucleolus and basal location."
  }
]
"""
            # Identify cells that benefit from multimodal arbitration (low CONCH margin or low confidence)
            ambiguous_indices = [
                i for i, d in enumerate(conch_classified)
                if float(d.get("conch_margin", 1.0)) < 0.20 or float(d.get("conch_confidence", 1.0)) < 0.50
            ]
            # If small detection set (<= 20 cells), evaluate all of them
            if len(conch_classified) <= 20:
                ambiguous_indices = list(range(len(conch_classified)))
            elif len(ambiguous_indices) > 24:
                # Prioritize the most borderline cells by lowest margin
                ambiguous_indices = sorted(
                    ambiguous_indices,
                    key=lambda idx: float(conch_classified[idx].get("conch_margin", 0.0))
                )[:24]

            logger.info(
                f"Gemini HD Crop Arbitration: Evaluating {len(ambiguous_indices)} ambiguous cells "
                f"(out of {num_dets} total segmented cells)..."
            )

            # Process ambiguous crops in chunks of up to 12
            chunk_size = 12
            for start_pos in range(0, len(ambiguous_indices), chunk_size):
                sub_indices = ambiguous_indices[start_pos : start_pos + chunk_size]

                user_parts: List[Any] = [
                    f"Histology Slide Analysis (Ontology: {ontology_name or 'Histopathology'})\n"
                    f"CANDIDATE CELL CLASSES:\n{classes_desc}\n\n"
                    f"Analyze each of the following {len(sub_indices)} high-resolution cell crops:\n"
                ]

                for local_i, global_i in enumerate(sub_indices):
                    det = conch_classified[global_i]
                    crop_img = crops[global_i]
                    top_k = det.get("class_key", "")
                    top_c = det.get("conch_confidence", 0.0)
                    scores = det.get("conch_scores", {})
                    sorted_scores = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:3]
                    scores_summary = ", ".join([f"{k}: {v:.2f}" for k, v in sorted_scores])

                    user_parts.append(
                        f"\n--- Cell #{global_i + 1} ---\n"
                        f"CONCH Top: '{top_k}' (conf: {top_c:.2f}, margin: {det.get('conch_margin', 0.0):.2f})\n"
                        f"Top scores: [{scores_summary}]\n"
                        f"Crop image for Cell #{global_i + 1}:"
                    )
                    user_parts.append(crop_img)

                user_parts.append(
                    "\nOutput JSON array with classifications for each of the cells listed above:"
                )

                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=user_parts,
                    config={
                        "system_instruction": sys_inst,
                        "temperature": 0.1,
                        "response_mime_type": "application/json",
                    },
                )

                raw_text = (response.text or "").strip()
                if raw_text.startswith("```"):
                    raw_text = re.sub(r"^```(?:json)?\s*", "", raw_text)
                    raw_text = re.sub(r"\s*```$", "", raw_text)

                parsed = json.loads(raw_text)
                if isinstance(parsed, list):
                    for item in parsed:
                        c_idx = item.get("cell_index")
                        if c_idx is not None and isinstance(c_idx, int) and 1 <= c_idx <= num_dets:
                            gemini_predictions[c_idx - 1] = {
                                "class_key": item.get("class_key"),
                                "confidence": float(item.get("confidence", 0.85)),
                                "reasoning": item.get("reasoning", ""),
                            }

            logger.info(f"Gemini Multimodal HD Crops arbitrated {len(gemini_predictions)} ambiguous cells.")

        except Exception as gemini_err:
            logger.error(f"Gemini Multimodal reasoning error: {gemini_err}", exc_info=True)

    # 5. Calibrated Multimodal Fusion (CONCH + Virchow 2 + Gemini HD Crops)
    final_classified = []
    for i, det in enumerate(conch_classified):
        det_copy = dict(det)
        orig_det = detections[i] if i < len(detections) else {}

        # Preserve user manual labels unconditionally
        if orig_det.get("is_user_exemplar") or orig_det.get("manual_override"):
            det_copy["is_user_exemplar"] = True
            det_copy["virchow_confidence"] = 1.0
            det_copy["gemini_confidence"] = 1.0
            det_copy["score"] = 1.0
            det_copy["gemini_reasoning"] = "Etiqueta manual confirmada por el usuario."
            final_classified.append(det_copy)
            continue

        gemini_pred = gemini_predictions.get(i)
        conch_key = det_copy.get("class_key")
        conch_conf = float(det_copy.get("conch_confidence", 0.5))
        conch_margin = float(det_copy.get("conch_margin", 0.1))
        conch_scores = det_copy.get("conch_scores", {})

        if gemini_pred and gemini_pred.get("class_key") in class_meta_map:
            g_key = gemini_pred["class_key"]
            g_conf = float(gemini_pred.get("confidence", 0.85))
            g_reason = gemini_pred.get("reasoning", "")

            # Decision Logic: Calibrated Agreement vs Arbitration
            if g_key == conch_key:
                # Strong consensus between CONCH and Gemini HD Crop
                chosen_key = conch_key
                fused_score = min(0.98, max(conch_conf, 0.75) + 0.10)
                reason = f"Consenso patológico (CONCH {conch_conf:.2f} + Gemini HD). {g_reason}"
            elif conch_margin >= 0.25 and conch_conf >= 0.60:
                # CONCH has high margin confidence on morphological features -> trust CONCH
                chosen_key = conch_key
                fused_score = round(0.70 * conch_conf + 0.30 * g_conf, 4)
                reason = f"Morfología CONCH dominante (margen {conch_margin:.2f})."
            else:
                # CONCH was ambiguous/low margin -> Gemini HD Crop arbitration resolves the tie
                chosen_key = g_key
                fused_score = round(0.65 * g_conf + 0.35 * conch_scores.get(g_key, 0.35), 4)
                reason = f"Arbitraje visual Gemini sobre crop HD: {g_reason}"

            meta = class_meta_map[chosen_key]
            det_copy["category_id"] = chosen_key
            det_copy["class_key"] = chosen_key
            det_copy["class_label"] = meta["label"]
            det_copy["color"] = meta["color"]
            det_copy["gemini_confidence"] = round(g_conf, 4)
            det_copy["gemini_reasoning"] = reason
            det_copy["multimodal_fused"] = True
            det_copy["score"] = round(fused_score, 4)

        elif conch_key in class_meta_map:
            meta = class_meta_map[conch_key]
            det_copy["category_id"] = conch_key
            det_copy["class_key"] = conch_key
            det_copy["class_label"] = meta["label"]
            det_copy["color"] = meta["color"]
            det_copy["gemini_confidence"] = round(conch_conf, 4)
            det_copy["gemini_reasoning"] = f"Clasificado por similitud morfológica CONCH (margen: {conch_margin:.2f})."
            det_copy["score"] = round(conch_conf, 4)

        final_classified.append(det_copy)

    # 6. Data-driven neighbourhood consistency with Dual Foundation (Virchow 2 1280d + UNI 1024d)
    if morph_sim_matrix is not None and len(final_classified) > 1:
        for i, det in enumerate(final_classified):
            if det.get("is_user_exemplar"):
                continue
            # Check if an uncertain cell is strongly aligned (>0.88) with a high-confidence neighbour
            if float(det.get("score", 0.0)) < 0.50:
                best_sim = -1.0
                best_j = -1
                for j, other in enumerate(final_classified):
                    if i == j:
                        continue
                    if float(other.get("score", 0.0)) >= 0.75:
                        sim = float(morph_sim_matrix[i, j])
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
                    det["gemini_reasoning"] += f" (Consistencia morfológica Virchow2+UNI: {best_sim:.2f})"

    return final_classified


def validate_uncertain_detections_with_gemini(
    image: Image.Image,
    uncertain_detections: List[Dict[str, Any]],
    ontology_classes: List[Dict[str, Any]],
    organ_context: str = "histología",
    api_key: Optional[str] = None,
    max_to_validate: int = 25,
) -> List[Dict[str, Any]]:
    """
    Arbitrates and validates ambiguous/uncertain histological instances using Gemini 2.5 Flash Vision.

    Selects ambiguous detections, passes contextual high-resolution crops along with the full slide context,
    and returns adjudications conforming strictly to candidate ontology classes.
    """
    if not uncertain_detections or not ontology_classes:
        return uncertain_detections

    client = _get_gemini_client(api_key)
    if client is None:
        logger.warning("Gemini client not available for uncertain detection validation.")
        return uncertain_detections

    class_meta = {
        c.get("key"): {
            "label": c.get("label", c.get("name", c.get("key"))),
            "color": c.get("color", "#8b5cf6"),
            "prompt": c.get("prompt", ""),
        }
        for c in ontology_classes
        if c.get("key")
    }
    class_list_desc = [
        f"- key: '{c.get('key')}', name: '{c.get('label', c.get('name'))}', description: '{c.get('prompt', '')}'"
        for c in ontology_classes
    ]

    target_dets = uncertain_detections[:max_to_validate]
    img_w, img_h = image.size

    # Batch crops into composite or individual evaluations
    try:
        crops_to_send = []
        for idx, det in enumerate(target_dets):
            bbox = det.get("bbox", [0, 0, img_w, img_h])
            bx, by, bw, bh = bbox
            pad = max(12, int(max(bw, bh) * 0.4))
            x1 = max(0, bx - pad)
            y1 = max(0, by - pad)
            x2 = min(img_w, bx + bw + pad)
            y2 = min(img_h, by + bh + pad)
            crop = image.crop((x1, y1, x2, y2)).convert("RGB")
            crops_to_send.append((idx, crop, det.get("class_key", "unknown"), det.get("score", 0.0)))

        # Build prompt
        prompt_parts = [
            f"You are an expert histopathologist evaluating ambiguous microscopic cell/tissue instances in {organ_context}.",
            "Here is the list of VALID ONTOLOGY CLASSES you must choose from:",
            "\n".join(class_list_desc),
            "",
            "Review each numbered crop image and assign the most accurate ontology class_key, confidence (0.0 to 1.0), and short rationale.",
            "Respond ONLY with valid JSON in this exact structure:",
            "```json",
            "{",
            '  "adjudications": [',
            '    {"crop_index": 0, "class_key": "<exact_key>", "confidence": 0.88, "reasoning": "<brief explanation>"}',
            "  ]",
            "}",
            "```",
        ]

        contents_list = ["\n".join(prompt_parts)]
        for idx, crop, tentative_key, tentative_score in crops_to_send:
            contents_list.append(f"--- Crop #{idx} (Tentative ensemble class: '{tentative_key}', score: {tentative_score:.2f}) ---")
            contents_list.append(crop)

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=contents_list,
        )

        resp_text = response.text or ""
        match = re.search(r"\{[\s\S]*\}", resp_text)
        if match:
            parsed = json.loads(match.group(0))
            adjudications = parsed.get("adjudications", [])
            for adj in adjudications:
                crop_idx = int(adj.get("crop_index", -1))
                if 0 <= crop_idx < len(target_dets):
                    target_det = target_dets[crop_idx]
                    adj_key = adj.get("class_key")
                    adj_conf = float(adj.get("confidence", 0.75))
                    adj_reason = adj.get("reasoning", "")

                    if adj_key in class_meta:
                        meta = class_meta[adj_key]
                        target_det["category_id"] = adj_key
                        target_det["class_key"] = adj_key
                        target_det["class_label"] = meta["label"]
                        target_det["color"] = meta["color"]
                        target_det["gemini_validated"] = True
                        target_det["gemini_confidence"] = round(adj_conf, 4)
                        target_det["gemini_reasoning"] = adj_reason
                        target_det["score"] = round(max(adj_conf, float(target_det.get("score", 0.5))), 4)
                        if adj_conf >= 0.70:
                            target_det["classification_uncertain"] = False
                        target_det["decision_source"] = "gemini_multimodal_arbitration"

    except Exception as e:
        logger.warning(f"Error during Gemini uncertain detection validation: {e}")

    return uncertain_detections


def detect_histological_macro_layers_gemini(
    image: Image.Image,
    organ_context: str = "histología",
    ontology_structures: Optional[List[Dict[str, Any]]] = None,
    api_key: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Multimodal Spatial Grounding for Macro-Histological Layers & Compartments:
    Dynamically extracts target layer descriptions directly from the ontology structures
    (e.g. from the PDF) and detects 2D bounding boxes [ymin, xmin, ymax, xmax].
    Zero hardcoded tissue lists.
    """
    client = _get_gemini_client(api_key)
    if client is None:
        logger.warning("Gemini client not available for macro-layer grounding.")
        return []

    img_w, img_h = image.size
    img_preview = image.copy().convert("RGB")
    max_dim = 1024
    if max(img_preview.size) > max_dim:
        ratio = max_dim / max(img_preview.size)
        img_preview = img_preview.resize(
            (int(img_preview.width * ratio), int(img_preview.height * ratio)),
            Image.LANCZOS,
        )

    # Dynamically extract target descriptions from the provided ontology document
    target_descriptions: List[str] = []
    class_meta_lookup: Dict[str, Dict[str, Any]] = {}

    if ontology_structures:
        for idx, s in enumerate(ontology_structures):
            k = s.get("key", f"struct_{idx + 1}").strip()
            name = s.get("label", s.get("name", k))
            prompt_desc = s.get("prompt", "")
            stype = s.get("structure_type", "")

            class_meta_lookup[k] = {
                "name": name,
                "label": name,
                "color": s.get("color", DEFAULT_COLORS[idx % len(DEFAULT_COLORS)]),
                "prompt": prompt_desc,
                "structure_type": stype,
            }
            target_descriptions.append(f"- '{k}': {name} — visual descriptor: '{prompt_desc}'")
    else:
        target_descriptions = [
            f"- Distinct visual anatomical structures, tissue layers, and compartments visible in {organ_context}"
        ]

    prompt = f"""\
You are an expert anatomical histopathologist. Analyze this histological photomicrograph of {organ_context}.
Your task is to identify and spatially localize ALL continuous macro-architectural layers, lumens, and tissue compartments present in the image.

TARGET STRUCTURES FROM ONTOLOGY:
{chr(10).join(target_descriptions)}

For EACH distinct compartment or layer present in the image, output its exact bounding box coordinates in normalized scale [ymin, xmin, ymax, xmax] (integers 0 to 1000).

Return ONLY valid JSON matching this schema:
```json
{{
  "layers": [
    {{
      "key": "<exact_key_from_ontology>",
      "name": "<canonical_name>",
      "box_2d": [ymin, xmin, ymax, xmax],
      "description": "<concise visual description>",
      "confidence": 0.95
    }}
  ]
}}
```
"""

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[prompt, img_preview],
        )
        resp_text = response.text or ""
        match = re.search(r"\{[\s\S]*\}", resp_text)
        if not match:
            return []

        parsed = json.loads(match.group(0))
        raw_layers = parsed.get("layers", [])

        grounded_layers: List[Dict[str, Any]] = []
        for idx, item in enumerate(raw_layers):
            box = item.get("box_2d")
            if not box or len(box) != 4:
                continue

            ymin, xmin, ymax, xmax = [float(v) for v in box]
            px_x1 = max(0.0, min(float(img_w), (xmin / 1000.0) * img_w))
            px_y1 = max(0.0, min(float(img_h), (ymin / 1000.0) * img_h))
            px_x2 = max(0.0, min(float(img_w), (xmax / 1000.0) * img_w))
            px_y2 = max(0.0, min(float(img_h), (ymax / 1000.0) * img_h))
            bw = max(1.0, px_x2 - px_x1)
            bh = max(1.0, px_y2 - px_y1)

            k = item.get("key", f"layer_{idx + 1}").lower().replace("-", "_")
            meta = class_meta_lookup.get(k, {})
            name = meta.get("label", item.get("name", k.replace("_", " ").title()))
            color = meta.get("color", DEFAULT_COLORS[idx % len(DEFAULT_COLORS)])
            stype = meta.get("structure_type", "tissue_layer")

            # Estimate initial polygon box
            poly = [
                px_x1, px_y1,
                px_x2, px_y1,
                px_x2, px_y2,
                px_x1, px_y2,
            ]

            grounded_layers.append({
                "id": f"layer_{idx + 1}",
                "key": k,
                "class_key": k,
                "class_label": name,
                "category_id": k,
                "label": name,
                "color": color,
                "structure_type": stype,
                "is_macro_layer": True,
                "score": float(item.get("confidence", 0.90)),
                "box": [round(px_x1, 1), round(px_y1, 1), round(px_x2, 1), round(px_y2, 1)],
                "bbox": [round(px_x1, 1), round(px_y1, 1), round(bw, 1), round(bh, 1)],
                "segmentation": [poly],
                "area": round(bw * bh, 1),
                "description": item.get("description", ""),
                "decision_source": "gemini_spatial_grounding",
            })

        logger.info(f"Gemini grounded {len(grounded_layers)} macro-layers for {organ_context}: {[l['key'] for l in grounded_layers]}")
        return grounded_layers

    except Exception as e:
        logger.warning(f"Error in macro-layer grounding with Gemini: {e}")
        return []



