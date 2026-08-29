"""
Histology Multi-Agent Workflow using LangGraph, Gemini, CONCH, and Virchow 2.

Architecture:
1. OntologyReaderNode (Gemini / Ontology Store): Reads domain ontology, extracts candidate cellular classes & visual cues.
2. ImageLabelDetectorNode (Gemini Vision OCR & Figure Grounding): Detects embedded visual labels, letter codes (e.g. A, B, S, L), arrows, and legend keys in histological figures.
3. SegmentationCropperNode (Deterministic PyTorch / OpenCV / Cellpose / SAM3): Extracts cell/nuclei ROIs, patches, and morphological descriptors.
4. FoundationMatcherNode (CONCH & Virchow 2 PyTorch Workers): Computes zero-shot vision-language similarities and ViT-H 1280d morphological embeddings.
5. FinalClassifierNode (Gemini 3.1 / 2.5 Flash): Synthesizes foundation model scores, morphological metrics, spatial figure labels, and ontology constraints to output validated annotations.
"""

import io
import json
import logging
import math
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple, TypedDict, Union

import cv2
import numpy as np
from PIL import Image
import torch

import sys
from pathlib import Path

# Ensure backend and root are in sys.path for direct and modular imports
_backend_dir = str(Path(__file__).resolve().parent)
if _backend_dir not in sys.path:
    sys.path.insert(0, _backend_dir)
_root_dir = str(Path(__file__).resolve().parent.parent)
if _root_dir not in sys.path:
    sys.path.insert(0, _root_dir)

try:
    from langgraph.graph import StateGraph, START, END
except ImportError:
    StateGraph = None
    START = "__start__"
    END = "__end__"

try:
    from backend.gemini_vision import _get_gemini_client
    from backend.pathology_models import (
        classify_detections_with_conch,
        classify_with_virchow_prototypes,
        classify_with_morphological_ensemble,
        filter_cellular_candidate_classes,
        enrich_histology_prompt,
        extract_crops_from_detections,
        ConchModelWrapper,
        VirchowModelWrapper,
        UniModelWrapper,
    )
    from backend.pdf_ontology import list_ontologies, load_ontology
except ImportError:
    from gemini_vision import _get_gemini_client
    from pathology_models import (
        classify_detections_with_conch,
        classify_with_virchow_prototypes,
        classify_with_morphological_ensemble,
        filter_cellular_candidate_classes,
        enrich_histology_prompt,
        extract_crops_from_detections,
        ConchModelWrapper,
        VirchowModelWrapper,
        UniModelWrapper,
    )
    from pdf_ontology import list_ontologies, load_ontology

logger = logging.getLogger("sam3-langgraph")


# ---------------------------------------------------------------------------
# State Definition
# ---------------------------------------------------------------------------

class DetectedFigureLabel(TypedDict, total=False):
    text: str
    meaning: Optional[str]
    box_2d: Optional[List[int]]  # [ymin, xmin, ymax, xmax] in 0-1000 scale
    box_pixels: Optional[List[int]]  # [x1, y1, x2, y2]
    center: Optional[Tuple[float, float]]
    confidence: Optional[float]


class HistologyGraphState(TypedDict, total=False):
    # Input Data
    image_bytes: Optional[bytes]
    image_pil: Optional[Any]  # PIL.Image instance
    image_size: Tuple[int, int]  # (width, height)
    ontology_name: Optional[str]
    user_context_hint: Optional[str]

    # Node 1: Ontology Output
    raw_ontology: List[Dict[str, Any]]
    candidate_classes: List[Dict[str, Any]]
    organ_tissue_context: str

    # Node 2: Image Visual Labels & Legend Detection
    detected_figure_labels: List[DetectedFigureLabel]
    figure_abbreviations_map: Dict[str, str]
    figure_visual_notes: str

    # Node 3: Segmentations & Morphometry
    detections: List[Dict[str, Any]]
    segmentation_source: str

    # Node 4: Foundation Models (CONCH, Virchow2 & UNI)
    conch_scored_detections: List[Dict[str, Any]]
    virchow_features_computed: bool
    uni_features_computed: bool

    # Node 5: Final Decisions & Reasoning
    final_detections: List[Dict[str, Any]]
    classification_summary: Dict[str, Any]
    reasoning_log: List[Dict[str, Any]]
    execution_metrics: Dict[str, Any]
    error: Optional[str]


# ---------------------------------------------------------------------------
# Node 1: Ontology Reader Agent
# ---------------------------------------------------------------------------

def ontology_reader_node(state: HistologyGraphState) -> Dict[str, Any]:
    """
    Reads active histology ontology, filters cellular classes, and generates
    specialized prompt descriptions for vision-language matching.
    """
    start_t = time.time()
    ontology_name = state.get("ontology_name")
    raw_ontology = state.get("raw_ontology") or []
    explicit_candidates = state.get("candidate_classes") or []

    cellular_classes: List[Dict[str, Any]] = []

    # Priority 1: Use explicitly provided candidate classes
    if explicit_candidates:
        cellular_classes = filter_cellular_candidate_classes(explicit_candidates)
        if not cellular_classes:
            cellular_classes = list(explicit_candidates)

    # Priority 2: Load from active ontology name
    if not cellular_classes and ontology_name:
        try:
            loaded = load_ontology(ontology_name)
            if loaded:
                if isinstance(loaded, dict):
                    raw_ontology = loaded.get("structures", [])
                elif isinstance(loaded, list):
                    raw_ontology = loaded
                cellular_classes = filter_cellular_candidate_classes(raw_ontology)
        except Exception as e:
            logger.warning(f"Could not load ontology '{ontology_name}': {e}")

    # Priority 3: Fallback to first available saved ontology
    if not cellular_classes and not raw_ontology:
        try:
            all_onts = list_ontologies()
            if all_onts:
                first_name = all_onts[0].get("name")
                if first_name:
                    loaded = load_ontology(first_name)
                    if loaded:
                        if isinstance(loaded, dict):
                            raw_ontology = loaded.get("structures", [])
                        elif isinstance(loaded, list):
                            raw_ontology = loaded
                        cellular_classes = filter_cellular_candidate_classes(raw_ontology)
        except Exception as e:
            logger.debug(f"Fallback ontology load: {e}")

    # Priority 4: If still empty, provide standard histological candidate classes
    if not cellular_classes:
        cellular_classes = [
            {
                "key": "spermatogonia",
                "label": "Espermatogonia",
                "name": "Espermatogonia",
                "prompt": "round dark basal germ cell nucleus with dense chromatin",
                "color": "#e11d48",
            },
            {
                "key": "spermatocyte",
                "label": "Espermatocito primario",
                "name": "Espermatocito primario",
                "prompt": "large rounded nucleus with thread-like meiotic chromatin prophase",
                "color": "#8b5cf6",
            },
            {
                "key": "spermatid",
                "label": "Espermátide",
                "name": "Espermátide",
                "prompt": "small compact round or elongated condensed nucleus near lumen",
                "color": "#06b6d4",
            },
            {
                "key": "sertoli_cell",
                "label": "Célula de Sertoli",
                "name": "Célula de Sertoli",
                "prompt": "pale vesicular oval nucleus with prominent central nucleolus",
                "color": "#10b981",
            },
            {
                "key": "leydig_cell",
                "label": "Célula de Leydig",
                "name": "Célula de Leydig",
                "prompt": "interstitial steroidogenic polygonal cell with round nucleus and rich eosinophilic cytoplasm",
                "color": "#f59e0b",
            },
            {
                "key": "fibroblast",
                "label": "Fibroblasto / Mioide",
                "name": "Fibroblasto / Mioide",
                "prompt": "flattened spindle-shaped elongated dark nucleus in peritubular wall",
                "color": "#6366f1",
            },
        ]

    # Ensure all classes have valid keys, labels, prompts, and colors
    from backend.gemini_vision import DEFAULT_COLORS
    for i, c in enumerate(cellular_classes):
        c["key"] = c.get("key") or f"class_{i}"
        c["label"] = c.get("label") or c.get("name") or c["key"]
        c["name"] = c.get("name") or c["label"]
        c["color"] = c.get("color") or DEFAULT_COLORS[i % len(DEFAULT_COLORS)]
        c["prompt"] = c.get("prompt") or c["label"]

    # Derive organ/tissue context
    organ_context = state.get("user_context_hint") or "Histología / Microscopía Óptica"

    elapsed = time.time() - start_t
    logger.info(f"[OntologyReaderNode] Prepared {len(cellular_classes)} candidate classes in {elapsed:.3f}s: {[c['label'] for c in cellular_classes]}")

    return {
        "raw_ontology": raw_ontology,
        "candidate_classes": cellular_classes,
        "organ_tissue_context": organ_context,
    }


# ---------------------------------------------------------------------------
# Node 2: Image Visual Labels & Legend Detector (Gemini Vision OCR & Callouts)
# ---------------------------------------------------------------------------

def image_label_detector_node(state: HistologyGraphState) -> Dict[str, Any]:
    """
    Multimodal agent that scans the full histological microphotograph to detect
    embedded letter codes (A, B, S, L, N, Sp), arrows, pointers, scale bars,
    panel captions, and figure legends that provide spatial ground truth.
    """
    start_t = time.time()
    img_pil = state.get("image_pil")
    img_bytes = state.get("image_bytes")

    if img_pil is None and img_bytes:
        try:
            img_pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        except Exception as e:
            logger.error(f"Failed to open image in label detector: {e}")
            return {"detected_figure_labels": [], "figure_abbreviations_map": {}, "figure_visual_notes": "Image parse error"}

    if img_pil is None:
        return {"detected_figure_labels": [], "figure_abbreviations_map": {}, "figure_visual_notes": "No image provided"}

    img_w, img_h = img_pil.size
    client = _get_gemini_client()

    detected_labels: List[DetectedFigureLabel] = []
    abbreviations_map: Dict[str, str] = {}
    visual_notes = ""

    if client is not None:
        try:
            # Resize preview image for efficient multimodal OCR
            preview = img_pil.copy()
            max_d = 1024
            if max(preview.size) > max_d:
                scale = max_d / max(preview.size)
                preview = preview.resize((int(preview.width * scale), int(preview.height * scale)), Image.LANCZOS)

            candidate_names = [c.get("label", c.get("name", c.get("key", ""))) for c in state.get("candidate_classes", [])]

            prompt_text = f"""
You are an expert histopathology figure analyst. Inspect this histological image and detect all embedded annotations, text labels, letter badges (e.g., 'A', 'B', 'S', 'L', 'Sp', 'N'), pointer arrows, callouts, and figure legends.

Candidate tissue classes in this domain:
{json.dumps(candidate_names, ensure_ascii=False)}

Task:
1. Identify every text label, letter badge, or arrow pointer visible in the micrograph.
2. If letter codes are used (like 'S' for Sertoli, 'L' for Leydig, 'A' for Spermatogonia A), interpret their biological meaning based on context and candidates.
3. Return bounding boxes for each label in [ymin, xmin, ymax, xmax] (normalized from 0 to 1000).

Return ONLY valid JSON matching this structure:
{{
  "abbreviations": {{ "S": "Célula de Sertoli", "L": "Célula de Leydig" }},
  "labels": [
    {{
      "text": "S",
      "meaning": "Célula de Sertoli",
      "box_2d": [100, 200, 150, 250],
      "confidence": 0.95
    }}
  ],
  "visual_notes": "Brief summary of visible tissue structures and annotations"
}}
"""

            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[prompt_text, preview],
                config={"response_mime_type": "application/json"},
            )

            res_text = response.text or "{}"
            parsed = json.loads(res_text)

            abbreviations_map = parsed.get("abbreviations", {})
            raw_labels = parsed.get("labels", [])
            visual_notes = parsed.get("visual_notes", "")

            # Convert 0-1000 coordinates to actual pixel coordinates
            for item in raw_labels:
                box_2d = item.get("box_2d")
                box_px = None
                center = None
                if box_2d and len(box_2d) == 4:
                    ymin, xmin, ymax, xmax = box_2d
                    x1 = int((xmin / 1000.0) * img_w)
                    y1 = int((ymin / 1000.0) * img_h)
                    x2 = int((xmax / 1000.0) * img_w)
                    y2 = int((ymax / 1000.0) * img_h)
                    box_px = [x1, y1, x2, y2]
                    center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

                detected_labels.append({
                    "text": item.get("text", ""),
                    "meaning": item.get("meaning") or abbreviations_map.get(item.get("text", "")),
                    "box_2d": box_2d,
                    "box_pixels": box_px,
                    "center": center,
                    "confidence": float(item.get("confidence", 0.9)),
                })

            logger.info(f"[ImageLabelDetectorNode] Detected {len(detected_labels)} visual figure labels in {time.time() - start_t:.3f}s")
        except Exception as e:
            logger.warning(f"[ImageLabelDetectorNode] Error detecting visual labels: {e}")
            visual_notes = f"Label detector skipped: {e}"

    return {
        "detected_figure_labels": detected_labels,
        "figure_abbreviations_map": abbreviations_map,
        "figure_visual_notes": visual_notes,
    }


# ---------------------------------------------------------------------------
# Node 3: Segmentation & Crop Extractor (Deterministic PyTorch/OpenCV)
# ---------------------------------------------------------------------------

def segmentation_cropper_node(state: HistologyGraphState) -> Dict[str, Any]:
    """
    Ensures all detections have bounding boxes, polygons, morphological descriptors
    (area, aspect ratio, circularity, mean intensity), and associates nearby
    figure labels as prior hints.
    """
    start_t = time.time()
    detections = state.get("detections") or []
    img_pil = state.get("image_pil")
    img_bytes = state.get("image_bytes")

    if img_pil is None and img_bytes:
        try:
            img_pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        except Exception as e:
            logger.error(f"Failed to open image in cropper: {e}")

    np_img = np.array(img_pil) if img_pil is not None else None
    gray_img = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY) if np_img is not None else None

    figure_labels = state.get("detected_figure_labels") or []

    enriched_detections: List[Dict[str, Any]] = []

    for idx, det in enumerate(detections):
        d = dict(det)
        det_id = d.get("id", f"cell_{idx:04d}")
        d["id"] = det_id

        # Calculate bounding box from polygon if missing
        polygon = d.get("polygon")
        bbox = d.get("bbox")

        if (not bbox or len(bbox) != 4) and polygon and len(polygon) >= 3:
            xs = [p[0] for p in polygon]
            ys = [p[1] for p in polygon]
            x1, x2 = min(xs), max(xs)
            y1, y2 = min(ys), max(ys)
            bbox = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
            d["bbox"] = bbox

        # Compute morphometric features
        area = d.get("area", 0.0)
        circularity = 1.0
        aspect_ratio = 1.0
        mean_intensity = 128.0

        if polygon and len(polygon) >= 3:
            pts = np.array(polygon, dtype=np.int32)
            calc_area = float(cv2.contourArea(pts))
            perimeter = float(cv2.arcLength(pts, True))
            if calc_area > 0:
                area = calc_area
            if perimeter > 0:
                circularity = float((4.0 * math.pi * area) / (perimeter * perimeter))
                circularity = min(1.0, max(0.0, circularity))

            # Bounding box aspect ratio
            if bbox and len(bbox) == 4:
                w, h = max(1.0, bbox[2]), max(1.0, bbox[3])
                aspect_ratio = float(min(w, h) / max(w, h))

            # Mean pixel intensity inside mask
            if gray_img is not None and np_img is not None:
                mask = np.zeros(gray_img.shape, dtype=np.uint8)
                cv2.drawContours(mask, [pts], -1, 255, -1)
                mean_val = cv2.mean(gray_img, mask=mask)[0]
                if mean_val > 0:
                    mean_intensity = float(mean_val)

        d["area"] = area
        d["circularity"] = circularity
        d["aspect_ratio"] = aspect_ratio
        d["mean_intensity"] = mean_intensity

        # Spatial association with detected figure labels / arrows
        if bbox and len(bbox) == 4 and figure_labels:
            cx = bbox[0] + bbox[2] / 2.0
            cy = bbox[1] + bbox[3] / 2.0

            min_dist = float("inf")
            nearest_label: Optional[DetectedFigureLabel] = None

            for fl in figure_labels:
                lbl_center = fl.get("center")
                if lbl_center:
                    dist = math.hypot(cx - lbl_center[0], cy - lbl_center[1])
                    if dist < min_dist:
                        min_dist = dist
                        nearest_label = fl

            # If within 150px of an arrow or label, add spatial label hint
            if nearest_label and min_dist < 150.0:
                d["spatial_label_hint"] = {
                    "text": nearest_label.get("text"),
                    "meaning": nearest_label.get("meaning"),
                    "distance_px": round(min_dist, 1),
                }

        enriched_detections.append(d)

    logger.info(f"[SegmentationCropperNode] Processed morphometry for {len(enriched_detections)} nuclei in {time.time() - start_t:.3f}s")
    return {"detections": enriched_detections}


# ---------------------------------------------------------------------------
# Node 4: Foundation Models Evaluator (CONCH & Virchow 2 Workers)
# ---------------------------------------------------------------------------

def foundation_matcher_node(state: HistologyGraphState) -> Dict[str, Any]:
    """
    Executes CONCH zero-shot vision-language classification and Virchow 2
    ViT-H 1280d morphological representations on segmented cellular patches.
    """
    start_t = time.time()
    detections = state.get("detections") or []
    candidate_classes = state.get("candidate_classes") or []
    img_pil = state.get("image_pil")
    img_bytes = state.get("image_bytes")

    if not detections or not candidate_classes:
        return {"conch_scored_detections": detections, "virchow_features_computed": False}

    if img_pil is None and img_bytes:
        try:
            img_pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        except Exception as e:
            logger.error(f"Failed to open image in foundation matcher: {e}")
            return {"conch_scored_detections": detections, "virchow_features_computed": False}

    if img_pil is None:
        return {"conch_scored_detections": detections, "virchow_features_computed": False}

    scored_detections = detections

    # 1. Run CONCH Zero-Shot Classification on all detections
    try:
        logger.info(f"[FoundationMatcherNode] Running CONCH zero-shot matching on {len(detections)} cells with {len(candidate_classes)} candidate classes (T=0.12)...")
        scored_detections = classify_detections_with_conch(
            image=img_pil,
            detections=detections,
            candidate_classes=candidate_classes,
            temperature=0.12,
            is_histology=True,
        )
    except Exception as e:
        logger.warning(f"[FoundationMatcherNode] CONCH zero-shot skipped or failed: {e}")

    # 2. Run Morphological Ensemble Evaluation (Virchow 2 1280d + UNI 1024d)
    virchow_computed = False
    uni_computed = False
    try:
        virchow_wrapper = VirchowModelWrapper.get_instance()
        if not virchow_wrapper.is_loaded:
            virchow_wrapper.load()
        virchow_computed = bool(virchow_wrapper.is_loaded)

        uni_wrapper = UniModelWrapper.get_instance()
        if not uni_wrapper.is_loaded:
            uni_wrapper.load()
        uni_computed = bool(uni_wrapper.is_loaded)

        if virchow_computed or uni_computed:
            logger.info(f"[FoundationMatcherNode] Running Dual Foundation Ensemble (Virchow 2: {virchow_computed}, UNI: {uni_computed})...")
            scored_detections = classify_with_morphological_ensemble(
                image=img_pil,
                detections=scored_detections,
                candidate_classes=candidate_classes,
                temperature=0.12,
                is_histology=True,
            )
        else:
            logger.warning("[FoundationMatcherNode] Neither Virchow 2 nor UNI could be loaded; continuing with CONCH zero-shot.")
    except Exception as e:
        logger.warning(f"[FoundationMatcherNode] Morphological ensemble skipped: {e}")

    logger.info(f"[FoundationMatcherNode] Completed in {time.time() - start_t:.3f}s")
    return {
        "conch_scored_detections": scored_detections,
        "virchow_features_computed": virchow_computed,
        "uni_features_computed": uni_computed,
    }


# ---------------------------------------------------------------------------
# Node 5: Final Classifier & Synthesis Agent (Gemini 3.1 / 2.5 Flash Multimodal)
# ---------------------------------------------------------------------------

def final_classifier_node(state: HistologyGraphState) -> Dict[str, Any]:
    """
    Synthesizes:
    - CONCH zero-shot predictions & probability distribution across classes
    - Dual Foundation Morphological Ensemble (Virchow 2 1280d + UNI 1024d)
    - Visual crops passed directly to Gemini for multimodal visual arbitration
    - Spatial figure labels / legend codes (Agent 2)
    - Domain ontology constraints (Agent 1)
    to output canonical, diversified, reasoned cellular classes for each nucleus.
    """
    start_t = time.time()
    detections = state.get("conch_scored_detections") or state.get("detections") or []
    candidate_classes = state.get("candidate_classes") or []
    figure_labels = state.get("detected_figure_labels") or []
    abbrev_map = state.get("figure_abbreviations_map") or {}
    visual_notes = state.get("figure_visual_notes") or ""
    img_pil = state.get("image_pil")
    img_bytes = state.get("image_bytes")

    if not detections:
        return {
            "final_detections": [],
            "classification_summary": {"total": 0, "by_class": {}, "by_decision_source": {}},
            "reasoning_log": [],
            "execution_metrics": {
                "total_time_seconds": round(time.time() - start_t, 3),
                "decision_counts": {},
                "morphometric_heuristic_count": 0,
            },
        }

    if img_pil is None and img_bytes:
        try:
            img_pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        except Exception as e:
            logger.warning(f"[FinalClassifierNode] Could not load image from bytes: {e}")
            img_pil = None

    client = _get_gemini_client()
    final_detections: List[Dict[str, Any]] = []
    reasoning_log: List[Dict[str, Any]] = []

    # Map candidate keys/labels for fast lookup
    class_map = {c.get("key"): c for c in candidate_classes}

    # Step 1: Initial pass with spatial consensus and foundation model scores
    ambiguous_items: List[Dict[str, Any]] = []

    for d in detections:
        conch_key = d.get("class_key")
        conf = float(d.get("conch_confidence") or d.get("confidence") or 0.5)
        scores = d.get("conch_scores") or d.get("class_scores") or {}
        spatial_hint = d.get("spatial_label_hint")

        # Check if spatial label matches candidate class
        matched_by_spatial = False
        if spatial_hint:
            hint_text = str(spatial_hint.get("text", "")).lower().strip()
            hint_meaning = str(spatial_hint.get("meaning", "")).lower().strip()
            for k, c in class_map.items():
                c_lbl = str(c.get("label") or c.get("name") or "").lower()
                c_key = str(k).lower()
                if (
                    (hint_text and (hint_text == c_key or hint_text in c_lbl or c_lbl in hint_text))
                    or (hint_meaning and (hint_meaning in c_lbl or c_lbl in hint_meaning))
                    or (hint_meaning and (hint_meaning in c_key or c_key in hint_meaning))
                ):
                    d["class_key"] = k
                    d["class_label"] = c.get("label", c.get("name", k))
                    d["color"] = c.get("color", "#8b5cf6")
                    d["confidence"] = max(conf, 0.92)
                    d["decision_source"] = "spatial_figure_label_consensus"
                    d["agent_reasoning"] = f"Aligned with figure label '{spatial_hint.get('text')}' ({spatial_hint.get('meaning')}) at {spatial_hint.get('distance_px')}px"
                    matched_by_spatial = True
                    break

        if not matched_by_spatial:
            # Check if CONCH already gave a valid candidate class
            if conch_key and conch_key in class_map:
                matched_c = class_map[conch_key]
                d["class_label"] = matched_c.get("label", matched_c.get("name", conch_key))
                d["color"] = matched_c.get("color", "#8b5cf6")
                d["decision_source"] = "foundation_conch_zero_shot"
                d["agent_reasoning"] = f"CONCH vision-language similarity ({conf:.2f})"

                sorted_scores = sorted(scores.values(), reverse=True) if scores else []
                margin = (sorted_scores[0] - sorted_scores[1]) if len(sorted_scores) >= 2 else 1.0

                # Calibrated ambiguity condition: only truly borderline margins (< 0.12) or very low confidence (< 0.40)
                if conf < 0.40 or margin < 0.12:
                    ambiguous_items.append(d)
            else:
                ambiguous_items.append(d)

    # Step 2: Visual Multimodal Arbitration with Gemini Vision (Passing Image Crops)
    if ambiguous_items and client is not None:
        try:
            chunk_size = 12  # Chunk size optimized for high-quality multimodal visual attention
            for chunk_start in range(0, len(ambiguous_items), chunk_size):
                chunk = ambiguous_items[chunk_start : chunk_start + chunk_size]

                # Extract visual crops for the ambiguous cells in this chunk
                chunk_crops = extract_crops_from_detections(img_pil, chunk, margin_ratio=0.35, min_size=80) if img_pil is not None else [None] * len(chunk)

                header_prompt = f"""You are a senior histological pathologist performing visual multimodal arbitration on ambiguous cellular nuclei in tissue sections.

Available Ontological Classes:
{json.dumps([{"key": c["key"], "label": c.get("label", c.get("name")), "prompt": c.get("prompt")} for c in candidate_classes], ensure_ascii=False, indent=2)}

Detected Figure Legend & Abbreviations:
{json.dumps(abbrev_map, ensure_ascii=False, indent=2)}
Figure Visual Notes: {visual_notes}

Examine each cell's high-resolution visual crop, its cytological morphometry, its dual-foundation embeddings (Virchow 2 + UNI), and its CONCH zero-shot scores to determine the precise class_key.
"""
                contents_payload: List[Any] = [header_prompt]

                for idx, it in enumerate(chunk):
                    crop_img = chunk_crops[idx]
                    det_id = it.get("id", f"cell_{idx+1}")
                    morph_ensemble_str = ""
                    if it.get("morphological_ensemble_similarity") is not None:
                        morph_ensemble_str = f"Morphological Ensemble Affinity (Virchow 2 + UNI): {float(it['morphological_ensemble_similarity']):.3f}\n"
                    elif it.get("virchow_similarity") is not None:
                        morph_ensemble_str = f"Virchow 2 Affinity: {float(it['virchow_similarity']):.3f}\n"

                    item_text = (
                        f"--- Cell #{det_id} ---\n"
                        f"CONCH Top-1: '{it.get('class_key')}' (confidence: {float(it.get('conch_confidence') or 0.0):.2f})\n"
                        f"{morph_ensemble_str}"
                        f"Score Distribution: {json.dumps(it.get('conch_scores', {}))}\n"
                        f"Morphometry: Area={it.get('area', 0)}px, Circularity={it.get('circularity', 1.0)}, MeanIntensity={it.get('mean_intensity', 128)}\n"
                        f"Spatial Figure Hint: {json.dumps(it.get('spatial_label_hint'))}\n"
                        f"Visual Crop of Cell #{det_id}:"
                    )
                    contents_payload.append(item_text)
                    if crop_img is not None:
                        contents_payload.append(crop_img)

                footer_instruction = """
Task:
For each cell above, output your visual adjudication as a valid JSON array of objects:
[
  { "id": "<cell_id>", "class_key": "<class_key_from_available_classes>", "confidence": 0.88, "reasoning": "<concise visual cytological justification e.g. large voluminous nucleus with open chromatin patterns>" }
]
Return ONLY valid JSON.
"""
                contents_payload.append(footer_instruction)

                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=contents_payload,
                    config={"response_mime_type": "application/json"},
                )

                decisions = json.loads(response.text or "[]")
                decision_by_id = {dec.get("id"): dec for dec in decisions if isinstance(dec, dict)}

                for it in chunk:
                    dec = decision_by_id.get(it.get("id"))
                    if dec and dec.get("class_key") in class_map:
                        ck = dec["class_key"]
                        matched_cls = class_map[ck]
                        conch_k = it.get("class_key")
                        conch_c = float(it.get("conch_confidence") or 0.5)
                        g_c = float(dec.get("confidence", 0.85))

                        # Calibrated confidence calculation
                        if ck == conch_k:
                            final_c = min(0.98, max(conch_c, 0.75) + 0.10)
                            reason = f"Consenso (CONCH {conch_c:.2f} + Gemini HD). {dec.get('reasoning', '')}"
                        else:
                            conch_scores = it.get("conch_scores", {})
                            final_c = round(0.65 * g_c + 0.35 * conch_scores.get(ck, 0.30), 4)
                            reason = f"Arbitraje visual Gemini sobre crop HD: {dec.get('reasoning', '')}"

                        it["class_key"] = ck
                        it["class_label"] = matched_cls.get("label", matched_cls.get("name", ck))
                        it["color"] = matched_cls.get("color", "#8b5cf6")
                        it["confidence"] = final_c
                        it["decision_source"] = "gemini_multimodal_arbitration"
                        it["agent_reasoning"] = reason

        except Exception as e:
            logger.warning(f"[FinalClassifierNode] Multimodal visual adjudication skipped or failed: {e}")

    # Step 3: Final validation — ensure EVERY cell has a valid class in class_map
    morphometric_fallback_count = 0
    for d in detections:
        ck = d.get("class_key")
        if not ck or ck not in class_map:
            # Check highest score in conch_scores
            scores = d.get("conch_scores") or d.get("class_scores") or {}
            valid_scores = {k: v for k, v in scores.items() if k in class_map}
            if valid_scores:
                best_k = max(valid_scores, key=valid_scores.get)
                matched_c = class_map[best_k]
                d["class_key"] = best_k
                d["class_label"] = matched_c.get("label", matched_c.get("name", best_k))
                d["color"] = matched_c.get("color", "#8b5cf6")
                d["confidence"] = float(valid_scores[best_k])
                d["decision_source"] = "foundation_conch_argmax"
                d["agent_reasoning"] = f"Assigned to top scoring class '{best_k}'."
            else:
                # If no scores exist, assign the first candidate class
                class_keys = list(class_map.keys())
                assigned_k = class_keys[0] if class_keys else "unclassified"
                matched_c = class_map.get(assigned_k, {"label": assigned_k, "color": "#8b5cf6"})
                d["class_key"] = assigned_k
                d["class_label"] = matched_c.get("label", matched_c.get("name", assigned_k))
                d["color"] = matched_c.get("color", "#8b5cf6")
                d["confidence"] = 0.50
                d["decision_source"] = "default_ontology_assignment"
                d["agent_reasoning"] = "Asignado por clase predeterminada de la ontología."
                morphometric_fallback_count += 1

    # Collect summary stats and build reasoning log
    class_counts: Dict[str, int] = {}
    decision_counts: Dict[str, int] = {}
    for d in detections:
        ck = d.get("class_key", "unclassified")
        ds = d.get("decision_source", "unknown")
        class_counts[ck] = class_counts.get(ck, 0) + 1
        decision_counts[ds] = decision_counts.get(ds, 0) + 1
        final_detections.append(d)
        reasoning_log.append({
            "id": d.get("id"),
            "class_key": d.get("class_key"),
            "class_label": d.get("class_label"),
            "confidence": d.get("confidence"),
            "decision_source": d.get("decision_source"),
            "reasoning": d.get("agent_reasoning"),
        })

    summary = {
        "total_nuclei_classified": len(final_detections),
        "by_class": class_counts,
        "by_decision_source": decision_counts,
        "figure_labels_detected": len(figure_labels),
        "ambiguous_nuclei_adjudicated": len(ambiguous_items),
        "morphometric_fallback_count": morphometric_fallback_count,
    }

    elapsed = time.time() - start_t
    logger.info(
        f"[FinalClassifierNode] Decision breakdown: {decision_counts} | "
        f"Classes: {class_counts} in {elapsed:.3f}s"
    )
    if morphometric_fallback_count > 0:
        logger.warning(
            f"[FinalClassifierNode] ⚠ {morphometric_fallback_count}/{len(final_detections)} "
            f"({morphometric_fallback_count / len(final_detections) * 100:.1f}%) detections fell back to morphometric_heuristic."
        )

    return {
        "final_detections": final_detections,
        "classification_summary": summary,
        "reasoning_log": reasoning_log,
        "execution_metrics": {
            "total_time_seconds": round(elapsed, 3),
            "decision_counts": decision_counts,
            "morphometric_heuristic_count": morphometric_fallback_count,
        },
    }


# ---------------------------------------------------------------------------
# LangGraph Workflow Construction
# ---------------------------------------------------------------------------

def build_histology_graph():
    """Constructs and compiles the multi-agent LangGraph workflow."""
    if StateGraph is None:
        raise ImportError("langgraph package is required to build Histology Graph.")

    workflow = StateGraph(HistologyGraphState)

    # Add agent nodes
    workflow.add_node("ontology_reader", ontology_reader_node)
    workflow.add_node("image_label_detector", image_label_detector_node)
    workflow.add_node("segmentation_cropper", segmentation_cropper_node)
    workflow.add_node("foundation_matcher", foundation_matcher_node)
    workflow.add_node("final_classifier", final_classifier_node)

    # Define execution edges
    workflow.add_edge(START, "ontology_reader")
    workflow.add_edge("ontology_reader", "image_label_detector")
    workflow.add_edge("image_label_detector", "segmentation_cropper")
    workflow.add_edge("segmentation_cropper", "foundation_matcher")
    workflow.add_edge("foundation_matcher", "final_classifier")
    workflow.add_edge("final_classifier", END)

    app = workflow.compile()
    return app


# ---------------------------------------------------------------------------
# High-Level Pipeline Runner
# ---------------------------------------------------------------------------

class HistologyMultiAgentPipeline:
    """Convenient high-level interface for running the multi-agent LangGraph pipeline."""

    _instance: Optional["HistologyMultiAgentPipeline"] = None

    def __init__(self):
        self.app = build_histology_graph()

    @classmethod
    def get_instance(cls) -> "HistologyMultiAgentPipeline":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def run(
        self,
        image: Union[Image.Image, bytes],
        detections: List[Dict[str, Any]],
        ontology_name: Optional[str] = None,
        candidate_classes: Optional[List[Dict[str, Any]]] = None,
        raw_ontology: Optional[List[Dict[str, Any]]] = None,
        user_context_hint: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute the multi-agent pipeline on an image and set of segmented detections.
        """
        img_pil = None
        img_bytes = None

        if isinstance(image, Image.Image):
            img_pil = image
            buf = io.BytesIO()
            image.save(buf, format="JPEG", quality=90)
            img_bytes = buf.getvalue()
        elif isinstance(image, (bytes, bytearray)):
            img_bytes = bytes(image)
            try:
                img_pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            except Exception as e:
                logger.error(f"Failed to open image bytes: {e}")

        img_size = img_pil.size if img_pil is not None else (1024, 1024)

        initial_state: HistologyGraphState = {
            "image_bytes": img_bytes,
            "image_pil": img_pil,
            "image_size": img_size,
            "ontology_name": ontology_name,
            "candidate_classes": candidate_classes or [],
            "raw_ontology": raw_ontology or [],
            "user_context_hint": user_context_hint,
            "detections": detections,
        }

        result_state = self.app.invoke(initial_state)

        return {
            "detections": result_state.get("final_detections", []),
            "detected_figure_labels": result_state.get("detected_figure_labels", []),
            "figure_abbreviations": result_state.get("figure_abbreviations_map", {}),
            "figure_visual_notes": result_state.get("figure_visual_notes", ""),
            "classification_summary": result_state.get("classification_summary", {}),
            "reasoning_log": result_state.get("reasoning_log", []),
            "candidate_classes": result_state.get("candidate_classes", []),
            "execution_metrics": result_state.get("execution_metrics", {}),
        }
