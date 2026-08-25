import os
from dotenv import load_dotenv
load_dotenv()

# Configure PyTorch CUDA memory allocator before importing torch
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import io
import time
import math
import base64
import json
import logging
import warnings
from typing import List, Dict, Any

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import torch
import numpy as np
import cv2

# Suppress non-fatal deprecation warnings from third-party libraries
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from contextlib import asynccontextmanager
from typing import List, Dict, Any, AsyncGenerator, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sam3-backend")

# Try importing SAM3 packages
try:
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
except ImportError as e:
    logger.error(f"Error importing SAM3 modules: {e}")
    raise

# Try importing Ultralytics SAM 3 Semantic Predictor
try:
    from ultralytics.models.sam import SAM3SemanticPredictor
except ImportError:
    SAM3SemanticPredictor = None
    logger.warning("Ultralytics SAM3SemanticPredictor not found.")

# Import Roboflow integration
from roboflow_integration import (
    check_connection as rf_check_connection,
    get_roboflow_models_and_versions,
    build_coco_json,
    build_multi_image_coco,
    upload_dataset_to_roboflow,
    trigger_training,
    export_dataset_version,
)

# Import PDF ontology pipeline
from pdf_ontology import (
    extract_pdf_content,
    generate_ontology_with_gemini,
    build_ontology_document,
    save_ontology,
    load_ontology,
    list_ontologies,
    update_ontology_structures,
    get_ontology_prompts,
    get_pdf_image_path,
    get_extracted_text,
    get_pdf_metadata,
    add_pdf_image,
    update_pdf_image_metadata,
    delete_pdf_image,
    merge_ontology_structures,
    PDF_IMAGES_DIR,
)

# Import Pathology Foundation Models (CONCH & UNI)
from pathology_models import (
    get_pathology_models_status,
    classify_detections_with_conch,
    extract_detection_embeddings_uni,
    discriminate_and_cluster_with_pathology_models,
    group_detections_by_class,
)

# Import Dynamic Multimodal LLM Vision Assistant (Gemini 2.5 Flash)
from gemini_vision import (
    refine_prompt_multimodal,
    discover_visual_primitives_from_image,
)

# Global variables for model and processor
processor = None
sam3_semantic_predictor = None
device = "cuda" if torch.cuda.is_available() else "cpu"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    global processor, sam3_semantic_predictor
    logger.info("Initializing SAM 3 models...")
    start_time = time.time()

    # Enable bfloat16 autocast globally, as recommended by the official SAM3 notebooks
    torch.autocast(device_type=device, dtype=torch.bfloat16).__enter__()
    torch.inference_mode().__enter__()

    # 1. Initialize Ultralytics SAM3SemanticPredictor if available
    if SAM3SemanticPredictor is not None:
        try:
            model_file = "sam3.pt"
            if not os.path.exists(model_file):
                hf_cache = os.path.expanduser("~/.cache/huggingface/hub/models--facebook--sam3/snapshots")
                if os.path.exists(hf_cache):
                    for root, _, files in os.walk(hf_cache):
                        if "sam3.pt" in files:
                            model_file = os.path.join(root, "sam3.pt")
                            break
            if os.path.exists(model_file):
                sam3_semantic_predictor = SAM3SemanticPredictor(overrides={
                    "conf": 0.15,
                    "task": "segment",
                    "mode": "predict",
                    "model": model_file,
                    "quantize": 16,
                    "save": False,
                })
                logger.info(f"Ultralytics SAM3SemanticPredictor loaded successfully from '{model_file}'.")
        except Exception as e:
            logger.warning(f"Could not load Ultralytics SAM3SemanticPredictor: {e}")
            sam3_semantic_predictor = None

    # 2. Initialize Meta SAM 3 image model & processor
    try:
        model = build_sam3_image_model(device=device, version="sam3.1")
        # Initialize with flexible threshold (will be dynamically adjusted per request)
        processor = Sam3Processor(model, device=device, confidence_threshold=0.05)
        logger.info(f"SAM 3.1 Processor loaded successfully on {device} in {time.time() - start_time:.2f} seconds.")
    except Exception as e:
        logger.error(f"Failed to load SAM 3.1 model: {e}")
        processor = None

    yield

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


app = FastAPI(
    title="SAM 3 Histological & Universal Segmenter API",
    description="Backend for Segment Anything Model v3 automated cell & structure segmentation with Roboflow integration",
    lifespan=lifespan,
)

# Enable CORS for frontend communication (Astro usually runs on 4321)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)


MAX_INFERENCE_DIM = 1440  # Max dimension for SAM3 inference interpolation to prevent VRAM OOM

def clean_value(val: Any) -> Any:
    """Helper to convert tensors/arrays/scalars to standard Python serializable types."""
    if hasattr(val, "tolist"):
        return val.tolist()
    if hasattr(val, "item"):
        return val.item()
    if isinstance(val, (np.ndarray, list)):
        return [clean_value(x) for x in val]
    return val


def mask_to_polygons(mask, scale_x: float = 1.0, scale_y: float = 1.0, simplify_tolerance: float = 1.5) -> List[List[float]]:
    """
    Convert a binary mask to polygon contours (COCO segmentation format).

    Returns a list of polygon coordinate lists, each polygon as [x1,y1,x2,y2,...,xn,yn].
    Coordinates are rescaled to original image dimensions.
    """
    # Convert mask to numpy uint8
    if hasattr(mask, "cpu"):
        if hasattr(mask, "float"):
            mask_np = mask.float().cpu().numpy()
        else:
            mask_np = mask.cpu().numpy()
    elif isinstance(mask, np.ndarray):
        mask_np = mask
    else:
        mask_np = np.array(mask)

    # Ensure 2D binary mask
    if mask_np.ndim > 2:
        mask_np = mask_np.squeeze()
    mask_uint8 = (mask_np > 0).astype(np.uint8) * 255

    # Find contours
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    polygons = []
    for contour in contours:
        # Skip tiny contours (noise)
        if len(contour) < 3:
            continue

        # Simplify polygon to reduce point count (important for histology with many cells)
        epsilon = simplify_tolerance
        approx = cv2.approxPolyDP(contour, epsilon, True)

        if len(approx) < 3:
            continue

        # Flatten to COCO format [x1,y1,x2,y2,...] and rescale
        polygon = []
        for point in approx:
            px, py = point[0]
            polygon.append(float(px * scale_x))
            polygon.append(float(py * scale_y))

        # Must have at least 6 values (3 points)
        if len(polygon) >= 6:
            polygons.append(polygon)

    return polygons


def _extract_detections(
    output,
    umbral: float,
    scale_x: float = 1.0,
    scale_y: float = 1.0,
    include_polygons: bool = True,
    img_w: Optional[int] = None,
    img_h: Optional[int] = None,
    iou_threshold: float = 0.5,
    **kwargs
) -> list:
    """
    Extract detections from processor output state dict, apply NMS to remove
    overlapping/duplicate masks, filter whole-frame artifacts, and rescale boxes
    to original image coordinates.
    """
    masks = output.get("masks")
    boxes = output.get("boxes")
    scores = output.get("scores")

    if masks is None or boxes is None or scores is None:
        return []

    num_raw = len(scores)
    if num_raw == 0:
        return []

    clean_boxes = clean_value(boxes)
    clean_scores = clean_value(scores)

    # Ensure flat list
    if isinstance(clean_scores, (float, int)):
        clean_scores = [clean_scores]
    if isinstance(clean_boxes, list) and len(clean_boxes) > 0 and not isinstance(clean_boxes[0], list):
        clean_boxes = [clean_boxes]

    # Filter indices by threshold
    valid_indices = [i for i, s in enumerate(clean_scores) if float(s) >= umbral]
    if not valid_indices:
        return []

    # Apply Non-Maximum Suppression (NMS) to eliminate duplicate/overlapping boxes
    if len(valid_indices) > 1:
        try:
            import torchvision.ops
            boxes_tensor = torch.tensor([clean_boxes[i] for i in valid_indices], dtype=torch.float32)
            scores_tensor = torch.tensor([float(clean_scores[i]) for i in valid_indices], dtype=torch.float32)
            keep = torchvision.ops.nms(boxes_tensor, scores_tensor, iou_threshold=iou_threshold)
            valid_indices = [valid_indices[k] for k in keep.tolist()]
        except Exception as nms_err:
            logger.warning(f"NMS filtering skipped: {nms_err}")

    total_img_area = (float(img_w) * float(img_h)) if (img_w and img_h and img_w > 0 and img_h > 0) else None

    detections = []
    for i in valid_indices:
        s = float(clean_scores[i])
        box = clean_boxes[i]

        # Rescale box from inference dimensions back to original image dimensions
        x1 = float(box[0] * scale_x)
        y1 = float(box[1] * scale_y)
        x2 = float(box[2] * scale_x)
        y2 = float(box[3] * scale_y)

        if img_w is not None and img_w > 0:
            x1 = max(0.0, min(float(img_w), x1))
            x2 = max(0.0, min(float(img_w), x2))
        if img_h is not None and img_h > 0:
            y1 = max(0.0, min(float(img_h), y1))
            y2 = max(0.0, min(float(img_h), y2))

        bw = max(0.0, x2 - x1)
        bh = max(0.0, y2 - y1)
        box_area = bw * bh

        # Filter out macro-boxes covering almost the entire image (> 85% area),
        # which occur when SAM 3 selects the whole slide/frame instead of cells
        if total_img_area and total_img_area > 0 and (box_area / total_img_area) > 0.85:
            continue

        rescaled_box = [x1, y1, x2, y2]

        detection = {
            "box": rescaled_box,
            "bbox": [x1, y1, bw, bh],
            "score": round(s, 4)
        }

        # Extract polygon from mask
        if include_polygons and masks is not None:
            try:
                mask_i = masks[i]
                polys = mask_to_polygons(mask_i, scale_x, scale_y)
                if polys:
                    detection["segmentation"] = polys
                    if hasattr(mask_i, "cpu"):
                        if hasattr(mask_i, "float"):
                            area = float(mask_i.float().cpu().numpy().sum()) * scale_x * scale_y
                        else:
                            area = float(mask_i.cpu().numpy().sum()) * scale_x * scale_y
                    else:
                        area = float(np.array(mask_i).sum()) * scale_x * scale_y
                    detection["area"] = round(area, 2)
            except Exception as e:
                logger.warning(f"Failed to extract polygon for detection {i}: {e}")

        detections.append(detection)

    return detections


def _extract_detections_ultralytics(
    results,
    elements: List[str],
    img_w: int,
    img_h: int,
    conf_thresh: float = 0.15,
) -> List[Dict[str, Any]]:
    """Extract detections and polygons from Ultralytics SAM3SemanticPredictor results."""
    detections = []
    det_counter = 1
    total_img_area = float(img_w) * float(img_h) if (img_w > 0 and img_h > 0) else None

    for r in results:
        masks = getattr(r, "masks", None)
        if masks is None or len(masks) == 0:
            continue

        masks_xy = masks.xy
        boxes = getattr(r, "boxes", None)
        names = getattr(r, "names", elements)

        cls_arr = boxes.cls.cpu().numpy().astype(int) if boxes is not None and hasattr(boxes, "cls") and boxes.cls is not None else [0] * len(masks_xy)
        conf_arr = boxes.conf.cpu().numpy() if boxes is not None and hasattr(boxes, "conf") and boxes.conf is not None else [conf_thresh] * len(masks_xy)
        xyxy_arr = boxes.xyxy.cpu().numpy() if boxes is not None and hasattr(boxes, "xyxy") and boxes.xyxy is not None else None

        for i in range(len(masks_xy)):
            score = float(conf_arr[i])
            if score < conf_thresh:
                continue

            poly_pts = masks_xy[i]
            if len(poly_pts) < 3:
                continue

            poly_flat = []
            for pt in poly_pts:
                px = max(0.0, min(float(img_w), float(pt[0])))
                py = max(0.0, min(float(img_h), float(pt[1])))
                poly_flat.extend([round(px, 2), round(py, 2)])

            if len(poly_flat) < 6:
                continue

            cls_idx = int(cls_arr[i])
            if isinstance(names, list) and cls_idx < len(names):
                cls_name = names[cls_idx]
            elif isinstance(names, dict):
                cls_name = names.get(cls_idx, str(cls_idx))
            else:
                cls_name = str(cls_idx)

            if xyxy_arr is not None and i < len(xyxy_arr):
                x1 = max(0.0, min(float(img_w), float(xyxy_arr[i][0])))
                y1 = max(0.0, min(float(img_h), float(xyxy_arr[i][1])))
                x2 = max(0.0, min(float(img_w), float(xyxy_arr[i][2])))
                y2 = max(0.0, min(float(img_h), float(xyxy_arr[i][3])))
            else:
                x_pts = poly_flat[0::2]
                y_pts = poly_flat[1::2]
                x1, y1, x2, y2 = min(x_pts), min(y_pts), max(x_pts), max(y_pts)

            bw = max(0.0, x2 - x1)
            bh = max(0.0, y2 - y1)
            box_area = bw * bh

            if total_img_area and (box_area / total_img_area) > 0.90:
                continue

            detections.append({
                "id": f"det_{det_counter}",
                "category_name": cls_name,
                "score": round(score, 4),
                "box": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
                "bbox": [round(x1, 2), round(y1, 2), round(bw, 2), round(bh, 2)],
                "segmentation": [poly_flat],
                "area": round(box_area, 2),
            })
            det_counter += 1

    return detections


def _prepare_image_for_inference(pil_image: Image.Image):
    """
    Prepares image for SAM3 inference.

    IMPORTANT: Sam3Processor already resizes to its native 1008×1008 internally
    with proper normalization. We do NOT pre-resize here to avoid double-resize
    quality loss. We only record the original dimensions so that output boxes
    and masks can be mapped back to original coordinates.

    Returns (pil_image, orig_width, orig_height, scale_x, scale_y)
    """
    orig_w, orig_h = pil_image.size
    return pil_image, orig_w, orig_h, 1.0, 1.0


@app.get("/api/health")
def health_check() -> Dict[str, Any]:
    return {
        "status": "ok" if (processor is not None or sam3_semantic_predictor is not None) else "model_not_loaded",
        "device": device,
        "model": "sam3",
        "ultralytics_sam3": sam3_semantic_predictor is not None,
    }

@app.get("/api/prompts")
def get_prompts() -> Dict[str, Any]:
    """Returns the list of available saved ontologies and their dynamic prompts."""
    ontologies = list_ontologies()
    return {"ontologies": ontologies}


@app.post("/api/segment")
async def segment_image(
    image: UploadFile = File(...),
    prompt: Optional[str] = Form(None),
    elements: Optional[str] = Form(None),
    umbral: float = Form(0.25),
    ontology_name: Optional[str] = Form(None),
) -> Dict[str, Any]:
    """
    Zero-shot semantic segmentation endpoint with SAM 3 (Ultralytics / Meta SAM 3).
    Supports single or multiple concepts separated by commas (e.g. 'person, glasses' or 'cell, nucleus').
    """
    if sam3_semantic_predictor is None and processor is None:
        raise HTTPException(status_code=503, detail="SAM 3 model is not loaded.")

    query_text = (elements if elements and elements.strip() else (prompt if prompt and prompt.strip() else "cell nucleus")).strip()
    logger.info(f"Received segment request: query='{query_text}', umbral={umbral}")
    start_time = time.time()

    try:
        contents = await image.read()
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
        width, height = pil_image.size

        # Parse elements list (split by comma)
        concept_list = [c.strip() for c in query_text.split(",") if c.strip()]
        if not concept_list:
            concept_list = ["cell nucleus"]

        detections = []
        # Priority 1: Ultralytics SAM3SemanticPredictor (Fast multi-concept zero-shot)
        if sam3_semantic_predictor is not None:
            try:
                if hasattr(sam3_semantic_predictor, "args") and sam3_semantic_predictor.args is not None:
                    sam3_semantic_predictor.args.conf = float(umbral)
                sam3_semantic_predictor.set_image(pil_image)
                results = sam3_semantic_predictor(text=concept_list)
                detections = _extract_detections_ultralytics(
                    results=results,
                    elements=concept_list,
                    img_w=width,
                    img_h=height,
                    conf_thresh=float(umbral)
                )
            except Exception as ultra_err:
                logger.warning(f"SAM3SemanticPredictor failed, falling back to processor: {ultra_err}")
                detections = []

        # Priority 2: Fallback to Sam3Processor if predictor was unavailable or returned empty due to error
        if not detections and processor is not None:
            inf_image, w, h, sx, sy = _prepare_image_for_inference(pil_image)
            processor.confidence_threshold = max(0.01, min(0.35, float(umbral)))
            inference_state = processor.set_image(inf_image)
            for c_text in concept_list:
                output = processor.set_text_prompt(state=inference_state, prompt=c_text)
                c_dets = _extract_detections(output, umbral, sx, sy, img_w=width, img_h=height)
                for d in c_dets:
                    d["category_name"] = c_text
                detections.extend(c_dets)

        # Group detections by category_name for frontend compatibility
        palette = ["#3b82f6", "#10b981", "#ef4444", "#f59e0b", "#8b5cf6", "#ec4899", "#06b6d4", "#14b8a6", "#f97316", "#84cc16"]
        groups_map = {}
        for d in detections:
            cat = d.get("category_name", "Objeto")
            if cat not in groups_map:
                idx = len(groups_map)
                color = palette[idx % len(palette)]
                key = cat.lower().replace(" ", "_")
                groups_map[cat] = {
                    "key": key,
                    "label": cat,
                    "color": color,
                    "detections": []
                }
            groups_map[cat]["detections"].append(d)

        groups = list(groups_map.values())

        inference_time = time.time() - start_time
        logger.info(f"Found {len(detections)} detections for concepts {concept_list} in {inference_time:.2f}s")

        return {
            "width": width,
            "height": height,
            "detections": detections,
            "groups": groups,
            "total_detections": len(detections),
            "inference_time_seconds": round(inference_time, 2),
            "prompt": query_text,
            "elements": concept_list,
            "umbral": umbral,
        }

    except Exception as e:
        logger.error(f"Error during segmentation: {e}", exc_info=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


@app.post("/api/segment-auto")
async def segment_auto(
    image: UploadFile = File(...),
    umbral: float = Form(0.05),
    custom_prompt: Optional[str] = Form(None),
    ontology_name: Optional[str] = Form(None),
) -> Dict[str, Any]:
    """
    Exhaustive automatic multi-prompt segmentation and semantic discrimination.
    Prompts and classes are 100% dynamically inferred from the active ontology or
    discovered on-the-fly with Gemini Vision multimodal analysis, and discriminated
    with MahmoodLab/UNI (1024-dim) and MahmoodLab/CONCH (512-dim).
    """
    if processor is None:
        raise HTTPException(status_code=503, detail="SAM 3 model is not loaded.")

    logger.info(f"Received segment-auto request: umbral={umbral}, custom_prompt='{custom_prompt}', ontology_name='{ontology_name}'")
    start_time = time.time()

    try:
        contents = await image.read()
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
        inf_image, width, height, scale_x, scale_y = _prepare_image_for_inference(pil_image)

        # 1. Dynamically resolve prompts to run
        prompts_to_run: List[Dict[str, Any]] = []

        # A. Priority 1: From active ontology
        if ontology_name and ontology_name.strip():
            ont_prompts = get_ontology_prompts(ontology_name.strip())
            if ont_prompts:
                prompts_to_run = list(ont_prompts)
                logger.info(f"Loaded {len(prompts_to_run)} dynamic prompts from ontology '{ontology_name}'")

        # B. Priority 2: If no ontology is available, discover visual structures dynamically with Gemini Vision
        if not prompts_to_run:
            logger.info("No ontology specified or empty. Using Gemini Vision for dynamic visual structure discovery...")
            prompts_to_run = discover_visual_primitives_from_image(pil_image)

        # C. User custom prompt refinement (multimodal)
        if custom_prompt and custom_prompt.strip():
            cp_text = custom_prompt.strip()
            refined_cp = refine_prompt_multimodal(pil_image, cp_text, prompts_to_run)
            prompts_to_run.insert(0, {
                "key": f"custom_{len(prompts_to_run) + 1}",
                "prompt": refined_cp,
                "label": cp_text,
                "color": "#a855f7",
            })

        # Ensure we have visual prompts to run
        if not prompts_to_run:
            # Dynamic fallback
            prompts_to_run = [
                {"key": "cell_nucleus", "prompt": "dark round cell nucleus", "label": "Núcleos celulares", "color": "#f43f5e"},
                {"key": "tissue_structure", "prompt": "stained tissue structure", "label": "Estructuras tisulares", "color": "#38bdf8"},
                {"key": "connective_fiber", "prompt": "connective tissue fiber", "label": "Fibras de tejido", "color": "#8b5cf6"},
                {"key": "lumen_space", "prompt": "empty cavity lumen", "label": "Luz / Cavidad", "color": "#6366f1"},
            ]

        # 2. SAM 3.1 Inference with adaptive sensitivity
        proc_thresh = max(0.01, min(0.35, float(umbral)))
        processor.confidence_threshold = proc_thresh

        inference_state = processor.set_image(inf_image)
        all_raw_detections = []

        for prompt_info in prompts_to_run:
            prompt_text = prompt_info.get("prompt", "")
            prompt_key = prompt_info.get("key", "struct")
            prompt_label = prompt_info.get("label", prompt_info.get("name", prompt_key))
            prompt_color = prompt_info.get("color", "#8b5cf6")

            if not prompt_text:
                continue

            processor.reset_all_prompts(inference_state)
            output = processor.set_text_prompt(state=inference_state, prompt=prompt_text)

            detections = _extract_detections(
                output,
                umbral=umbral,
                scale_x=scale_x,
                scale_y=scale_y,
                include_polygons=True,
                img_w=width,
                img_h=height,
                iou_threshold=0.65,
            )

            for d in detections:
                d_copy = dict(d)
                d_copy["initial_class_key"] = prompt_key
                d_copy["initial_label"] = prompt_label
                d_copy["color"] = prompt_color
                all_raw_detections.append(d_copy)

        logger.info(f"SAM 3.1 generated {len(all_raw_detections)} candidate detections across {len(prompts_to_run)} dynamic prompts")

        # 3. Global IoU NMS to deduplicate overlapping masks from different prompts
        filtered_candidates = []
        if len(all_raw_detections) > 1:
            try:
                import torchvision.ops
                boxes = torch.tensor([d["box"] for d in all_raw_detections], dtype=torch.float32)
                scores = torch.tensor([d.get("score", 0.5) for d in all_raw_detections], dtype=torch.float32)
                keep_indices = torchvision.ops.nms(boxes, scores, iou_threshold=0.60).tolist()
                filtered_candidates = [all_raw_detections[idx] for idx in keep_indices]
            except Exception as nms_err:
                logger.warning(f"Global NMS skipped: {nms_err}")
                filtered_candidates = all_raw_detections
        else:
            filtered_candidates = all_raw_detections

        logger.info(f"Retained {len(filtered_candidates)} distinct candidate instances after spatial deduplication")

        # 4. Discrimination and Semantic Clustering with UNI (1024-dim) and CONCH (512-dim)
        classified_detections = []
        if filtered_candidates:
            try:
                classified_detections = discriminate_and_cluster_with_pathology_models(
                    image=pil_image,
                    detections=filtered_candidates,
                    candidate_classes=prompts_to_run,
                    temperature=0.05,
                )
            except Exception as disc_err:
                logger.warning(f"Pathology discrimination fallback: {disc_err}", exc_info=True)
                classified_detections = filtered_candidates

        # 5. Group into structured categories for UI rendering
        all_groups = group_detections_by_class(classified_detections, candidate_classes=prompts_to_run)
        total_detections = sum(g["count"] for g in all_groups)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        inference_time = time.time() - start_time
        logger.info(f"AUTO: Exhaustive segmentation finished with {total_detections} detections across {len(all_groups)} categories in {inference_time:.2f}s")

        return {
            "width": width,
            "height": height,
            "groups": all_groups,
            "total_detections": total_detections,
            "inference_time_seconds": round(inference_time, 2),
            "umbral": umbral,
        }

    except Exception as e:
        logger.error(f"Error during auto-segmentation: {e}", exc_info=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


@app.post("/api/segment-point")
async def segment_point(
    image: UploadFile = File(...),
    x: float = Form(...),
    y: float = Form(...),
    prompt: str = Form("object"),
    umbral: float = Form(0.05)
) -> Dict[str, Any]:
    """
    Interactive click-to-segment endpoint.
    Extracts the precise polygon segmentation mask around point (x, y) on the image.
    """
    if processor is None:
        raise HTTPException(status_code=503, detail="SAM 3 model is not loaded.")

    logger.info(f"Received segment-point request at x={x}, y={y}, prompt='{prompt}'")

    try:
        contents = await image.read()
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
        orig_w, orig_h = pil_image.size

        # Convert normalized coordinates if x, y are <= 1.0
        px = int(x * orig_w) if x <= 1.0 else int(x)
        py = int(y * orig_h) if y <= 1.0 else int(y)

        # 1. Try text prompt on full image first
        inf_image, width, height, scale_x, scale_y = _prepare_image_for_inference(pil_image)
        state = processor.set_image(inf_image)
        output = processor.set_text_prompt(state=state, prompt=prompt if prompt else "object")
        detections = _extract_detections(output, umbral, scale_x, scale_y, include_polygons=True)

        matched_det = None
        min_dist = float("inf")

        for det in detections:
            bbox = det["bbox"]  # [x, y, w, h]
            bx, by, bw, bh = bbox
            if bx <= px <= bx + bw and by <= py <= by + bh:
                matched_det = det
                break
            cx, cy = bx + bw / 2, by + bh / 2
            dist = math.hypot(px - cx, py - cy)
            if dist < min_dist:
                min_dist = dist
                matched_det = det

        # 2. Localized crop fallback around (px, py) if no detection matched
        if matched_det is None or min_dist > 150:
            crop_size = min(max(orig_w, orig_h) // 3, 300)
            left = max(0, px - crop_size // 2)
            top = max(0, py - crop_size // 2)
            right = min(orig_w, left + crop_size)
            bottom = min(orig_h, top + crop_size)

            crop_img = pil_image.crop((left, top, right, bottom))
            crop_inf, cw, ch, c_scale_x, c_scale_y = _prepare_image_for_inference(crop_img)

            c_state = processor.set_image(crop_inf)
            c_output = processor.set_text_prompt(state=c_state, prompt=prompt if prompt else "object")
            c_dets = _extract_detections(c_output, 0.01, c_scale_x, c_scale_y, include_polygons=True)

            if c_dets:
                best_c = c_dets[0]
                bx, by, bw, bh = best_c["bbox"]
                best_c["bbox"] = [bx + left, by + top, bw, bh]
                if "segmentation" in best_c:
                    new_seg = []
                    for poly in best_c["segmentation"]:
                        new_poly = []
                        for i in range(0, len(poly), 2):
                            new_poly.extend([poly[i] + left, poly[i+1] + top])
                        new_seg.append(new_poly)
                    best_c["segmentation"] = new_seg
                matched_det = best_c

        # Fallback bounding box polygon if no detection returned
        if matched_det is None:
            box_r = 45
            matched_det = {
                "class_id": 1,
                "confidence": 0.95,
                "bbox": [max(0, px - box_r), max(0, py - box_r), box_r * 2, box_r * 2],
                "segmentation": [[
                    max(0, px - box_r), max(0, py - box_r),
                    min(orig_w, px + box_r), max(0, py - box_r),
                    min(orig_w, px + box_r), min(orig_h, py + box_r),
                    max(0, px - box_r), min(orig_h, py + box_r)
                ]]
            }

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return {
            "success": True,
            "click_x": px,
            "click_y": py,
            "detection": matched_det,
        }

    except Exception as e:
        logger.error(f"Error in segment_point: {e}", exc_info=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise HTTPException(status_code=500, detail=str(e))


# ======================== PDF Ontology Endpoints ========================

@app.post("/api/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Upload a PDF and extract text + embedded images.
    Returns a preview of the extracted content (text summary, image list).
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Solo se aceptan archivos PDF.")

    try:
        pdf_bytes = await file.read()
        result = extract_pdf_content(pdf_bytes, file.filename)

        # Return a summary (not the full text, to keep response light)
        text_preview = result["text"][:2000] + ("..." if len(result["text"]) > 2000 else "")

        return {
            "success": True,
            "pdf_id": result["pdf_id"],
            "filename": result["filename"],
            "total_pages": result["total_pages"],
            "total_images": result["total_images"],
            "text_length": result["text_length"],
            "text_preview": text_preview,
            "images": result["images"],
        }
    except ImportError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"PDF extraction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error al procesar PDF: {str(e)}")


@app.post("/api/generate-ontology")
async def generate_ontology(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    """
    Generate a domain ontology from previously extracted PDF text.

    Expects:
      - pdf_id: str (from upload-pdf response)
      - filename: str (original PDF filename)
      - text: str (full extracted text — frontend sends it back)
      - images: list (extracted images metadata)
      - domain_name: str (optional, auto-derived from filename if omitted)
      - merge_into_ontology: str (optional, name of existing ontology to merge into)
    """
    gemini_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not gemini_key:
        raise HTTPException(
            status_code=500,
            detail="GEMINI_API_KEY no está configurada en .env"
        )

    pdf_id = payload.get("pdf_id", "unknown")
    text = payload.get("text", "")
    filename = payload.get("filename", "unknown.pdf")
    images = payload.get("images", [])
    merge_into = payload.get("merge_into_ontology", "").strip() or None

    # Always prefer full cached text extracted from PDF file if available
    cached_text = get_extracted_text(pdf_id)
    if cached_text:
        text = cached_text

    try:
        structures = generate_ontology_with_gemini(
            extracted_text=text,
            api_key=gemini_key,
            model_name="gemini-2.5-flash",
            pdf_id=pdf_id,
        )

        merge_stats: Optional[Dict[str, Any]] = None

        if merge_into:
            # Incremental merge into an existing ontology
            existing = load_ontology(merge_into)
            if existing:
                existing_count = len(existing.get("structures", []))
                ontology = merge_ontology_structures(
                    existing_ontology=existing,
                    new_structures=structures,
                    new_pdf_id=pdf_id,
                    new_filename=filename,
                    new_images=images,
                )
                new_count = len(ontology.get("structures", []))
                merge_stats = {
                    "merged_into": merge_into,
                    "previous_structures": existing_count,
                    "new_structures_from_pdf": len(structures),
                    "total_after_merge": new_count,
                    "added": new_count - existing_count,
                }
                logger.info(
                    f"Merged {len(structures)} structures from {filename} "
                    f"into '{merge_into}': {existing_count} → {new_count}"
                )
            else:
                # Target ontology not found — create as new with requested name
                logger.warning(
                    f"Ontology '{merge_into}' not found for merge, creating new."
                )
                ontology = build_ontology_document(
                    pdf_id=pdf_id,
                    filename=filename,
                    structures=structures,
                    extracted_images=images,
                    domain_name=merge_into,
                )
        else:
            # Create a brand-new ontology
            ontology = build_ontology_document(
                pdf_id=pdf_id,
                filename=filename,
                structures=structures,
                extracted_images=images,
                domain_name=payload.get("domain_name"),
            )

        filepath = save_ontology(ontology)

        result: Dict[str, Any] = {
            "success": True,
            "ontology": ontology,
            "saved_to": filepath,
        }
        if merge_stats:
            result["merge_stats"] = merge_stats

        return result

    except ImportError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except json.JSONDecodeError as e:
        logger.error(f"LLM returned invalid JSON: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"El LLM devolvió JSON inválido. Intentá de nuevo."
        )
    except Exception as e:
        logger.error(f"Ontology generation error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error generando ontología: {str(e)}")


@app.get("/api/ontologies")
def get_ontologies() -> Dict[str, Any]:
    """List all saved ontologies."""
    return {"ontologies": list_ontologies()}


@app.get("/api/ontology/{name}")
def get_ontology(name: str) -> Dict[str, Any]:
    """Get a specific ontology by domain name."""
    ontology = load_ontology(name)
    if ontology is None:
        raise HTTPException(status_code=404, detail=f"Ontología '{name}' no encontrada.")
    return ontology


@app.put("/api/ontology/{name}")
async def update_ontology(name: str, payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    """
    Update the structures of a saved ontology.
    Expects { "structures": [...] } in body.
    """
    structures = payload.get("structures")
    if structures is None or not isinstance(structures, list):
        raise HTTPException(
            status_code=400,
            detail="Se requiere un array 'structures' en el body."
        )

    updated = update_ontology_structures(name, structures)
    if updated is None:
        raise HTTPException(status_code=404, detail=f"Ontología '{name}' no encontrada.")

    return {"success": True, "ontology": updated}


@app.get("/api/pdf-images/{pdf_id}")
def get_pdf_images(pdf_id: str) -> Dict[str, Any]:
    """Get the current list of images and metadata for a PDF (CRUD: Read)."""
    meta = get_pdf_metadata(pdf_id)
    if not meta:
        raise HTTPException(status_code=404, detail="PDF no encontrado.")
    return {"success": True, "pdf_id": pdf_id, "images": meta.get("images", [])}


@app.get("/api/pdf-image/{pdf_id}/{filename}")
async def serve_pdf_image(pdf_id: str, filename: str):
    """Serve an extracted PDF image for the frontend."""
    from fastapi.responses import FileResponse

    path = get_pdf_image_path(pdf_id, filename)
    if path is None:
        raise HTTPException(
            status_code=404,
            detail="Imagen no encontrada.",
            headers={"Access-Control-Allow-Origin": "*"},
        )
    return FileResponse(
        path,
        media_type="image/png",
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, HEAD, OPTIONS",
            "Access-Control-Allow-Headers": "*",
            "Cache-Control": "public, max-age=3600",
        },
    )


@app.post("/api/pdf-image/{pdf_id}/upload")
async def upload_custom_pdf_image(
    pdf_id: str,
    file: UploadFile = File(...),
    caption: str = Form(None)
) -> Dict[str, Any]:
    """
    Upload a new image to an existing PDF collection (CRUD: Create).
    """
    try:
        contents = await file.read()
        img_info = add_pdf_image(pdf_id, contents, file.filename or "image.png", caption)
        return {"success": True, "image": img_info}
    except Exception as e:
        logger.error(f"Error uploading image to PDF {pdf_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/api/pdf-image/{pdf_id}/{filename}")
async def update_pdf_image(
    pdf_id: str,
    filename: str,
    payload: Dict[str, Any] = Body(...)
) -> Dict[str, Any]:
    """
    Update caption/label of a PDF image (CRUD: Update).
    """
    caption = payload.get("caption")
    label = payload.get("label")
    updated = update_pdf_image_metadata(pdf_id, filename, caption=caption, label=label)
    if not updated:
        raise HTTPException(status_code=404, detail="Imagen no encontrada.")
    return {"success": True, "image": updated}


@app.delete("/api/pdf-image/{pdf_id}/{filename}")
def delete_extracted_pdf_image(pdf_id: str, filename: str) -> Dict[str, Any]:
    """
    Delete an image from a PDF collection (CRUD: Delete).
    """
    deleted = delete_pdf_image(pdf_id, filename)
    if not deleted:
        raise HTTPException(status_code=404, detail="Imagen no encontrada o no se pudo eliminar.")
    return {"success": True, "filename": filename}


# ======================== Pathology Foundation Models (CONCH & UNI) ========================

@app.get("/api/pathology-models-status")
def pathology_models_status() -> Dict[str, Any]:
    """Check availability and device status for CONCH and UNI foundation models."""
    return get_pathology_models_status()


@app.post("/api/classify-detections-conch")
async def classify_conch(
    image: UploadFile = File(...),
    detections: str = Form(...),
    classes: str = Form(...),
    temperature: float = Form(0.05),
) -> Dict[str, Any]:
    """
    Run Zero-Shot Pathology Classification with CONCH on segmented crops.

    Expects:
      - image: original image file
      - detections: JSON string representing list of detection objects
      - classes: JSON string representing candidate classes [{ key, prompt, label, color }]
      - temperature: float (softmax scaling factor)
    """
    try:
        contents = await image.read()
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")

        detections_list = json.loads(detections)
        classes_list = json.loads(classes)

        if not isinstance(detections_list, list) or not isinstance(classes_list, list):
            raise HTTPException(status_code=400, detail="Formato JSON inválido para detections o classes.")

        classified = classify_detections_with_conch(
            image=pil_image,
            detections=detections_list,
            candidate_classes=classes_list,
            temperature=temperature,
        )

        return {
            "success": True,
            "total_classified": len(classified),
            "detections": classified,
        }
    except Exception as e:
        logger.error(f"Error in CONCH zero-shot classification: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/extract-features-uni")
async def extract_uni_features(
    image: UploadFile = File(...),
    detections: str = Form(...),
) -> Dict[str, Any]:
    """
    Extract 1024-dim UNI pathology embeddings for all detection crops.
    """
    try:
        contents = await image.read()
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")

        detections_list = json.loads(detections)
        if not isinstance(detections_list, list):
            raise HTTPException(status_code=400, detail="Formato JSON inválido para detections.")

        embeddings = extract_detection_embeddings_uni(
            image=pil_image,
            detections=detections_list,
        )

        return {
            "success": True,
            "total_extracted": len(embeddings),
            "embedding_dim": 1024,
            "embeddings": embeddings,
        }
    except Exception as e:
        logger.error(f"Error in UNI feature extraction: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ======================== Roboflow Integration Endpoints ========================

@app.get("/api/roboflow-status")
def roboflow_status() -> Dict[str, Any]:
    """Check Roboflow connection and list project info."""
    return rf_check_connection()


@app.post("/api/export-coco")
async def export_coco(payload: Dict[str, Any] = Body(...)) -> JSONResponse:
    """
    Generate a COCO JSON from the curated annotations.

    Accepts either:
    - A single image payload (has "image_filename" key)
    - A multi-image payload (has "images" key with a list)
    """
    try:
        if "images" in payload:
            coco = build_multi_image_coco(payload["images"])
        else:
            coco = build_coco_json(payload)

        return JSONResponse(content=coco)
    except Exception as e:
        logger.error(f"COCO export error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/upload-roboflow")
async def upload_roboflow(
    annotations: str = Form(default=""),
    images: List[UploadFile] = File(default=[])
) -> Dict[str, Any]:
    """
    Upload annotated images to Roboflow.

    - annotations: JSON string with the multi-image annotation payload
    - images: list of image files
    """
    logger.info(f"Received upload-roboflow request: {len(images)} files attached, annotations len={len(annotations)}")

    if not annotations or not annotations.strip():
        logger.error("Upload error: annotations payload is empty")
        raise HTTPException(status_code=400, detail="El campo de anotaciones ('annotations') está vacío.")

    try:
        annotations_data = json.loads(annotations)
        if isinstance(annotations_data, dict):
            images_list = annotations_data.get("images", [annotations_data])
        else:
            images_list = annotations_data

        # Read image files
        image_files = {}
        for img_file in images:
            if img_file and img_file.filename:
                img_bytes = await img_file.read()
                if len(img_bytes) > 0:
                    image_files[img_file.filename] = img_bytes

        logger.info(f"Read {len(image_files)} image files from upload payload.")

        result = upload_dataset_to_roboflow(images_list, image_files)

        if result.get("success"):
            return result
        else:
            err_msg = result.get("error", "Upload failed")
            logger.error(f"Roboflow upload failed: {err_msg}")
            raise HTTPException(status_code=500, detail=err_msg)

    except json.JSONDecodeError as e:
        logger.error(f"JSONDecodeError in upload_roboflow: {e}")
        raise HTTPException(status_code=400, detail=f"JSON de anotaciones inválido: {str(e)}")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/roboflow-models")
def roboflow_models() -> Dict[str, Any]:
    """Get available dataset versions and supported model architectures from Roboflow."""
    return get_roboflow_models_and_versions()


@app.post("/api/train-roboflow")
async def train_roboflow(payload: Dict[str, Any] = Body(default={})) -> Dict[str, Any]:
    """Trigger model training on Roboflow for specified model_type and dataset version."""
    model_type = payload.get("model_type", "yolov8")
    version = payload.get("version", None)
    result = trigger_training(model_type=model_type, version=version)

    if result.get("success"):
        return result
    else:
        raise HTTPException(status_code=500, detail=result.get("error", "Training trigger failed"))


@app.post("/api/export-roboflow-dataset")
async def export_roboflow_dataset(payload: Dict[str, Any] = Body(default={})) -> Dict[str, Any]:
    """Generate version snapshot in Roboflow and download dataset locally for GPU training on PC."""
    model_format = payload.get("model_type", "yolov8")
    version = payload.get("version", None)
    result = export_dataset_version(model_format=model_format, version=version)

    if result.get("success"):
        return result
    else:
        raise HTTPException(status_code=500, detail=result.get("error", "Export failed"))


if __name__ == "__main__":
    import uvicorn
    from pathlib import Path
    backend_dir = str(Path(__file__).resolve().parent)
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        app_dir=backend_dir,
    )
