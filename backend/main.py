import os
from dotenv import load_dotenv
load_dotenv()

# Configure PyTorch CUDA memory allocator before importing torch
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import io
import time
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
    PDF_IMAGES_DIR,
)

# Import Pathology Foundation Models (CONCH & UNI)
from pathology_models import (
    get_pathology_models_status,
    classify_detections_with_conch,
    extract_detection_embeddings_uni,
)

# Global variables for model and processor
processor = None
device = "cuda" if torch.cuda.is_available() else "cpu"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    global processor
    logger.info("Initializing SAM 3 model...")
    start_time = time.time()

    # Enable bfloat16 autocast globally, as recommended by the official SAM3 notebooks
    torch.autocast(device_type=device, dtype=torch.bfloat16).__enter__()
    torch.inference_mode().__enter__()

    try:
        model = build_sam3_image_model(device=device, version="sam3.1")
        # Use a reasonable threshold — too low generates hundreds of garbage
        # detections that waste VRAM during mask interpolation. The official
        # default is 0.5; 0.35 balances recall vs. precision for histology.
        processor = Sam3Processor(model, device=device, confidence_threshold=0.35)
        logger.info(f"SAM 3 loaded successfully on {device} in {time.time() - start_time:.2f} seconds.")
    except Exception as e:
        logger.error(f"Failed to load SAM 3 model: {e}")
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
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Fine-grained visual prompts for automatic histology & cell instance segmentation.
# Replaces generic macro-prompts ("object", "person", "animal") that cause SAM3 to select the whole image frame.
AUTO_SEGMENT_PROMPTS = [
    {"key": "clase_1", "prompt": "small dark round cell nucleus",             "label": "Núcleos oscuros",     "color": "#f43f5e"},
    {"key": "clase_2", "prompt": "round cell with pale nucleus",              "label": "Células claras",      "color": "#38bdf8"},
    {"key": "clase_3", "prompt": "elongated spindle cell nucleus",            "label": "Núcleos alargados",   "color": "#e11d48"},
    {"key": "clase_4", "prompt": "circular tubule lumen cavity",             "label": "Lumen tubular",       "color": "#6366f1"},
    {"key": "clase_5", "prompt": "dense connective tissue band",             "label": "Tejido conectivo",    "color": "#8b5cf6"},
    {"key": "clase_6", "prompt": "red blood cell in vessel",                 "label": "Eritrocitos",         "color": "#ef4444"},
    {"key": "clase_7", "prompt": "large Leydig cell in intertubular space",  "label": "Células intersticiales", "color": "#f59e0b"},
    {"key": "clase_8", "prompt": "spermatogonium near basement membrane",    "label": "Espermatogonias",     "color": "#10b981"},
]

SPANISH_PROMPT_TRANSLATION_MAP = {
    "células germinales": "small round cell nucleus in seminiferous epithelium",
    "celulas germinales": "small round cell nucleus in seminiferous epithelium",
    "espermatogonia": "small round cell nucleus near basement membrane",
    "espermatogonia a clara": "round cell with pale chromatin near tubule basement membrane",
    "espermatogonia b": "small dark round nucleus at tubule wall",
    "espermatocito": "large round cell with mottled nucleus in tubule wall",
    "espermátida": "small dense dark nucleus near tubule lumen",
    "célula de sertoli": "tall cell with pale triangular nucleus",
    "celula de sertoli": "tall cell with pale triangular nucleus",
    "célula de leydig": "polygonal cell with eosinophilic cytoplasm in interstitial space",
    "celula de leydig": "polygonal cell with eosinophilic cytoplasm in interstitial space",
    "túbulo seminífero": "circular tissue structure with central lumen",
    "tubulo seminifero": "circular tissue structure with central lumen",
    "lumen": "empty circular lumen cavity",
    "núcleos": "dark round cell nucleus",
    "nucleos": "dark round cell nucleus",
}


def translate_prompt_if_needed(prompt_text: str) -> str:
    """Translates common Spanish medical/histology terms into visual English prompts for SAM 3."""
    if not prompt_text:
        return "cell nucleus"
    cleaned = prompt_text.strip().lower()
    if cleaned in SPANISH_PROMPT_TRANSLATION_MAP:
        translated = SPANISH_PROMPT_TRANSLATION_MAP[cleaned]
        logger.info(f"Translated Spanish prompt '{prompt_text}' -> '{translated}'")
        return translated
    return prompt_text


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
                        area = float(mask_i.cpu().numpy().sum()) * scale_x * scale_y
                    else:
                        area = float(np.array(mask_i).sum()) * scale_x * scale_y
                    detection["area"] = round(area, 2)
            except Exception as e:
                logger.warning(f"Failed to extract polygon for detection {i}: {e}")

        detections.append(detection)

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
    # Sam3Processor handles all resizing internally — scale factors are 1.0
    # because the processor already maps outputs back to original_height/width.
    return pil_image, orig_w, orig_h, 1.0, 1.0


@app.get("/api/health")
def health_check() -> Dict[str, Any]:
    return {
        "status": "ok" if processor is not None else "model_not_loaded",
        "device": device,
        "model": "sam3"
    }

@app.get("/api/prompts")
def get_prompts() -> Dict[str, Any]:
    """Returns the list of automatic prompts used for segmentation."""
    return {"prompts": AUTO_SEGMENT_PROMPTS}


@app.post("/api/segment")
async def segment_image(
    image: UploadFile = File(...),
    prompt: str = Form("object"),
    umbral: float = Form(0.05)
) -> Dict[str, Any]:
    """Single-prompt segmentation endpoint."""
    if processor is None:
        raise HTTPException(status_code=503, detail="SAM 3 model is not loaded.")

    logger.info(f"Received request: prompt='{prompt}', umbral={umbral}")
    start_time = time.time()

    try:
        contents = await image.read()
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
        inf_image, width, height, scale_x, scale_y = _prepare_image_for_inference(pil_image)

        translated_prompt = translate_prompt_if_needed(prompt)
        inference_state = processor.set_image(inf_image)
        output = processor.set_text_prompt(state=inference_state, prompt=translated_prompt)

        detections = _extract_detections(output, umbral, scale_x, scale_y, img_w=width, img_h=height)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        inference_time = time.time() - start_time
        logger.info(f"Found {len(detections)} detections in {inference_time:.2f}s")

        return {
            "width": width,
            "height": height,
            "detections": detections,
            "inference_time_seconds": round(inference_time, 2),
            "prompt": prompt,
            "umbral": umbral
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
    custom_prompt: str = Form(None),
    ontology_name: str = Form(None)
) -> Dict[str, Any]:
    """
    Universal automatic multi-prompt segmentation for ANY image.
    Runs universal visual prompts on the image and returns grouped detections
    with polygon segmentation data.
    """
    if processor is None:
        raise HTTPException(status_code=503, detail="SAM 3 model is not loaded.")

    logger.info(f"Received AUTO segment request, umbral={umbral}, custom_prompt='{custom_prompt}'")
    start_time = time.time()

    try:
        contents = await image.read()
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
        inf_image, width, height, scale_x, scale_y = _prepare_image_for_inference(pil_image)

        # Set image once (backbone features are cached in state)
        inference_state = processor.set_image(inf_image)

        all_groups = []
        total_detections = 0

        # Use ontology prompts if specified, otherwise fall back to generic
        if ontology_name and ontology_name.strip():
            ont_prompts = get_ontology_prompts(ontology_name.strip())
            if ont_prompts:
                prompts_to_run = ont_prompts
                logger.info(f"Using ontology '{ontology_name}' with {len(prompts_to_run)} prompts")
            else:
                logger.warning(f"Ontology '{ontology_name}' not found, using defaults")
                prompts_to_run = list(AUTO_SEGMENT_PROMPTS)
        else:
            prompts_to_run = list(AUTO_SEGMENT_PROMPTS)

        if custom_prompt and custom_prompt.strip():
            cp_text = custom_prompt.strip()
            translated_cp = translate_prompt_if_needed(cp_text)
            prompts_to_run.insert(0, {
                "key": f"clase_custom_{len(prompts_to_run)+1}",
                "prompt": translated_cp,
                "label": cp_text,  # Keep original user text for UI display
                "color": "#a855f7"
            })

        for prompt_info in prompts_to_run:
            raw_prompt_text = prompt_info["prompt"]
            prompt_text = translate_prompt_if_needed(raw_prompt_text)
            prompt_key = prompt_info["key"]
            prompt_label = prompt_info["label"]
            prompt_color = prompt_info["color"]

            # Reset prompts but keep image features
            processor.reset_all_prompts(inference_state)

            # Run text prompt
            output = processor.set_text_prompt(state=inference_state, prompt=prompt_text)

            detections = _extract_detections(output, umbral, scale_x, scale_y, include_polygons=True, img_w=width, img_h=height)
            count = len(detections)

            if count > 0:
                logger.info(f"  '{prompt_text}' -> {count} raw detections (with polygons)")
                all_groups.append({
                    "key": prompt_key,
                    "prompt": prompt_text,
                    "label": prompt_label,
                    "color": prompt_color,
                    "detections": detections,
                    "count": count
                })

        # --- Pathology Foundation Model Refinement (CONCH Zero-Shot) ---
        raw_candidates = []
        for g in all_groups:
            for d in g["detections"]:
                d_copy = dict(d)
                d_copy["initial_class_key"] = g["key"]
                d_copy["initial_label"] = g["label"]
                d_copy["color"] = g["color"]
                raw_candidates.append(d_copy)

        if raw_candidates and len(prompts_to_run) > 0:
            try:
                classified = classify_detections_with_conch(
                    image=pil_image,
                    detections=raw_candidates,
                    candidate_classes=prompts_to_run,
                    temperature=0.05
                )

                grouped_by_key = {}
                for c in prompts_to_run:
                    grouped_by_key[c["key"]] = {
                        "key": c["key"],
                        "prompt": c["prompt"],
                        "label": c["label"],
                        "color": c["color"],
                        "detections": [],
                        "count": 0
                    }

                for det in classified:
                    target_key = det.get("class_key", det.get("initial_class_key"))
                    if target_key not in grouped_by_key:
                        grouped_by_key[target_key] = {
                            "key": target_key,
                            "prompt": det.get("prompt", target_key),
                            "label": det.get("class_label", target_key),
                            "color": det.get("color", "#8b5cf6"),
                            "detections": [],
                            "count": 0
                        }
                    grouped_by_key[target_key]["detections"].append(det)
                    grouped_by_key[target_key]["count"] += 1

                all_groups = [g for g in grouped_by_key.values() if g["count"] > 0]
                total_detections = sum(g["count"] for g in all_groups)
                logger.info(f"CONCH zero-shot verified {total_detections} detections across {len(all_groups)} categories")
            except Exception as conch_err:
                logger.warning(f"CONCH refinement skipped (fallback to raw SAM 3): {conch_err}")
                total_detections = sum(g["count"] for g in all_groups)
        else:
            total_detections = sum(g["count"] for g in all_groups)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        inference_time = time.time() - start_time
        logger.info(f"AUTO: Total {total_detections} detections across {len(all_groups)} categories in {inference_time:.2f}s")

        return {
            "width": width,
            "height": height,
            "groups": all_groups,
            "total_detections": total_detections,
            "inference_time_seconds": round(inference_time, 2),
            "umbral": umbral
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
    """
    gemini_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not gemini_key:
        raise HTTPException(
            status_code=500,
            detail="GEMINI_API_KEY no está configurada en .env"
        )

    pdf_id = payload.get("pdf_id", "unknown")
    text = payload.get("text", "")
    
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

        ontology = build_ontology_document(
            pdf_id=pdf_id,
            filename=payload.get("filename", "unknown.pdf"),
            structures=structures,
            extracted_images=payload.get("images", []),
            domain_name=payload.get("domain_name"),
        )

        filepath = save_ontology(ontology)

        return {
            "success": True,
            "ontology": ontology,
            "saved_to": filepath,
        }
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
        raise HTTPException(status_code=404, detail="Imagen no encontrada.")
    return FileResponse(path, media_type="image/png")


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
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
