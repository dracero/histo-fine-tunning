import os
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
from typing import List, Dict, Any, AsyncGenerator

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
        model = build_sam3_image_model(device=device)
        # Very low internal threshold so processor returns all detections
        processor = Sam3Processor(model, device=device, confidence_threshold=0.01)
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

# Universal prompts for automatic image partition & segmentation (works for general photos & histology).
# The user receives pure generic classes (Clase 1, Clase 2...) without domain identification.
AUTO_SEGMENT_PROMPTS = [
    {"key": "clase_1", "prompt": "object",                         "label": "Clase 1", "color": "#8b5cf6"},
    {"key": "clase_2", "prompt": "person",                         "label": "Clase 2", "color": "#38bdf8"},
    {"key": "clase_3", "prompt": "animal",                         "label": "Clase 3", "color": "#f59e0b"},
    {"key": "clase_4", "prompt": "clothing",                       "label": "Clase 4", "color": "#ec4899"},
    {"key": "clase_5", "prompt": "head",                           "label": "Clase 5", "color": "#06b6d4"},
    {"key": "clase_6", "prompt": "plant or tree",                  "label": "Clase 6", "color": "#10b981"},
    {"key": "clase_7", "prompt": "structure or shape",             "label": "Clase 7", "color": "#fb7185"},
    {"key": "clase_8", "prompt": "cell or nucleus",                "label": "Clase 8", "color": "#f43f5e"},
    {"key": "clase_9", "prompt": "elongated dark nucleus",         "label": "Clase 9", "color": "#e11d48"},
    {"key": "clase_10","prompt": "circular tissue structure",      "label": "Clase 10","color": "#6366f1"},
]

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


def _extract_detections(output, umbral: float, scale_x: float = 1.0, scale_y: float = 1.0, include_polygons: bool = True) -> list:
    """Extract detections from processor output state dict and rescale boxes to original image size."""
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

    detections = []
    for i, score in enumerate(clean_scores):
        s = float(score)
        if s >= umbral:
            box = clean_boxes[i]
            # Rescale box from inference dimensions back to original image dimensions
            rescaled_box = [
                float(box[0] * scale_x),
                float(box[1] * scale_y),
                float(box[2] * scale_x),
                float(box[3] * scale_y)
            ]

            detection = {
                "box": rescaled_box,
                "score": round(s, 4)
            }

            # Extract polygon from mask
            if include_polygons and masks is not None:
                try:
                    mask_i = masks[i]
                    polys = mask_to_polygons(mask_i, scale_x, scale_y)
                    if polys:
                        detection["segmentation"] = polys
                        # Compute area from mask
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
    Resizes high-res images to MAX_INFERENCE_DIM to prevent CUDA VRAM OOM during mask interpolation.
    Returns (inference_pil_image, orig_width, orig_height, scale_x, scale_y)
    """
    orig_w, orig_h = pil_image.size
    max_dim = max(orig_w, orig_h)

    if max_dim > MAX_INFERENCE_DIM:
        scale = MAX_INFERENCE_DIM / float(max_dim)
        new_w = max(1, int(orig_w * scale))
        new_h = max(1, int(orig_h * scale))
        inf_image = pil_image.resize((new_w, new_h), Image.Resampling.BILINEAR)
        scale_x = orig_w / float(new_w)
        scale_y = orig_h / float(new_h)
        logger.info(f"Resized input from {orig_w}x{orig_h} to {new_w}x{new_h} for inference (scale factors: {scale_x:.3f}, {scale_y:.3f})")
    else:
        inf_image = pil_image
        scale_x = 1.0
        scale_y = 1.0

    return inf_image, orig_w, orig_h, scale_x, scale_y


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

        inference_state = processor.set_image(inf_image)
        output = processor.set_text_prompt(state=inference_state, prompt=prompt)

        detections = _extract_detections(output, umbral, scale_x, scale_y)

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
    custom_prompt: str = Form(None)
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

        # Create active prompts list
        prompts_to_run = list(AUTO_SEGMENT_PROMPTS)
        if custom_prompt and custom_prompt.strip():
            cp_text = custom_prompt.strip()
            prompts_to_run.insert(0, {
                "key": f"clase_custom_{len(prompts_to_run)+1}",
                "prompt": cp_text,
                "label": f"Clase {len(prompts_to_run)+1}",
                "color": "#a855f7"
            })

        for prompt_info in prompts_to_run:
            prompt_text = prompt_info["prompt"]
            prompt_key = prompt_info["key"]
            prompt_label = prompt_info["label"]
            prompt_color = prompt_info["color"]

            # Reset prompts but keep image features
            processor.reset_all_prompts(inference_state)

            # Run text prompt
            output = processor.set_text_prompt(state=inference_state, prompt=prompt_text)

            detections = _extract_detections(output, umbral, scale_x, scale_y, include_polygons=True)
            count = len(detections)
            total_detections += count

            if count > 0:
                logger.info(f"  '{prompt_text}' -> {count} detections (with polygons)")
                all_groups.append({
                    "key": prompt_key,
                    "prompt": prompt_text,
                    "label": prompt_label,
                    "color": prompt_color,
                    "detections": detections,
                    "count": count
                })

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
