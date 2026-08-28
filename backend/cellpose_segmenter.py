"""
Cellpose & Cellpose-SAM Histological Segmentation Module.
Optimized for 6GB GPUs (RTX 3050/3060) with dynamic memory management,
bfloat16 precision, expandable segments allocation, and automatic CPU fallback on OOM.
"""

import io
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple, Union

# Optimize CUDA allocator to avoid memory fragmentation
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import cv2
import numpy as np
from PIL import Image
import torch

logger = logging.getLogger(__name__)

# Cache loaded Cellpose models
_CELLPOSE_MODELS: Dict[str, Any] = {}
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

AVAILABLE_CELLPOSE_MODELS = {
    "cpsam": "Cellpose-SAM (ViT-L Segment Anything - Recomendado)",
    "cpsam_v2": "Cellpose-SAM v2 (High-Res ViT-L)",
    "cpdino": "Cellpose DINO (ViT-L Histología)",
    "cpdino-vitb": "Cellpose DINO ViT-B (Ligero y Rápido)",
}


def is_cellpose_available() -> bool:
    """Check if cellpose library is installed and importable."""
    try:
        import cellpose  # noqa: F401
        from cellpose import models  # noqa: F401
        return True
    except ImportError:
        return False


def offload_cellpose_to_cpu() -> None:
    """Offload any cached Cellpose models from GPU to CPU to free VRAM for other models."""
    for key, model in list(_CELLPOSE_MODELS.items()):
        try:
            if hasattr(model, "net") and hasattr(model.net, "to"):
                model.net.to("cpu")
                model.device = torch.device("cpu")
                model.gpu = False
        except Exception as e:
            logger.warning(f"Error offloading Cellpose model {key}: {e}")
    _CELLPOSE_MODELS.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("Offloaded Cellpose models from GPU to CPU; VRAM released.")


def get_cellpose_status() -> Dict[str, Any]:
    """Return status dictionary of Cellpose models and hardware acceleration."""
    available = is_cellpose_available()
    vram_info = None
    if torch.cuda.is_available():
        try:
            total_vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            allocated_vram = torch.cuda.memory_allocated(0) / (1024 ** 3)
            free_vram = total_vram - allocated_vram
            vram_info = f"{free_vram:.2f} GB libres de {total_vram:.2f} GB"
        except Exception:
            vram_info = None

    return {
        "available": available,
        "device": _DEVICE,
        "gpu_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "vram_status": vram_info,
        "models": AVAILABLE_CELLPOSE_MODELS if available else {},
        "default_model": "cpsam",
    }


def load_cellpose_model(model_type: str = "cpsam", device: Optional[str] = None) -> Optional[Any]:
    """
    Load or retrieve cached Cellpose / Cellpose-SAM model.
    Includes automatic VRAM cleanup, bfloat16 quantization, and graceful CPU fallback on OOM.
    """
    if not is_cellpose_available():
        logger.error("Cellpose is not installed in current environment.")
        return None

    from cellpose import models

    # Normalize model_type if legacy name was passed
    if model_type not in AVAILABLE_CELLPOSE_MODELS:
        model_type = "cpsam"

    target_device = device or _DEVICE
    use_gpu = (target_device == "cuda" and torch.cuda.is_available())

    cache_key = f"{model_type}_{target_device}"
    if cache_key in _CELLPOSE_MODELS:
        return _CELLPOSE_MODELS[cache_key]

    logger.info(f"Loading Cellpose model '{model_type}' on GPU={use_gpu} ({target_device})...")
    start_t = time.time()

    # Step 1: Clean PyTorch CUDA cache before allocating new model weights
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    try:
        model = models.CellposeModel(
            gpu=use_gpu,
            pretrained_model=model_type,
            device=torch.device(target_device) if use_gpu else None,
            use_bfloat16=True if use_gpu else False,
        )
        _CELLPOSE_MODELS[cache_key] = model
        logger.info(f"Cellpose '{model_type}' loaded in {time.time() - start_t:.2f}s on {target_device}.")
        return model

    except (torch.OutOfMemoryError, RuntimeError) as oom_err:
        logger.warning(f"OOM loading Cellpose model '{model_type}' on {target_device}: {oom_err}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Step 2: Fallback to CPU with bfloat16=False (guarantees completion without crashing backend)
        logger.info(f"Falling back to Cellpose '{model_type}' on CPU...")
        try:
            model = models.CellposeModel(
                gpu=False,
                pretrained_model=model_type,
                use_bfloat16=False,
            )
            _CELLPOSE_MODELS[f"{model_type}_cpu"] = model
            logger.info("Cellpose loaded on CPU fallback.")
            return model
        except Exception as cpu_err:
            logger.error(f"CPU fallback also failed: {cpu_err}")
            return None


def masks_to_detections(
    masks: np.ndarray,
    label: str = "cell",
    color: str = "#06b6d4",
    class_id: int = 1,
    min_area: int = 15,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Convert a 2D integer mask array (H, W) where 0=background, 1..N=instances
    into COCO/frontend compatible detection objects with bounding boxes and polygon contours.
    """
    detections: List[Dict[str, Any]] = []
    unique_ids = np.unique(masks)

    group_key = label.lower().replace(" ", "_")

    for inst_id in unique_ids:
        if inst_id == 0:
            continue

        bin_mask = (masks == inst_id).astype(np.uint8)
        area = int(np.sum(bin_mask))
        if area < min_area:
            continue

        # Find external polygon contour
        contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        polys: List[List[float]] = []

        for cnt in contours:
            # Approximate polygon with smooth tolerance if too dense
            epsilon = 0.005 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            if len(approx) >= 3:
                flat = approx.reshape(-1, 2).flatten().astype(float).tolist()
                polys.append(flat)

        if not polys:
            continue

        ys, xs = np.where(bin_mask > 0)
        x1, y1 = int(xs.min()), int(ys.min())
        x2, y2 = int(xs.max()), int(ys.max())
        w, h = max(1, x2 - x1), max(1, y2 - y1)

        detection = {
            "id": int(inst_id),
            "class_id": class_id,
            "category_id": group_key,
            "label": label,
            "score": 0.95,
            "bbox": [x1, y1, w, h],
            "box": [x1, y1, x2, y2],
            "segmentation": polys,
            "area": float(area),
            "engine": "cellpose",
        }
        detections.append(detection)

    group = {
        "key": group_key,
        "prompt": label,
        "label": label,
        "color": color,
        "detections": detections,
        "count": len(detections),
    }

    return detections, group


def run_cellpose_segmentation(
    image_input: Union[bytes, Image.Image, np.ndarray],
    model_type: str = "cpdino-vitb",
    diameter: Optional[float] = None,
    channels: Optional[List[int]] = None,
    cellprob_threshold: float = 0.0,
    flow_threshold: float = 0.4,
    prompt_label: str = "cell",
    color: str = "#06b6d4",
    min_area: int = 15,
) -> Dict[str, Any]:
    """
    Execute histological/cellular segmentation with Cellpose or Cellpose-SAM.
    """
    start_time = time.time()

    # Free cache before inference
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Normalize image to RGB numpy array
    if isinstance(image_input, bytes):
        pil_img = Image.open(io.BytesIO(image_input)).convert("RGB")
        img_np = np.array(pil_img)
    elif isinstance(image_input, Image.Image):
        pil_img = image_input.convert("RGB")
        img_np = np.array(pil_img)
    elif isinstance(image_input, np.ndarray):
        img_np = image_input
        if len(img_np.shape) == 2:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
        elif img_np.shape[2] == 4:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)
    else:
        raise ValueError(f"Unsupported image input type: {type(image_input)}")

    h, w = img_np.shape[:2]

    # Load model with fallback
    model = load_cellpose_model(model_type=model_type)
    if model is None:
        raise RuntimeError("No se pudo inicializar ningún modelo de Cellpose en GPU o CPU.")

    chan = channels if channels is not None else [0, 0]

    logger.info(f"Running Cellpose ({getattr(model, 'pretrained_model', model_type)}) on image {w}x{h} with diameter={diameter}...")

    # Run inference with GPU inference mode
    eval_kwargs = {
        "diameter": diameter,
        "cellprob_threshold": cellprob_threshold,
        "flow_threshold": flow_threshold,
        "progress": None,
    }
    # Cellpose 4+ uses channel_axis; only pass channels if specified or fallback
    if channels is not None:
        eval_kwargs["channels"] = channels

    try:
        with torch.inference_mode():
            try:
                eval_res = model.eval(img_np, **eval_kwargs)
            except (TypeError, ValueError):
                eval_kwargs.pop("channels", None)
                eval_res = model.eval(img_np, **eval_kwargs)
    except (torch.OutOfMemoryError, RuntimeError) as eval_oom:
        logger.warning(f"CUDA OOM during Cellpose eval: {eval_oom}. Retrying on CPU...")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        cpu_model = load_cellpose_model(model_type="cpdino-vitb", device="cpu")
        with torch.inference_mode():
            try:
                eval_res = cpu_model.eval(img_np, **eval_kwargs)
            except (TypeError, ValueError):
                eval_kwargs.pop("channels", None)
                eval_res = cpu_model.eval(img_np, **eval_kwargs)

    # Free cache after inference
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    masks = eval_res[0]

    # Convert masks to detections and group
    all_detections, group = masks_to_detections(
        masks=masks,
        label=prompt_label or "cell",
        color=color,
        min_area=min_area,
    )

    inference_time = round(time.time() - start_time, 3)
    logger.info(f"Cellpose segmented {len(all_detections)} instances in {inference_time}s.")

    return {
        "success": True,
        "engine": "cellpose",
        "model_type": model_type,
        "width": w,
        "height": h,
        "total_detections": len(all_detections),
        "groups": [group] if all_detections else [],
        "detections": all_detections,
        "inference_time_seconds": inference_time,
        "estimated_diameter": getattr(model, "diam_labels", None),
    }
