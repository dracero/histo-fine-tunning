"""
Pathology Foundation Models Module: CONCH & UNI.

Provides:
- CONCH (512-dim Vision-Language): Zero-shot classification of SAM 3 segmented cell/tissue crops.
- UNI (1024-dim ViT-Large): Dense morphological feature extraction for pathology patches.
"""

import io
import logging
import os
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

logger = logging.getLogger("sam3-backend")

# Device configuration: route auxiliary models (CONCH/UNI) to CPU on GPUs with <12GB VRAM
# to save GPU VRAM for SAM 3.
def _get_aux_device() -> torch.device:
    env_dev = os.getenv("PATHOLOGY_DEVICE", "").lower().strip()
    if env_dev == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if env_dev == "cpu":
        return torch.device("cpu")
    # Auto-detect: if CUDA is available, use GPU (with half precision) on modern RTX cards (>= 5.5GB)
    if torch.cuda.is_available():
        try:
            total_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            if total_mem_gb >= 5.5:
                return torch.device("cuda")
        except Exception as e:
            logger.debug(f"Could not read GPU device properties: {e}")
        return torch.device("cuda")
    return torch.device("cpu")

DEVICE = _get_aux_device()


# ---------------------------------------------------------------------------
# CONCH Wrapper (Vision-Language Foundation Model)
# ---------------------------------------------------------------------------

class ConchModelWrapper:
    """Singleton wrapper for MahmoodLab/CONCH vision-language foundation model."""

    _instance: Optional["ConchModelWrapper"] = None

    def __init__(self):
        self.model = None
        self.preprocess = None
        self.tokenizer = None
        self.is_loaded = False
        self.device = DEVICE

    @classmethod
    def get_instance(cls) -> "ConchModelWrapper":
        if cls._instance is None:
            cls._instance = ConchModelWrapper()
        return cls._instance

    def load(self, force_reload: bool = False) -> bool:
        """Load CONCH model weights onto GPU/CPU."""
        if self.is_loaded and not force_reload:
            return True

        try:
            from conch.open_clip_custom import create_model_from_pretrained, get_tokenizer

            logger.info(f"Loading CONCH model on {self.device}...")
            model, preprocess = create_model_from_pretrained(
                "conch_ViT-B-16",
                checkpoint_path="hf_hub:MahmoodLab/CONCH",
                device=self.device,
            )
            model.eval()
            self.model = model
            self.preprocess = preprocess
            self.tokenizer = get_tokenizer()
            self.is_loaded = True
            logger.info("CONCH model loaded successfully.")
            return True
        except Exception as e:
            logger.error(f"Failed to load CONCH model: {e}", exc_info=True)
            self.is_loaded = False
            return False

    def encode_texts(self, texts: List[str]) -> torch.Tensor:
        """Encode a list of text prompts into normalized 512-dim CONCH embeddings."""
        if not self.is_loaded:
            self.load()
        if not self.is_loaded or self.model is None or self.tokenizer is None:
            raise RuntimeError("CONCH model is not available.")

        from conch.open_clip_custom import tokenize

        tokens = tokenize(self.tokenizer, texts).to(self.device)
        dev_type = "cuda" if self.device.type == "cuda" else "cpu"
        with torch.inference_mode(), torch.autocast(device_type=dev_type, enabled=False):
            text_features = self.model.encode_text(tokens)
            text_features = F.normalize(text_features.float(), dim=-1)
        return text_features.float()

    def encode_image_crops(self, crops: List[Image.Image], batch_size: int = 16) -> torch.Tensor:
        """Encode a list of PIL image crops into normalized 512-dim CONCH embeddings."""
        if not self.is_loaded:
            self.load()
        if not self.is_loaded or self.model is None or self.preprocess is None:
            raise RuntimeError("CONCH model is not available.")

        dev_type = "cuda" if self.device.type == "cuda" else "cpu"
        all_embeddings = []
        for i in range(0, len(crops), batch_size):
            batch_crops = crops[i : i + batch_size]
            tensors = [self.preprocess(c) for c in batch_crops]
            batch_tensor = torch.stack(tensors).to(self.device).float()

            with torch.inference_mode(), torch.autocast(device_type=dev_type, enabled=False):
                image_features = self.model.encode_image(batch_tensor)
                image_features = F.normalize(image_features.float(), dim=-1)
                all_embeddings.append(image_features.float())

        if not all_embeddings:
            return torch.empty((0, 512), device=self.device, dtype=torch.float32)

        return torch.cat(all_embeddings, dim=0).float()


# ---------------------------------------------------------------------------
# UNI Wrapper (Pathology ViT-Large Foundation Model)
# ---------------------------------------------------------------------------

class UniModelWrapper:
    """Singleton wrapper for MahmoodLab/UNI 1024-dim foundation model."""

    _instance: Optional["UniModelWrapper"] = None

    def __init__(self):
        self.model = None
        self.transform = None
        self.is_loaded = False
        self.device = DEVICE

    @classmethod
    def get_instance(cls) -> "UniModelWrapper":
        if cls._instance is None:
            cls._instance = UniModelWrapper()
        return cls._instance

    def load(self, force_reload: bool = False) -> bool:
        """Load UNI model weights onto GPU/CPU."""
        if self.is_loaded and not force_reload:
            return True

        try:
            import timm
            from huggingface_hub import hf_hub_download

            logger.info(f"Loading UNI model on {self.device}...")
            checkpoint_path = hf_hub_download("MahmoodLab/UNI", "pytorch_model.bin")
            model = timm.create_model(
                "vit_large_patch16_224",
                pretrained=False,
                init_values=1.0,
                num_classes=0,
                dynamic_img_size=True,
            )
            model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"), strict=True)
            model.to(self.device)
            model.eval()

            transform = transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ])

            self.model = model
            self.transform = transform
            self.is_loaded = True
            logger.info("UNI model loaded successfully.")
            return True
        except Exception as e:
            logger.error(f"Failed to load UNI model: {e}", exc_info=True)
            self.is_loaded = False
            return False

    def encode_crops(self, crops: List[Image.Image], batch_size: int = 16) -> torch.Tensor:
        """Encode image crops into normalized 1024-dim UNI embeddings."""
        if not self.is_loaded:
            self.load()
        if not self.is_loaded or self.model is None or self.transform is None:
            raise RuntimeError("UNI model is not available.")

        dev_type = "cuda" if self.device.type == "cuda" else "cpu"
        all_embeddings = []
        for i in range(0, len(crops), batch_size):
            batch_crops = crops[i : i + batch_size]
            tensors = [self.transform(c) for c in batch_crops]
            batch_tensor = torch.stack(tensors).to(self.device).float()

            with torch.inference_mode(), torch.autocast(device_type=dev_type, enabled=False):
                features = self.model(batch_tensor)
                features = F.normalize(features.float(), dim=-1)
                all_embeddings.append(features.float())

        if not all_embeddings:
            return torch.empty((0, 1024), device=self.device, dtype=torch.float32)

        return torch.cat(all_embeddings, dim=0).float()


# ---------------------------------------------------------------------------
# Virchow Wrapper (Paige AI ViT-Huge 1280-dim Pathology Foundation Model)
# ---------------------------------------------------------------------------

class VirchowModelWrapper:
    """Singleton wrapper for Paige AI Virchow2 1280-dim foundation model."""

    _instance: Optional["VirchowModelWrapper"] = None

    def __init__(self):
        self.model = None
        self.transform = None
        self.is_loaded = False
        self.device = DEVICE
        self.model_id = os.getenv("VIRCHOW_MODEL_ID", "paige-ai/Virchow2")

    @classmethod
    def get_instance(cls) -> "VirchowModelWrapper":
        if cls._instance is None:
            cls._instance = VirchowModelWrapper()
        return cls._instance

    def load(self, force_reload: bool = False) -> bool:
        """Load Virchow model weights onto GPU/CPU with optimal precision and transforms."""
        if self.is_loaded and not force_reload:
            return True

        try:
            import timm
            from timm.data import resolve_data_config
            from timm.data.transforms_factory import create_transform
            from dotenv import load_dotenv

            load_dotenv()
            token = os.getenv("HF_TOKEN")
            if token:
                os.environ["HF_TOKEN"] = token

            # Target CUDA if available for fast ViT-Huge inference
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            logger.info(f"Loading Virchow model ({self.model_id}) on {self.device}...")
            model = timm.create_model(
                f"hf_hub:{self.model_id}",
                pretrained=True,
                mlp_layer=timm.layers.SwiGLUPacked,
                act_layer=torch.nn.SiLU,
            )

            # Move to target device (in half precision on CUDA to save VRAM and maximize throughput)
            if self.device.type == "cuda":
                model = model.to(self.device).half()
            else:
                model = model.to(self.device)
            model.eval()

            try:
                data_config = resolve_data_config(model.pretrained_cfg, model=model)
                transform = create_transform(**data_config, is_training=False)
            except Exception:
                transform = transforms.Compose([
                    transforms.Resize(224),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ])

            self.model = model
            self.transform = transform
            self.is_loaded = True
            logger.info(f"Virchow model ({self.model_id}) loaded successfully on {self.device}.")
            return True
        except Exception as e:
            logger.warning(f"Failed to load Virchow model ({self.model_id}): {e}")
            self.is_loaded = False
            return False

    def encode_crops(self, crops: List[Image.Image], batch_size: int = 32) -> torch.Tensor:
        """Encode image crops into normalized 1280-dim Virchow embeddings."""
        if not self.is_loaded:
            self.load()
        if not self.is_loaded or self.model is None or self.transform is None:
            raise RuntimeError(f"Virchow model ({self.model_id}) is not available.")

        dev_type = "cuda" if self.device.type == "cuda" else "cpu"
        all_embeddings = []
        for i in range(0, len(crops), batch_size):
            batch_crops = crops[i : i + batch_size]
            tensors = [self.transform(c) for c in batch_crops]
            batch_tensor = torch.stack(tensors).to(self.device)
            if dev_type == "cuda":
                batch_tensor = batch_tensor.half()

            with torch.inference_mode(), torch.autocast(device_type=dev_type, enabled=(dev_type == "cuda")):
                features = self.model(batch_tensor)
                if isinstance(features, (tuple, list)):
                    features = features[0]
                elif isinstance(features, dict):
                    features = features.get("embeddings", features.get("logits", list(features.values())[0]))
                if hasattr(features, "ndim") and features.ndim == 3:
                    # Token 0 is the CLS token (1280d) in Virchow / Virchow2 ViT
                    features = features[:, 0]
                features = F.normalize(features.float(), dim=-1)
                all_embeddings.append(features.float().cpu())

        if not all_embeddings:
            return torch.empty((0, 1280), device=self.device, dtype=torch.float32)

        return torch.cat(all_embeddings, dim=0).float()


# ---------------------------------------------------------------------------
# High-Level Pathology Processing Functions
# ---------------------------------------------------------------------------

def _compute_detection_area(det: Dict[str, Any]) -> float:
    """Compute pixel area from a detection dict, handling both bbox [x,y,w,h] and box [x1,y1,x2,y2] formats."""
    if "bbox" in det and isinstance(det["bbox"], (list, tuple)) and len(det["bbox"]) == 4:
        _, _, w, h = det["bbox"]
        return max(1.0, float(abs(w * h)))
    if "box" in det and isinstance(det["box"], (list, tuple)) and len(det["box"]) == 4:
        x1, y1, x2, y2 = det["box"]
        return max(1.0, float(abs((x2 - x1) * (y2 - y1))))
    if "area" in det:
        return max(1.0, float(det["area"]))
    return 100.0


def extract_crops_from_detections(
    image: Image.Image,
    detections: List[Dict[str, Any]],
    margin_ratio: float = 0.35,
    min_size: int = 64,
) -> List[Image.Image]:
    """
    Extract PIL crops for a list of detections with adaptive contextual padding.

    Smaller detections receive proportionally larger context margins so that
    Virchow/CONCH always see enough surrounding tissue to discriminate cell
    types by position (basal vs luminal) and neighbourhood.
    """
    img_w, img_h = image.size
    crops = []

    for det in detections:
        if "bbox" in det and isinstance(det["bbox"], (list, tuple)) and len(det["bbox"]) == 4:
            x, y, w, h = det["bbox"]
        elif "box" in det and isinstance(det["box"], (list, tuple)) and len(det["box"]) == 4:
            bx1, by1, bx2, by2 = det["box"]
            x, y, w, h = bx1, by1, max(1, bx2 - bx1), max(1, by2 - by1)
        else:
            x, y, w, h = 0, 0, img_w, img_h

        # Adaptive margin: smaller cells get proportionally more context
        # so that Virchow sees enough tissue around them (trained on 224×224 tiles)
        cell_dim = max(float(w), float(h), 1.0)
        adaptive_ratio = margin_ratio * max(1.0, 80.0 / cell_dim)
        adaptive_ratio = min(adaptive_ratio, 1.5)  # cap at 150% expansion

        pad_x = max(4, int(w * adaptive_ratio))
        pad_y = max(4, int(h * adaptive_ratio))

        x1 = max(0, int(x - pad_x))
        y1 = max(0, int(y - pad_y))
        x2 = min(img_w, int(x + w + pad_x))
        y2 = min(img_h, int(y + h + pad_y))

        # Ensure crop meets minimum size for foundation model input
        crop_w, crop_h = x2 - x1, y2 - y1
        if crop_w < min_size or crop_h < min_size:
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            half = min_size // 2
            x1 = max(0, cx - half)
            y1 = max(0, cy - half)
            x2 = min(img_w, x1 + min_size)
            y2 = min(img_h, y1 + min_size)

        crop = image.crop((x1, y1, x2, y2))
        crops.append(crop)

    return crops


# Universal morphological and tissue descriptors (language-agnostic morphological stems)
CELLULAR_INDICATORS = (
    "cell", "célula", "celula", "nuclei", "nucleus", "núcleo", "nucleo",
    "cyte", "cito", "blast", "blasto", "clast", "clasto",
    "gonia", "gonium", "tid", "tide", "zoon", "zoide", "phage", "fago",
    "phil", "filo", "karyo", "carionte", "cyte", "epitheliocyte",
)

NON_CELLULAR_INDICATORS = (
    "lumen", "luz", "cavity", "cavidad", "space", "espacio",
    "tubule", "tubulo", "túbulo", "layer", "capa", "epithelium", "epitelio",
    "stroma", "estroma", "tissue", "tejido", "wall", "pared",
    "membrane", "membrana", "organ", "organo", "órgano", "lobule", "lobulillo",
    "vessel", "vaso", "artery", "arteria", "vein", "vena", "fiber", "fibra",
    "matrix", "matriz", "interstitium", "intersticio",
    "inclusión", "inclusion", "cristaloide", "crystalloid",
)


def is_cellular_class(class_item: Union[Dict[str, Any], str]) -> bool:
    """
    Dynamically determine if a candidate structure represents a cell or nucleus instance
    versus a macro-compartment, cavity, tissue layer, or whole organ.
    Works universally across any tissue, organism, or pathology domain.
    """
    if isinstance(class_item, dict):
        if "is_cellular" in class_item:
            return bool(class_item["is_cellular"])
        if class_item.get("structure_type") in ("cell", "nucleus", "cellular"):
            return True
        if class_item.get("structure_type") in ("tissue_layer", "lumen_cavity", "macro_organ", "tissue", "cavity"):
            return False

        key = str(class_item.get("key", "")).lower().strip().replace("-", "_")
        label = str(class_item.get("label", class_item.get("name", ""))).lower().strip()
        prompt = str(class_item.get("prompt", "")).lower().strip()
        parent = str(class_item.get("parent", "") or "").lower().strip()
    else:
        key = str(class_item).lower().strip().replace("-", "_")
        label = key
        prompt = key
        parent = ""

    combined = f"{key} {label} {prompt}"
    key_label = f"{key} {label}"
    if any(term in key_label for term in NON_CELLULAR_INDICATORS):
        return False

    has_non_cellular = any(term in combined for term in NON_CELLULAR_INDICATORS)
    has_cellular = any(term in combined for term in CELLULAR_INDICATORS)

    if has_cellular and not has_non_cellular:
        return True
    if has_non_cellular and not has_cellular:
        return False

    # Constituent entities with parent tissues/compartments are granular structures/cells
    if parent and parent not in ("none", "null", ""):
        return True

    # Default to True so user-defined custom annotations are preserved
    return True


def filter_cellular_candidate_classes(classes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Filter candidate classes to retain only cellular entities for cell-level classification.
    Zero hardcoded tissue lists: relies purely on ontology metadata and universal morphological semantics.
    """
    filtered = [c for c in classes if is_cellular_class(c)]
    return filtered if filtered else classes


def enrich_histology_prompt(raw_key: str, raw_name: str, existing_prompt: Optional[str] = None) -> str:
    """
    Dynamically compose vision-language prompts using the active ontology and user metadata.
    Zero hardcoded tables: uses the domain visual descriptions extracted by the multimodal LLM or provided by the user.
    """
    name = (raw_name or raw_key or "structure").strip()
    prompt = (existing_prompt or "").strip()

    if prompt and len(prompt) >= 20:
        if not any(w in prompt.lower() for w in ("histolog", "h&e", "microscop", "stain", "tissue", "section", "cell")):
            return f"{prompt} in histological microscopic section"
        return prompt

    if prompt:
        return f"{name}: {prompt} in histological microscopic section"

    return f"{name} in histological tissue section"


def classify_detections_with_conch(
    image: Image.Image,
    detections: List[Dict[str, Any]],
    candidate_classes: List[Dict[str, str]],
    temperature: float = 0.12,
    is_histology: bool = True,
) -> List[Dict[str, Any]]:
    """
    Perform Zero-Shot Pathology Classification on a list of detections using CONCH.

    Args:
        image: Original full PIL image.
        detections: List of detection dicts (each containing 'id', 'bbox', 'polygon', etc.)
        candidate_classes: List of dicts with 'key', 'prompt', 'label', 'color'
        temperature: Softmax scaling temperature.
        is_histology: If False, skip CONCH as it is reserved for histology domains.

    Returns:
        List of classified detections with updated class_key, class_label, color,
        conch_confidence, and class_scores.
    """
    if not detections or not candidate_classes:
        return detections

    if not is_histology:
        logger.info("Non-histology domain: skipping CONCH zero-shot classification.")
        for det in detections:
            det["conch_skipped"] = True
            det["conch_reason"] = "CONCH is restricted to histology ontologies"
        return detections

    conch = ConchModelWrapper.get_instance()
    if not conch.is_loaded:
        success = conch.load()
        if not success:
            raise RuntimeError("CONCH model could not be initialized.")

    # 1. Prepare and enrich text prompts
    prompts_text = [
        enrich_histology_prompt(
            raw_key=c.get("key", ""),
            raw_name=c.get("label", c.get("name", "")),
            existing_prompt=c.get("prompt"),
        )
        for c in candidate_classes
    ]
    class_keys = [c.get("key") for c in candidate_classes]
    class_labels = [c.get("label", c.get("name", c.get("key"))) for c in candidate_classes]
    class_colors = [c.get("color", "#8b5cf6") for c in candidate_classes]

    # Encode text prompts
    text_embeddings = conch.encode_texts(prompts_text)  # (N_classes, 512)

    # 2. Extract crops for all detections
    crops = extract_crops_from_detections(image, detections)

    # Encode image crops
    image_embeddings = conch.encode_image_crops(crops, batch_size=16)  # (N_detections, 512)

    # 3. Compute cosine similarity matrix: (N_detections, N_classes) in float32
    image_embeddings = image_embeddings.float()
    text_embeddings = text_embeddings.float()
    similarity_matrix = torch.matmul(image_embeddings, text_embeddings.T).float()

    # Softmax probabilities
    probs = F.softmax(similarity_matrix / temperature, dim=-1).float().cpu().numpy()
    similarities = similarity_matrix.float().cpu().numpy()

    # 4. Assign best matching class
    classified_detections = []
    for i, det in enumerate(detections):
        det_copy = dict(det)
        det_probs = probs[i]
        best_idx = int(np.argmax(det_probs))
        best_conf = float(det_probs[best_idx])
        best_sim = float(similarities[i][best_idx])

        det_copy["category_id"] = class_keys[best_idx]
        det_copy["class_key"] = class_keys[best_idx]
        det_copy["class_label"] = class_labels[best_idx]
        det_copy["color"] = class_colors[best_idx]
        det_copy["conch_confidence"] = round(best_conf, 4)
        det_copy["conch_similarity"] = round(best_sim, 4)
        det_copy["conch_scores"] = {
            class_keys[k]: round(float(det_probs[k]), 4)
            for k in range(len(class_keys))
        }

        classified_detections.append(det_copy)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return classified_detections


def discriminate_and_cluster_with_pathology_models(
    image: Image.Image,
    detections: List[Dict[str, Any]],
    candidate_classes: Optional[List[Dict[str, Any]]] = None,
    temperature: float = 0.12,
    is_histology: bool = True,
) -> List[Dict[str, Any]]:
    """
    Exhaustive semantic discrimination and clustering of all detections using
    MahmoodLab/UNI (1024-dim morphology) and MahmoodLab/CONCH (512-dim vision-language).

    - Extracts contextual crops for every detected mask.
    - Computes deep morphological representations with UNI and cross-modal embeddings with CONCH.
    - Groups equivalent histological entities (all nuclei, all fibers, all lumens, all muscle)
      and discriminates distinct structures based on deep feature space.
    - Aligns each group with the active ontology concepts or dynamically discovered structures.

    Returns:
        List of enriched detection dictionaries with assigned class_key, class_label, color,
        conch_confidence, and morphology scores.
    """
    if not detections:
        return []

    # If domain is not histology, skip CONCH/UNI foundation models entirely
    if not is_histology:
        logger.info("Non-histology domain: skipping CONCH and UNI pathology models, retaining initial SAM 3 prompts.")
        classified_detections = []
        for det in detections:
            det_copy = dict(det)
            det_copy["class_key"] = det.get("initial_class_key", det.get("category_id", "default_class"))
            det_copy["class_label"] = det.get("initial_label", det.get("class_label", "Objeto"))
            det_copy["conch_skipped"] = True
            det_copy["conch_reason"] = "Restricted to histology ontologies"
            classified_detections.append(det_copy)
        return classified_detections

    # Filter candidate classes to retain only cellular entities (remove lumen, organs, macro-tissues)
    if candidate_classes:
        candidate_classes = filter_cellular_candidate_classes(candidate_classes)

    # 1. Extract crops for all detections
    crops = extract_crops_from_detections(image, detections)
    num_dets = len(detections)

    # 2. Extract Virchow 1280-dim WSI features (if available)
    virchow_features = None
    try:
        virchow = VirchowModelWrapper.get_instance()
        if not virchow.is_loaded:
            virchow.load()
        if virchow.is_loaded:
            virchow_features = virchow.encode_crops(crops, batch_size=16).float()  # (N, 1280)
            logger.info(f"Extracted Virchow 1280-dim embeddings for {num_dets} detections")
    except Exception as virchow_err:
        logger.warning(f"Virchow feature extraction skipped: {virchow_err}")

    # 3. Extract UNI 1024-dim morphological features (if available)
    uni_features = None
    try:
        uni = UniModelWrapper.get_instance()
        if not uni.is_loaded:
            uni.load()
        if uni.is_loaded:
            uni_features = uni.encode_crops(crops, batch_size=16).float()  # (N, 1024)
            logger.info(f"Extracted UNI 1024-dim embeddings for {num_dets} detections")
    except Exception as uni_err:
        logger.warning(f"UNI feature extraction skipped: {uni_err}")

    # 4. Extract CONCH 512-dim vision-language features
    conch_features = None
    try:
        conch = ConchModelWrapper.get_instance()
        if not conch.is_loaded:
            conch.load()
        if conch.is_loaded:
            conch_features = conch.encode_image_crops(crops, batch_size=16).float()  # (N, 512)
            logger.info(f"Extracted CONCH 512-dim embeddings for {num_dets} detections")
    except Exception as conch_err:
        logger.warning(f"CONCH feature extraction skipped: {conch_err}")

    # 5. Deduplicate spatially overlapping detections before classification (Fix 5)
    # When multiple SAM3 prompts detect the same cell, keep only the highest-score one
    dedup_indices = list(range(len(detections)))
    if len(detections) > 1:
        try:
            import torchvision.ops
            det_boxes = []
            det_scores = []
            for d in detections:
                if "box" in d and isinstance(d["box"], (list, tuple)) and len(d["box"]) == 4:
                    det_boxes.append(d["box"])
                elif "bbox" in d and isinstance(d["bbox"], (list, tuple)) and len(d["bbox"]) == 4:
                    bx, by, bw, bh = d["bbox"]
                    det_boxes.append([bx, by, bx + bw, by + bh])
                else:
                    det_boxes.append([0, 0, 1, 1])
                det_scores.append(d.get("score", 0.5))

            boxes_t = torch.tensor(det_boxes, dtype=torch.float32)
            scores_t = torch.tensor(det_scores, dtype=torch.float32)
            keep = torchvision.ops.nms(boxes_t, scores_t, iou_threshold=0.65).tolist()
            removed = len(detections) - len(keep)
            if removed > 0:
                logger.info(f"IoU deduplication removed {removed} overlapping detections before classification")
            dedup_indices = keep
        except Exception as nms_err:
            logger.warning(f"Pre-classification IoU dedup skipped: {nms_err}")

    # Apply deduplication to all feature tensors and detections list
    detections = [detections[i] for i in dedup_indices]
    crops = [crops[i] for i in dedup_indices]
    if virchow_features is not None:
        virchow_features = virchow_features[dedup_indices]
    if uni_features is not None:
        uni_features = uni_features[dedup_indices]
    if conch_features is not None:
        conch_features = conch_features[dedup_indices]
    num_dets = len(detections)

    # 6. Supervised classification with candidate classes (CONCH text + Virchow morphology)
    if candidate_classes and len(candidate_classes) > 0 and conch_features is not None:
        conch = ConchModelWrapper.get_instance()
        prompts_text = [
            enrich_histology_prompt(
                raw_key=c.get("key", ""),
                raw_name=c.get("label", c.get("name", "")),
                existing_prompt=c.get("prompt"),
            )
            for c in candidate_classes
        ]
        class_keys = [c.get("key") for c in candidate_classes]
        class_labels = [c.get("label", c.get("name", c.get("key"))) for c in candidate_classes]
        class_colors = [c.get("color", "#8b5cf6") for c in candidate_classes]

        text_embeddings = conch.encode_texts(prompts_text).float()  # (N_classes, 512)
        similarity_matrix = torch.matmul(conch_features.float(), text_embeddings.float().T).float()
        probs = F.softmax(similarity_matrix / temperature, dim=-1).float().cpu().numpy()
        similarities = similarity_matrix.float().cpu().numpy()

        # Virchow inter-detection similarity for neighbourhood consistency refinement
        virchow_sim_matrix = None
        if virchow_features is not None and len(virchow_features) > 1:
            virchow_sim_matrix = torch.matmul(
                F.normalize(virchow_features.float(), dim=-1),
                F.normalize(virchow_features.float(), dim=-1).T,
            ).cpu().numpy()

        classified_detections = []
        for i, det in enumerate(detections):
            det_copy = dict(det)
            det_probs = probs[i]
            best_idx = int(np.argmax(det_probs))
            best_conf = float(det_probs[best_idx])
            best_sim = float(similarities[i][best_idx])

            det_copy["category_id"] = class_keys[best_idx]
            det_copy["class_key"] = class_keys[best_idx]
            det_copy["class_label"] = class_labels[best_idx]
            det_copy["color"] = class_colors[best_idx]
            det_copy["conch_confidence"] = round(best_conf, 4)
            det_copy["conch_similarity"] = round(best_sim, 4)
            det_copy["conch_scores"] = {
                class_keys[k]: round(float(det_probs[k]), 4)
                for k in range(len(class_keys))
            }
            # Flag uncertain predictions for downstream refinement
            if best_conf < 0.40:
                det_copy["classification_uncertain"] = True
            det_copy["detection_area"] = round(_compute_detection_area(det), 1)
            classified_detections.append(det_copy)

        # Virchow neighbourhood consistency: align uncertain detections with confident neighbours
        if virchow_sim_matrix is not None:
            confident = [(j, d) for j, d in enumerate(classified_detections)
                         if not d.get("classification_uncertain")]
            for i, det in enumerate(classified_detections):
                if not det.get("classification_uncertain"):
                    continue
                best_sim_val = -1.0
                best_j = -1
                for j, _ in confident:
                    if i == j:
                        continue
                    s = float(virchow_sim_matrix[i, j])
                    if s > best_sim_val:
                        best_sim_val = s
                        best_j = j
                if best_j >= 0 and best_sim_val > 0.85:
                    donor = classified_detections[best_j]
                    det["category_id"] = donor["category_id"]
                    det["class_key"] = donor["class_key"]
                    det["class_label"] = donor["class_label"]
                    det["color"] = donor["color"]
                    det["classification_uncertain"] = False
                    det["neighbour_aligned"] = True
                    det["neighbour_similarity"] = round(best_sim_val, 4)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return classified_detections

    # 6. Unsupervised Morphological & WSI Clustering fallback (when no ontology classes provided)
    # Combine Virchow (1280d), UNI (1024d) and CONCH (512d) into a joint feature embedding space (up to 2816d)
    feat_tensors = []
    if virchow_features is not None:
        feat_tensors.append(virchow_features.float())
    if uni_features is not None:
        feat_tensors.append(uni_features.float())
    if conch_features is not None:
        feat_tensors.append(conch_features.float())

    if feat_tensors and num_dets >= 2:
        joint_features = torch.cat(feat_tensors, dim=-1).float()
        joint_features = F.normalize(joint_features, dim=-1).float().cpu().numpy()

        # Determine optimal number of clusters
        k = min(max(2, int(np.sqrt(num_dets))), min(8, num_dets))
        try:
            from sklearn.cluster import AgglomerativeClustering
            clustering = AgglomerativeClustering(n_clusters=k, metric="cosine", linkage="average")
            labels = clustering.fit_predict(joint_features)
        except Exception:
            # Fallback to simple KMeans
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
            labels = kmeans.fit_predict(joint_features)

        palette = [
            "#e11d48", "#8b5cf6", "#06b6d4", "#f59e0b", "#10b981",
            "#ec4899", "#6366f1", "#14b8a6", "#f97316", "#84cc16",
        ]

        classified_detections = []
        for i, det in enumerate(detections):
            det_copy = dict(det)
            c_id = int(labels[i])
            det_copy["category_id"] = f"cluster_{c_id + 1}"
            det_copy["class_key"] = f"cluster_{c_id + 1}"
            det_copy["class_label"] = f"Estructura Morfológica {c_id + 1}"
            det_copy["color"] = palette[c_id % len(palette)]
            det_copy["cluster_id"] = c_id
            classified_detections.append(det_copy)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return classified_detections

    # If no features could be extracted, return original detections
    return detections


def group_detections_by_class(
    classified_detections: List[Dict[str, Any]],
    candidate_classes: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """
    Group classified detection items into structured class buckets for UI rendering.
    """
    grouped_map: Dict[str, Dict[str, Any]] = {}

    # Initialize from candidate_classes if provided to preserve full ontology ordering
    if candidate_classes:
        for c in candidate_classes:
            k = c.get("key")
            if k:
                grouped_map[k] = {
                    "key": k,
                    "prompt": c.get("prompt", k),
                    "label": c.get("label", c.get("name", k)),
                    "color": c.get("color", "#8b5cf6"),
                    "detections": [],
                    "count": 0,
                }

    for det in classified_detections:
        k = det.get("class_key", det.get("category_id", det.get("initial_class_key", "default_class")))
        if k not in grouped_map:
            grouped_map[k] = {
                "key": k,
                "prompt": det.get("prompt", det.get("initial_label", k)),
                "label": det.get("class_label", det.get("initial_label", k)),
                "color": det.get("color", "#8b5cf6"),
                "detections": [],
                "count": 0,
            }
        grouped_map[k]["detections"].append(det)
        grouped_map[k]["count"] += 1

    # Filter only groups that have detections or are part of the active ontology
    result = [g for g in grouped_map.values() if g["count"] > 0]
    return result


def extract_detection_embeddings_uni(
    image: Image.Image,
    detections: List[Dict[str, Any]],
    is_histology: bool = True,
) -> List[List[float]]:
    """
    Extract 1024-dim UNI embeddings for all detections.
    """
    if not detections:
        return []

    if not is_histology:
        logger.info("Non-histology domain: skipping UNI embeddings extraction.")
        return []

    uni = UniModelWrapper.get_instance()
    if not uni.is_loaded:
        success = uni.load()
        if not success:
            raise RuntimeError("UNI model could not be initialized.")

    crops = extract_crops_from_detections(image, detections)
    embeddings = uni.encode_crops(crops, batch_size=16)

    embeddings_list = embeddings.float().cpu().numpy().tolist()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return embeddings_list


def extract_detection_embeddings_virchow(
    image: Image.Image,
    detections: List[Dict[str, Any]],
    is_histology: bool = True,
) -> List[List[float]]:
    """
    Extract 1280-dim Virchow pathology embeddings for all detections.
    """
    if not detections:
        return []

    if not is_histology:
        logger.info("Non-histology domain: skipping Virchow embeddings extraction.")
        return []

    virchow = VirchowModelWrapper.get_instance()
    if not virchow.is_loaded:
        success = virchow.load()
        if not success:
            raise RuntimeError("Virchow model could not be initialized.")

    crops = extract_crops_from_detections(image, detections)
    embeddings = virchow.encode_crops(crops, batch_size=16)

    embeddings_list = embeddings.float().cpu().numpy().tolist()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return embeddings_list


def classify_with_virchow_prototypes(
    image: Image.Image,
    detections: List[Dict[str, Any]],
    candidate_classes: Optional[List[Dict[str, Any]]] = None,
    temperature: float = 0.12,
    is_histology: bool = True,
) -> List[Dict[str, Any]]:
    """
    High-precision few-shot exemplar & morphological prototype ensemble classification using
    Paige AI Virchow 2 (1280-dim ViT-Huge), MahmoodLab UNI (1024-dim ViT-Large) and MahmoodLab CONCH (512-dim).

    Features:
    - Dual deep foundation embeddings: combines Virchow 2 (1280d) and UNI (1024d) for robust cytological representation.
    - Preserves all user-labeled exemplars without overwriting.
    - If 1+ exemplar classes are labeled, propagates them to morphologically identical cells via the blended ensemble.
    - Inter-cell neighbourhood consistency using dual-foundation cosine affinity.
    - For unlabeled cells, uses CONCH zero-shot guided by deep cellular prompts and relative cell volume priors.
    """
    if not detections:
        return []

    if not is_histology:
        logger.info("Non-histology domain: skipping foundation prototype classification.")
        return detections

    # 1. Dynamically resolve candidate classes from parameters, detections, or active ontology
    filtered_classes = filter_cellular_candidate_classes(candidate_classes or [])

    # If no candidate classes were explicitly provided, derive dynamically from unique detection labels
    if not filtered_classes:
        derived_classes = []
        seen_keys = set()
        for d in detections:
            k = d.get("class_key") or d.get("category_id")
            if k and str(k).lower().strip() not in ("clase_1", "clase_0", "default", "unlabeled", "default_class", ""):
                if k not in seen_keys:
                    seen_keys.add(k)
                    derived_classes.append({
                        "key": k,
                        "label": d.get("class_label", k),
                        "name": d.get("class_label", k),
                        "color": d.get("color", "#8b5cf6"),
                        "prompt": d.get("prompt", d.get("class_label", k)),
                    })
        filtered_classes = filter_cellular_candidate_classes(derived_classes)

    # Load Virchow 2 and UNI models
    virchow = VirchowModelWrapper.get_instance()
    if not virchow.is_loaded:
        virchow.load()

    uni = UniModelWrapper.get_instance()
    if not uni.is_loaded:
        uni.load()

    if not virchow.is_loaded and not uni.is_loaded:
        logger.warning("Neither Virchow nor UNI could be loaded, falling back to CONCH zero-shot.")
        return classify_detections_with_conch(
            image=image,
            detections=detections,
            candidate_classes=filtered_classes,
            temperature=temperature,
            is_histology=is_histology,
        )

    # 2. Extract crops for all detections
    crops = extract_crops_from_detections(image, detections)
    num_dets = len(detections)

    # 3. Extract Virchow 1280-dim and UNI 1024-dim embeddings
    virchow_feats_norm = None
    if virchow.is_loaded:
        try:
            virchow_feats = virchow.encode_crops(crops, batch_size=16).float()  # (N, 1280)
            virchow_feats_norm = F.normalize(virchow_feats, dim=-1)
        except Exception as e:
            logger.warning(f"Virchow feature extraction failed: {e}")

    uni_feats_norm = None
    if uni.is_loaded:
        try:
            uni_feats = uni.encode_crops(crops, batch_size=16).float()  # (N, 1024)
            uni_feats_norm = F.normalize(uni_feats, dim=-1)
        except Exception as e:
            logger.warning(f"UNI feature extraction failed: {e}")

    # 4. Check for labeled exemplars in current detections
    generic_keys = {"clase_1", "clase_0", "default", "unlabeled", "cell", "nucleus", "default_class", "objeto", ""}
    
    class_exemplars: Dict[str, List[int]] = {}
    class_meta: Dict[str, Dict[str, Any]] = {}

    for c in filtered_classes:
        k = c.get("key") or c.get("name")
        if k:
            class_meta[k] = {
                "key": k,
                "label": c.get("label", c.get("name", k)),
                "color": c.get("color", "#8b5cf6"),
                "prompt": c.get("prompt"),
            }

    user_labeled_indices = set()
    for idx, det in enumerate(detections):
        ck = det.get("class_key") or det.get("category_id")
        if ck and str(ck).lower().strip() not in generic_keys and not det.get("unassigned", False):
            if is_cellular_class(ck):
                if ck not in class_exemplars:
                    class_exemplars[ck] = []
                class_exemplars[ck].append(idx)
                user_labeled_indices.add(idx)
                if ck not in class_meta:
                    class_meta[ck] = {
                        "key": ck,
                        "label": det.get("class_label", ck),
                        "color": det.get("color", "#8b5cf6"),
                        "prompt": det.get("prompt"),
                    }

    # 5. First pass: Zero-Shot CONCH classification on all detections
    already_conch_scored = (
        len(detections) > 0
        and all(bool(d.get("conch_scores")) for d in detections)
    )
    if already_conch_scored:
        conch_classified = [dict(d) for d in detections]
    elif filtered_classes and len(filtered_classes) >= 1:
        conch_classified = classify_detections_with_conch(
            image=image,
            detections=detections,
            candidate_classes=filtered_classes,
            temperature=temperature,
            is_histology=is_histology,
        )
    else:
        conch_classified = [dict(d) for d in detections]

    # 6. Apply dynamic Morphometric statistics
    areas = [_compute_detection_area(d) for d in conch_classified]

    # 7. Few-Shot Exemplar Prototype Matching with Virchow 2 + UNI Ensemble
    exemplar_target_keys = list(class_exemplars.keys())
    ensemble_sim_matrix = None

    if exemplar_target_keys:
        virchow_ex_sims = None
        if virchow_feats_norm is not None:
            virchow_protos = []
            for k in exemplar_target_keys:
                ex_indices = class_exemplars[k]
                centroid = torch.mean(virchow_feats_norm[ex_indices], dim=0, keepdim=True)
                virchow_protos.append(F.normalize(centroid, dim=-1))
            virchow_protos_tensor = torch.cat(virchow_protos, dim=0)
            virchow_ex_sims = torch.matmul(virchow_feats_norm, virchow_protos_tensor.T).cpu().numpy()

        uni_ex_sims = None
        if uni_feats_norm is not None:
            uni_protos = []
            for k in exemplar_target_keys:
                ex_indices = class_exemplars[k]
                centroid = torch.mean(uni_feats_norm[ex_indices], dim=0, keepdim=True)
                uni_protos.append(F.normalize(centroid, dim=-1))
            uni_protos_tensor = torch.cat(uni_protos, dim=0)
            uni_ex_sims = torch.matmul(uni_feats_norm, uni_protos_tensor.T).cpu().numpy()

        if virchow_ex_sims is not None and uni_ex_sims is not None:
            ensemble_sim_matrix = 0.5 * virchow_ex_sims + 0.5 * uni_ex_sims
        elif virchow_ex_sims is not None:
            ensemble_sim_matrix = virchow_ex_sims
        elif uni_ex_sims is not None:
            ensemble_sim_matrix = uni_ex_sims

    # Inter-detection similarity matrix for neighbourhood consistency
    inter_sim_matrix = None
    if virchow_feats_norm is not None and uni_feats_norm is not None and num_dets > 1:
        v_sim = torch.matmul(virchow_feats_norm, virchow_feats_norm.T).cpu().numpy()
        u_sim = torch.matmul(uni_feats_norm, uni_feats_norm.T).cpu().numpy()
        inter_sim_matrix = 0.5 * v_sim + 0.5 * u_sim
    elif virchow_feats_norm is not None and num_dets > 1:
        inter_sim_matrix = torch.matmul(virchow_feats_norm, virchow_feats_norm.T).cpu().numpy()
    elif uni_feats_norm is not None and num_dets > 1:
        inter_sim_matrix = torch.matmul(uni_feats_norm, uni_feats_norm.T).cpu().numpy()

    final_classified = []
    for i, det in enumerate(conch_classified):
        det_copy = dict(det)
        area = areas[i]
        scores = det_copy.get("conch_scores", {})
        best_conf = det_copy.get("conch_confidence", 0.0)

        # A. USER LABELED EXEMPLAR: ALWAYS PRESERVE USER CHOICE
        if i in user_labeled_indices:
            orig_det = detections[i]
            ck = orig_det.get("class_key") or orig_det.get("category_id")
            meta = class_meta.get(ck, {"label": orig_det.get("class_label", ck), "color": orig_det.get("color", "#8b5cf6")})
            det_copy["category_id"] = ck
            det_copy["class_key"] = ck
            det_copy["class_label"] = meta["label"]
            det_copy["color"] = meta["color"]
            det_copy["virchow_confidence"] = 1.0
            det_copy["uni_confidence"] = 1.0
            det_copy["is_user_exemplar"] = True
            det_copy["virchow_verified"] = virchow.is_loaded
            det_copy["uni_verified"] = uni.is_loaded
            det_copy["detection_area"] = round(area, 1)
            final_classified.append(det_copy)
            continue

        # B. PROPAGATE FROM USER EXEMPLARS IF ENSEMBLE EMBEDDING IS VERY SIMILAR
        matched_exemplar = False
        if ensemble_sim_matrix is not None and len(exemplar_target_keys) > 0:
            ex_sims = ensemble_sim_matrix[i]
            best_ex_idx = int(np.argmax(ex_sims))
            best_ex_sim = float(ex_sims[best_ex_idx])
            # If blended similarity to a user exemplar is high (>= 0.76), propagate user's class
            if best_ex_sim >= 0.76:
                best_k = exemplar_target_keys[best_ex_idx]
                meta = class_meta.get(best_k, {"label": best_k, "color": "#8b5cf6"})
                det_copy["category_id"] = best_k
                det_copy["class_key"] = best_k
                det_copy["class_label"] = meta["label"]
                det_copy["color"] = meta["color"]
                det_copy["morphological_ensemble_similarity"] = round(best_ex_sim, 4)
                det_copy["virchow_confidence"] = round(best_ex_sim, 4)
                det_copy["propagated_from_exemplar"] = True
                det_copy["decision_source"] = "foundation_morphological_ensemble"
                matched_exemplar = True

        if not matched_exemplar:
            if best_conf < 0.40:
                det_copy["classification_uncertain"] = True

        det_copy["virchow_verified"] = virchow.is_loaded
        det_copy["uni_verified"] = uni.is_loaded
        det_copy["detection_area"] = round(area, 1)
        final_classified.append(det_copy)

    # 8. Data-driven neighbourhood consistency pass with blended similarity
    if inter_sim_matrix is not None:
        confident_dets = [(j, d) for j, d in enumerate(final_classified) if not d.get("classification_uncertain")]
        for i, det in enumerate(final_classified):
            if not det.get("classification_uncertain") or i in user_labeled_indices:
                continue
            best_sim = -1.0
            best_match_idx = -1
            for j, _ in confident_dets:
                if i == j:
                    continue
                sim = float(inter_sim_matrix[i, j])
                if sim > best_sim:
                    best_sim = sim
                    best_match_idx = j
            if best_match_idx >= 0 and best_sim > 0.84:
                donor = final_classified[best_match_idx]
                det["category_id"] = donor["category_id"]
                det["class_key"] = donor["class_key"]
                det["class_label"] = donor["class_label"]
                det["color"] = donor["color"]
                det["classification_uncertain"] = False
                det["neighbour_aligned"] = True
                det["neighbour_similarity"] = round(best_sim, 4)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return final_classified


# Alias for explicit ensemble nomenclature
classify_with_morphological_ensemble = classify_with_virchow_prototypes


def get_pathology_models_status() -> Dict[str, Any]:
    """Get status of CONCH, UNI, and Virchow foundation models."""
    conch = ConchModelWrapper.get_instance()
    uni = UniModelWrapper.get_instance()
    virchow = VirchowModelWrapper.get_instance()

    return {
        "device": str(DEVICE),
        "cuda_available": torch.cuda.is_available(),
        "conch": {
            "name": "CONCH (MahmoodLab/CONCH)",
            "embedding_dim": 512,
            "is_loaded": conch.is_loaded,
            "architecture": "CoCa ViT-B-16",
        },
        "uni": {
            "name": "UNI (MahmoodLab/UNI)",
            "embedding_dim": 1024,
            "is_loaded": uni.is_loaded,
            "architecture": "ViT-Large-patch16-224",
        },
        "virchow": {
            "name": f"Virchow ({virchow.model_id})",
            "embedding_dim": 1280,
            "is_loaded": virchow.is_loaded,
            "architecture": "ViT-Huge-SwiGLU-patch14-224",
        },
    }


def preload_all_pathology_models() -> Dict[str, Any]:
    """Preload all foundation models (CONCH, UNI, Virchow 2) onto GPU/CPU."""
    conch = ConchModelWrapper.get_instance()
    uni = UniModelWrapper.get_instance()
    virchow = VirchowModelWrapper.get_instance()

    c_ok = conch.load() if not conch.is_loaded else True
    u_ok = uni.load() if not uni.is_loaded else True
    v_ok = virchow.load() if not virchow.is_loaded else True

    return {
        "status": "ready" if (c_ok and u_ok and v_ok) else "partial",
        "conch_loaded": c_ok,
        "uni_loaded": u_ok,
        "virchow_loaded": v_ok,
        "device": str(DEVICE),
    }
