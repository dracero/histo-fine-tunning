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

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
# High-Level Pathology Processing Functions
# ---------------------------------------------------------------------------

def extract_crops_from_detections(
    image: Image.Image,
    detections: List[Dict[str, Any]],
    margin_ratio: float = 0.1,
    min_size: int = 16,
) -> List[Image.Image]:
    """
    Extract PIL crops for a list of detections with contextual padding.
    """
    img_w, img_h = image.size
    crops = []

    for det in detections:
        bbox = det.get("bbox", [0, 0, img_w, img_h])
        x, y, w, h = bbox

        # Add contextual margin
        pad_x = max(2, int(w * margin_ratio))
        pad_y = max(2, int(h * margin_ratio))

        x1 = max(0, int(x - pad_x))
        y1 = max(0, int(y - pad_y))
        x2 = min(img_w, int(x + w + pad_x))
        y2 = min(img_h, int(y + h + pad_y))

        # Ensure valid crop size
        if (x2 - x1) < min_size or (y2 - y1) < min_size:
            # Expand to minimum size if possible
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            half = min_size // 2
            x1 = max(0, cx - half)
            y1 = max(0, cy - half)
            x2 = min(img_w, x1 + min_size)
            y2 = min(img_h, y1 + min_size)

        crop = image.crop((x1, y1, x2, y2))
        crops.append(crop)

    return crops


def classify_detections_with_conch(
    image: Image.Image,
    detections: List[Dict[str, Any]],
    candidate_classes: List[Dict[str, str]],
    temperature: float = 0.05,
) -> List[Dict[str, Any]]:
    """
    Perform Zero-Shot Pathology Classification on a list of detections using CONCH.

    Args:
        image: Original full PIL image.
        detections: List of detection dicts (each containing 'id', 'bbox', 'polygon', etc.)
        candidate_classes: List of dicts with 'key', 'prompt', 'label', 'color'
        temperature: Softmax scaling temperature.

    Returns:
        List of classified detections with updated class_key, class_label, color,
        conch_confidence, and class_scores.
    """
    if not detections or not candidate_classes:
        return detections

    conch = ConchModelWrapper.get_instance()
    if not conch.is_loaded:
        success = conch.load()
        if not success:
            raise RuntimeError("CONCH model could not be initialized.")

    # 1. Prepare text prompts
    prompts_text = [
        c.get("prompt", c.get("label", c.get("name", c.get("key"))))
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
    temperature: float = 0.05,
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

    # 1. Extract crops for all detections
    crops = extract_crops_from_detections(image, detections)
    num_dets = len(detections)

    # 2. Extract UNI 1024-dim morphological features (if available)
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

    # 3. Extract CONCH 512-dim vision-language features
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

    # 4. If we have candidate classes from the active ontology / Gemini
    if candidate_classes and len(candidate_classes) > 0 and conch_features is not None:
        conch = ConchModelWrapper.get_instance()
        prompts_text = [
            c.get("prompt", c.get("label", c.get("name", c.get("key"))))
            for c in candidate_classes
        ]
        class_keys = [c.get("key") for c in candidate_classes]
        class_labels = [c.get("label", c.get("name", c.get("key"))) for c in candidate_classes]
        class_colors = [c.get("color", "#8b5cf6") for c in candidate_classes]

        text_embeddings = conch.encode_texts(prompts_text).float()  # (N_classes, 512)
        similarity_matrix = torch.matmul(conch_features.float(), text_embeddings.float().T).float()
        probs = F.softmax(similarity_matrix / temperature, dim=-1).float().cpu().numpy()
        similarities = similarity_matrix.float().cpu().numpy()

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

    # 5. Unsupervised Morphological Clustering fallback (when no ontology classes provided)
    # Combine UNI (morphology) and CONCH (semantics) into a normalized joint embedding
    feat_tensors = []
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
) -> List[List[float]]:
    """
    Extract 1024-dim UNI embeddings for all detections.
    """
    if not detections:
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


def get_pathology_models_status() -> Dict[str, Any]:
    """Get status of CONCH and UNI foundation models."""
    conch = ConchModelWrapper.get_instance()
    uni = UniModelWrapper.get_instance()

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
    }
