"""
Lunit DINO ViT-S/8 Histopathology Foundation Model Wrapper.

Provides 384-dim self-supervised embeddings trained on 33 million H&E histology patches
using the DINO self-supervised learning framework (CVPR 2023).

Source: https://huggingface.co/1aurent/vit_small_patch8_224.lunit_dino
Paper: "Benchmarking Self-Supervised Learning on Diverse Pathology Datasets" (CVPR 2023, Lunit)
"""

import logging
import os
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

logger = logging.getLogger("sam3-backend")


def _get_aux_device() -> torch.device:
    """Resolve device for auxiliary pathology models (same logic as pathology_models.py)."""
    env_dev = os.getenv("PATHOLOGY_DEVICE", "").lower().strip()
    if env_dev == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if torch.cuda.is_available():
        try:
            total_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            if total_mem_gb >= 5.5:
                return torch.device("cuda")
        except (RuntimeError, AssertionError) as cuda_err:
            logger.debug(f"Could not query CUDA device properties: {cuda_err}")
        return torch.device("cuda")
    return torch.device("cpu")


class LunitDinoModelWrapper:
    """
    Singleton wrapper for Lunit DINO ViT-S/8 (384-dim) histopathology foundation model.

    Trained on 33 million H&E histology patches using DINO self-supervised learning.
    Generates 384-dimensional normalized embeddings optimized for histomorphological
    feature extraction across diverse tissue types.

    Architecture: ViT-Small/8 (patch size 8 → higher spatial resolution than /16)
    Parameters: ~21M (~84MB in fp16)
    Input: 224×224 RGB images with ImageNet normalization
    Output: 384-dim L2-normalized embeddings
    """

    _instance: Optional["LunitDinoModelWrapper"] = None

    EMBEDDING_DIM: int = 384
    MODEL_ID: str = "hf-hub:1aurent/vit_small_patch8_224.lunit_dino"

    def __init__(self) -> None:
        self.model: Optional[Any] = None
        self.transform: Optional[Any] = None
        self.is_loaded: bool = False
        self.device: torch.device = _get_aux_device()

    @classmethod
    def get_instance(cls) -> "LunitDinoModelWrapper":
        """Get or create the singleton instance."""
        if cls._instance is None:
            cls._instance = LunitDinoModelWrapper()
        return cls._instance

    def load(self, force_reload: bool = False) -> bool:
        """
        Load Lunit DINO model weights via timm from HuggingFace Hub.

        Uses automatic OOM fallback to CPU if CUDA runs out of memory.
        No authentication required (public model).
        """
        if self.is_loaded and not force_reload:
            return True

        try:
            import timm
            from timm.data import resolve_data_config
            from timm.data.transforms_factory import create_transform

            logger.info(f"Loading Lunit DINO model ({self.MODEL_ID}) on {self.device}...")

            model = timm.create_model(
                self.MODEL_ID,
                pretrained=True,
                num_classes=0,  # Remove classification head → pure feature extractor
            )

            # Move to target device with fp16 on CUDA for VRAM efficiency
            try:
                if self.device.type == "cuda":
                    model = model.to(self.device).half()
                else:
                    model = model.to(self.device)
            except (torch.cuda.OutOfMemoryError, RuntimeError) as cuda_err:
                if "out of memory" in str(cuda_err).lower() or isinstance(
                    cuda_err, torch.cuda.OutOfMemoryError
                ):
                    logger.warning(
                        f"CUDA OOM while moving Lunit DINO to {self.device}, "
                        f"falling back to CPU: {cuda_err}"
                    )
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    self.device = torch.device("cpu")
                    model = model.to(self.device).float()
                else:
                    raise cuda_err

            model.eval()

            # Resolve model-specific transforms (normalization, resize, crop)
            try:
                data_config = resolve_data_config(model.pretrained_cfg, model=model)
                transform = create_transform(**data_config, is_training=False)
            except Exception:
                # Fallback to standard ImageNet normalization
                transform = transforms.Compose([
                    transforms.Resize(224),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225),
                    ),
                ])

            self.model = model
            self.transform = transform
            self.is_loaded = True

            param_count = sum(p.numel() for p in model.parameters()) / 1e6
            logger.info(
                f"Lunit DINO loaded successfully on {self.device} "
                f"({param_count:.1f}M params, {self.EMBEDDING_DIM}d embeddings)."
            )
            return True

        except Exception as e:
            logger.error(f"Failed to load Lunit DINO model: {e}", exc_info=True)
            self.is_loaded = False
            return False

    def encode_crops(
        self, crops: List[Image.Image], batch_size: int = 32
    ) -> torch.Tensor:
        """
        Encode a list of PIL image crops into normalized 384-dim Lunit DINO embeddings.

        Args:
            crops: List of PIL.Image crops (any size, will be resized to 224×224).
            batch_size: Batch size for GPU inference.

        Returns:
            Tensor of shape (N, 384) with L2-normalized embeddings on CPU.
        """
        if not self.is_loaded:
            self.load()
        if not self.is_loaded or self.model is None or self.transform is None:
            raise RuntimeError("Lunit DINO model is not available.")

        dev_type = "cuda" if self.device.type == "cuda" else "cpu"
        all_embeddings: List[torch.Tensor] = []

        for i in range(0, len(crops), batch_size):
            batch_crops = crops[i : i + batch_size]
            tensors = [self.transform(c.convert("RGB")) for c in batch_crops]
            batch_tensor = torch.stack(tensors).to(self.device)
            if dev_type == "cuda":
                batch_tensor = batch_tensor.half()

            with torch.inference_mode(), torch.autocast(
                device_type=dev_type, enabled=(dev_type == "cuda")
            ):
                features = self.model(batch_tensor)
                # Handle different output formats
                if isinstance(features, (tuple, list)):
                    features = features[0]
                elif isinstance(features, dict):
                    features = features.get(
                        "embeddings",
                        features.get("logits", list(features.values())[0]),
                    )
                # If 3D (batch, tokens, dim), take CLS token
                if hasattr(features, "ndim") and features.ndim == 3:
                    features = features[:, 0]
                features = F.normalize(features.float(), dim=-1)
                all_embeddings.append(features.float().cpu())

        if not all_embeddings:
            return torch.empty((0, self.EMBEDDING_DIM), dtype=torch.float32)

        return torch.cat(all_embeddings, dim=0).float()

    def offload_to_cpu(self) -> None:
        """Move model to CPU to free GPU VRAM for other models."""
        if self.model is not None and self.device.type == "cuda":
            try:
                self.model.to("cpu").float()
                self.device = torch.device("cpu")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                logger.info("Lunit DINO offloaded to CPU; GPU VRAM released.")
            except Exception as e:
                logger.warning(f"Error offloading Lunit DINO to CPU: {e}")

    def move_to_gpu(self) -> None:
        """Move model back to GPU for fast inference."""
        if self.model is not None and self.device.type == "cpu" and torch.cuda.is_available():
            try:
                self.model.to("cuda").half()
                self.device = torch.device("cuda")
                logger.info("Lunit DINO moved to GPU for fast inference.")
            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                logger.warning(f"Cannot move Lunit DINO to GPU (OOM): {e}")
