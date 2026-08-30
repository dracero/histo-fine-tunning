"""
SAM 3 + Virchow 2 Hybrid Segmentation & Feature Extraction Pipeline.

Workflow:
1. Segment cellular / tissue instances using SAM 3 (ultralytics).
2. Extract high-resolution crops with margin.
3. Compute 1280-dim histopathology embeddings using Paige AI Virchow 2 (ViT-H/14).
4. Save structured metadata and visual overlay annotations.
"""

import os
import shutil
import argparse
import logging
from typing import List, Tuple, Dict, Any, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

try:
    import timm
    from timm.data import resolve_data_config
    from timm.data.transforms_factory import create_transform
    from torchvision import transforms
except ImportError:
    timm = None

try:
    from ultralytics.models.sam import SAM3SemanticPredictor
    from ultralytics.utils.plotting import Annotator, colors
except ImportError:
    SAM3SemanticPredictor = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("sam3-virchow2")


class SamVirchowPipeline:
    """End-to-end pipeline combining SAM 3 segmenter and Virchow 2 pathology foundation model."""

    def __init__(
        self,
        sam_weights_path: str = "sam3.pt",
        virchow_model_id: str = "paige-ai/Virchow2",
        device: Optional[str] = None,
    ):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.sam_weights_path = sam_weights_path
        self.virchow_model_id = virchow_model_id
        self.virchow_model = None
        self.virchow_transform = None
        self.predictor = None

        logger.info(f"Initialized SamVirchowPipeline on device: {self.device}")

    def load_virchow(self) -> None:
        """Load Paige AI Virchow 2 ViT-Huge (1280-dim) model."""
        if self.virchow_model is not None:
            return

        logger.info(f"Loading Virchow 2 ({self.virchow_model_id})...")
        model = timm.create_model(
            f"hf-hub:{self.virchow_model_id}",
            pretrained=True,
            mlp_layer=timm.layers.SwiGLUPacked,
            act_layer=torch.nn.SiLU,
        )

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
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ])

        self.virchow_model = model
        self.virchow_transform = transform
        logger.info("✅ Virchow 2 loaded successfully.")

    def load_sam3(self, conf: float = 0.10) -> None:
        """Initialize SAM 3 Semantic Predictor."""
        logger.info(f"Initializing SAM 3 Predictor from {self.sam_weights_path}...")
        overrides = {
            "conf": conf,
            "task": "segment",
            "mode": "predict",
            "model": self.sam_weights_path,
            "quantize": 16 if self.device.type == "cuda" else 32,
            "save": False,
        }
        self.predictor = SAM3SemanticPredictor(overrides=overrides)
        logger.info("✅ SAM 3 Predictor ready.")

    def run(
        self,
        image_path: str,
        text_prompts: Optional[List[str]] = None,
        bboxes: Optional[List[List[int]]] = None,
        margin_ratio: float = 0.1,
        batch_size: int = 16,
        output_image_path: Optional[str] = "output_sam_virchow.png",
    ) -> Dict[str, Any]:
        """Execute full segmentation + feature extraction."""
        if self.virchow_model is None:
            self.load_virchow()
        if self.predictor is None:
            self.load_sam3()

        pil_img = Image.open(image_path).convert("RGB")
        w_orig, h_orig = pil_img.size

        # 1. Set image in SAM 3
        self.predictor.set_image(image_path)

        # 2. Inquire SAM 3
        if bboxes is not None and len(bboxes) > 0:
            results = self.predictor(bboxes=bboxes)
        elif text_prompts is not None and len(text_prompts) > 0:
            results = self.predictor(text=text_prompts)
        else:
            results = self.predictor(text=["cell", "nucleus"])

        # 3. Extract crops
        crops = []
        crop_metadata = []

        for r in results:
            if r.masks is None or len(r.masks) == 0:
                continue

            boxes = r.boxes.xyxy.cpu().numpy().astype(int)
            confs = r.boxes.conf.cpu().numpy()
            clses = r.boxes.cls.cpu().numpy().astype(int)

            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box
                w, h = x2 - x1, y2 - y1
                x1_m = max(0, int(x1 - margin_ratio * w))
                y1_m = max(0, int(y1 - margin_ratio * h))
                x2_m = min(w_orig, int(x2 + margin_ratio * w))
                y2_m = min(h_orig, int(y2 + margin_ratio * h))

                crop = pil_img.crop((x1_m, y1_m, x2_m, y2_m))
                crops.append(crop)
                crop_metadata.append({
                    "id": i,
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "margin_bbox": [x1_m, y1_m, x2_m, y2_m],
                    "conf": float(confs[i]),
                    "cls": int(clses[i]),
                })

        logger.info(f"Segmented {len(crops)} instances.")

        # 4. Inquire Virchow 2 embeddings
        embeddings_list = []
        if len(crops) > 0:
            for i in range(0, len(crops), batch_size):
                batch_crops = crops[i : i + batch_size]
                tensors = [self.virchow_transform(c) for c in batch_crops]
                batch_tensor = torch.stack(tensors).to(self.device)
                if self.device.type == "cuda":
                    batch_tensor = batch_tensor.half()

                with torch.inference_mode():
                    feat = self.virchow_model(batch_tensor)
                    if hasattr(feat, "ndim") and feat.ndim == 3:
                        feat = feat[:, 0]  # CLS token (1280d)
                    feat = F.normalize(feat.float(), dim=-1)
                    embeddings_list.append(feat.cpu())

            all_embeddings = torch.cat(embeddings_list, dim=0).numpy()
        else:
            all_embeddings = np.empty((0, 1280), dtype=np.float32)

        # 5. Visual output annotation
        if output_image_path and len(results) > 0:
            im_cv = cv2.imread(image_path)
            for r in results:
                if r.masks is None or len(r.masks) == 0:
                    continue
                annotator = Annotator(im_cv.copy(), pil=False)
                mask_data = r.masks.data.cpu().numpy()
                mask_colors = [colors(i, True) for i in range(len(mask_data))]
                annotator.masks(mask_data, mask_colors)

                for i, box in enumerate(r.boxes):
                    xyxy = box.xyxy[0].cpu().numpy().astype(int)
                    conf_val = float(box.conf.item())
                    annotator.box_label(xyxy, f"Cell {i} ({conf_val:.2f})", color=mask_colors[i])

                im_result = annotator.result()
                cv2.imwrite(output_image_path, im_result)
                logger.info(f"✅ Annotated result saved to {output_image_path}")

        return {
            "num_instances": len(crops),
            "metadata": crop_metadata,
            "embeddings": all_embeddings,
            "output_image": output_image_path,
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SAM 3 + Virchow 2 Pipeline")
    parser.add_argument("--image", type=str, required=True, help="Path to input histology image")
    parser.add_argument("--sam_weights", type=str, default="sam3.pt", help="Path to sam3.pt")
    parser.add_argument("--output", type=str, default="output_annotated.png", help="Output annotated image")
    args = parser.parse_args()

    pipeline = SamVirchowPipeline(sam_weights_path=args.sam_weights)
    res = pipeline.run(args.image, output_image_path=args.output)
    print(f"Extraction completed. Embeddings shape: {res['embeddings'].shape}")
