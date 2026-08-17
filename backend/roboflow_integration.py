"""
Roboflow integration module.

Handles:
  - Connection/authentication with Roboflow API
  - COCO JSON generation from curated frontend annotations (with polygons)
  - Dataset upload (images + annotations)
  - Training trigger
"""

import os
import io
import json
import logging
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv

logger = logging.getLogger("sam3-backend.roboflow")

# Load .env from project root
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")

ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY", "")
ROBOFLOW_WORKSPACE = os.getenv("ROBOFLOW_WORKSPACE", "")
ROBOFLOW_PROJECT = os.getenv("ROBOFLOW_PROJECT", "")


def _get_rf():
    """Lazy-initialise the Roboflow SDK and return the rf object."""
    import roboflow
    return roboflow.Roboflow(api_key=ROBOFLOW_API_KEY)


def check_connection() -> Dict[str, Any]:
    """Verify Roboflow connectivity and return workspace/project info."""
    if not ROBOFLOW_API_KEY:
        return {"connected": False, "error": "ROBOFLOW_API_KEY not set in .env"}

    try:
        rf = _get_rf()
        ws = rf.workspace(ROBOFLOW_WORKSPACE)
        project = ws.project(ROBOFLOW_PROJECT)
        return {
            "connected": True,
            "workspace": ROBOFLOW_WORKSPACE,
            "project": {
                "id": project.id,
                "name": project.name,
                "type": project.type,
            },
        }
    except Exception as e:
        logger.error(f"Roboflow connection failed: {e}")
        return {"connected": False, "error": str(e)}


def build_coco_json(annotations_payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert the frontend annotation payload into a COCO-format JSON dict.

    Expected input format (one image at a time):
    {
      "image_filename": "muestra_001.png",
      "image_width": 1920,
      "image_height": 1080,
      "classes": [
        {"id": 1, "name": "Espermatogonia B", "color": "#8b5cf6"},
        ...
      ],
      "annotations": [
        {
          "id": 1,
          "class_id": 1,
          "bbox": [x, y, w, h],
          "segmentation": [[x1,y1,x2,y2,...,xn,yn]],
          "area": 1234.5,
          "score": 0.87
        },
        ...
      ]
    }
    """
    classes = annotations_payload.get("classes", [])
    annotations = annotations_payload.get("annotations", [])
    filename = annotations_payload.get("image_filename", "image.png")
    width = annotations_payload.get("image_width", 0)
    height = annotations_payload.get("image_height", 0)

    # Build COCO categories
    categories = []
    for cls in classes:
        categories.append({
            "id": cls["id"],
            "name": cls["name"],
            "supercategory": "none",
        })

    # Build COCO image entry
    image_entry = {
        "id": 1,
        "file_name": filename,
        "width": width,
        "height": height,
    }

    # Build COCO annotations
    coco_annotations = []
    for ann in annotations:
        coco_ann = {
            "id": ann["id"],
            "image_id": 1,
            "category_id": ann["class_id"],
            "bbox": ann.get("bbox", [0, 0, 0, 0]),
            "area": ann.get("area", 0),
            "iscrowd": 0,
        }
        # Include segmentation polygons if available
        if "segmentation" in ann and ann["segmentation"]:
            coco_ann["segmentation"] = ann["segmentation"]
        else:
            # Fallback: create polygon from bbox
            x, y, w, h = ann.get("bbox", [0, 0, 0, 0])
            coco_ann["segmentation"] = [[x, y, x + w, y, x + w, y + h, x, y + h]]

        coco_annotations.append(coco_ann)

    return {
        "info": {
            "description": "SAM3 Histology Auto-Segmenter Export",
            "version": "1.0",
        },
        "licenses": [],
        "images": [image_entry],
        "annotations": coco_annotations,
        "categories": categories,
    }


def build_multi_image_coco(images_payload: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Build a single COCO JSON from multiple images' annotations.

    Each element in images_payload has the same structure as build_coco_json input.
    """
    all_categories = {}  # Deduplicate by (id, name)
    coco_images = []
    coco_annotations = []
    annotation_id_counter = 1

    for img_idx, img_data in enumerate(images_payload):
        image_id = img_idx + 1
        filename = img_data.get("image_filename", f"image_{image_id}.png")
        width = img_data.get("image_width", 0)
        height = img_data.get("image_height", 0)

        coco_images.append({
            "id": image_id,
            "file_name": filename,
            "width": width,
            "height": height,
        })

        for cls in img_data.get("classes", []):
            all_categories[cls["id"]] = {
                "id": cls["id"],
                "name": cls["name"],
                "supercategory": "none",
            }

        for ann in img_data.get("annotations", []):
            coco_ann = {
                "id": annotation_id_counter,
                "image_id": image_id,
                "category_id": ann["class_id"],
                "bbox": ann.get("bbox", [0, 0, 0, 0]),
                "area": ann.get("area", 0),
                "iscrowd": 0,
            }
            if "segmentation" in ann and ann["segmentation"]:
                coco_ann["segmentation"] = ann["segmentation"]
            else:
                x, y, w, h = ann.get("bbox", [0, 0, 0, 0])
                coco_ann["segmentation"] = [[x, y, x + w, y, x + w, y + h, x, y + h]]

            coco_annotations.append(coco_ann)
            annotation_id_counter += 1

    return {
        "info": {
            "description": "SAM3 Histology Auto-Segmenter Export",
            "version": "1.0",
        },
        "licenses": [],
        "images": coco_images,
        "annotations": coco_annotations,
        "categories": list(all_categories.values()),
    }


def upload_dataset_to_roboflow(
    images_data: List[Dict[str, Any]],
    image_files: Dict[str, bytes],
) -> Dict[str, Any]:
    """
    Upload a set of annotated images to Roboflow.

    Args:
        images_data: list of annotation payloads (one per image)
        image_files: dict mapping filename -> raw image bytes
    """
    if not ROBOFLOW_API_KEY:
        return {"success": False, "error": "ROBOFLOW_API_KEY not set"}

    try:
        # Create temporary directory with COCO structure
        tmp_dir = tempfile.mkdtemp(prefix="sam3_roboflow_")
        train_dir = os.path.join(tmp_dir, "train")
        os.makedirs(train_dir, exist_ok=True)

        # Save image files
        for filename, img_bytes in image_files.items():
            filepath = os.path.join(train_dir, filename)
            with open(filepath, "wb") as f:
                f.write(img_bytes)

        # Build and save COCO JSON
        coco_json = build_multi_image_coco(images_data)
        coco_path = os.path.join(train_dir, "_annotations.coco.json")
        with open(coco_path, "w") as f:
            json.dump(coco_json, f, indent=2)

        logger.info(f"Prepared dataset at {tmp_dir}: {len(image_files)} images, {len(coco_json['annotations'])} annotations")

        # Upload via Roboflow SDK
        rf = _get_rf()
        project = rf.workspace(ROBOFLOW_WORKSPACE).project(ROBOFLOW_PROJECT)

        project.upload_dataset(
            dataset_path=tmp_dir,
            num_workers=4,
            dataset_format="coco",
            project_license="MIT",
            project_type="instance-segmentation",
        )

        # Cleanup
        shutil.rmtree(tmp_dir, ignore_errors=True)

        return {
            "success": True,
            "images_uploaded": len(image_files),
            "annotations_uploaded": len(coco_json["annotations"]),
            "categories": len(coco_json["categories"]),
        }

    except Exception as e:
        logger.error(f"Upload to Roboflow failed: {e}", exc_info=True)
        # Cleanup on error
        if 'tmp_dir' in locals():
            shutil.rmtree(tmp_dir, ignore_errors=True)
        return {"success": False, "error": str(e)}


def trigger_training(model_type: str = "yolov8", auto_generate_version: bool = True) -> Dict[str, Any]:
    """Trigger model training on Roboflow (generates a version automatically if needed)."""
    if not ROBOFLOW_API_KEY:
        return {"success": False, "error": "ROBOFLOW_API_KEY no configurada en .env"}

    try:
        rf = _get_rf()
        project = rf.workspace(ROBOFLOW_WORKSPACE).project(ROBOFLOW_PROJECT)

        versions = []
        try:
            versions = project.versions()
        except Exception as ve:
            logger.warning(f"Could not fetch existing versions: {ve}")

        version_obj = None
        if not versions:
            if auto_generate_version:
                logger.info("No dataset versions found. Generating a new dataset version in Roboflow...")
                settings = {
                    "preprocessing": {
                        "auto-orient": {"enabled": True},
                        "resize": {"width": 640, "height": 640, "format": "stretch"}
                    },
                    "augmentation": {}
                }
                try:
                    version_obj = project.generate_version(settings=settings)
                except Exception as ge:
                    logger.error(f"Failed to generate version: {ge}")
                    return {"success": False, "error": f"No se pudo generar la versión del dataset: {str(ge)}"}
            else:
                return {"success": False, "error": "No hay versiones de dataset generadas en Roboflow. Genera una primero."}
        else:
            version_obj = versions[-1]

        # Trigger train
        version_obj.train(model_type=model_type)

        version_num = getattr(version_obj, "version", str(version_obj))
        return {
            "success": True,
            "version": version_num,
            "model_type": model_type,
            "message": f"Entrenamiento YOLO ({model_type}) iniciado en Roboflow para la Versión {version_num}.",
        }

    except Exception as e:
        logger.error(f"Training trigger failed: {e}", exc_info=True)
        return {"success": False, "error": str(e)}

