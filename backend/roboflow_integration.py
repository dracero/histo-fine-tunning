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


def _get_rf_project() -> Any:
    """
    Get or create the Roboflow project instance.
    Resolves project names (e.g. 'agent') to their actual Roboflow URL slug (e.g. 'agent-m51wr').
    """
    import requests
    rf = _get_rf()
    workspace = rf.workspace(ROBOFLOW_WORKSPACE)
    target = ROBOFLOW_PROJECT.strip()

    # Query workspace projects list to find real slug
    try:
        url = f"https://api.roboflow.com/{ROBOFLOW_WORKSPACE}?api_key={ROBOFLOW_API_KEY}"
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            projects = resp.json().get("workspace", {}).get("projects", [])
            for p_info in projects:
                p_id = str(p_info.get("id", ""))
                p_slug = p_id.rsplit("/", 1)[-1]
                p_name = str(p_info.get("name", ""))
                if (
                    target.lower() in (p_id.lower(), p_slug.lower(), p_name.lower())
                    or p_slug.lower().startswith(target.lower() + "-")
                    or p_name.lower().startswith(target.lower())
                ):
                    logger.info(f"Resolved Roboflow project target '{target}' to slug '{p_slug}'")
                    return workspace.project(p_slug)
    except Exception as e:
        logger.warning(f"Could not resolve Roboflow project slug dynamically: {e}")

    # Fallback to direct lookup or create project if it does not exist
    try:
        return workspace.project(target)
    except Exception:
        logger.info(f"Creating new Roboflow project '{target}' in workspace '{ROBOFLOW_WORKSPACE}'...")
        return workspace.create_project(
            project_name=target,
            project_type="object-detection",
            project_license="MIT",
            annotation="cells",
        )


SUPPORTED_MODEL_TYPES = [
    {"id": "yolov8-obb", "name": "YOLOv8-OBB (Oriented Bounding Boxes)"},
    {"id": "yolov8", "name": "YOLOv8 PyTorch / Ultralytics"},
    {"id": "yolov11", "name": "YOLOv11 PyTorch"},
    {"id": "yolov9", "name": "YOLOv9 PyTorch"},
    {"id": "yolov7", "name": "YOLOv7 PyTorch"},
    {"id": "yolov5", "name": "YOLOv5 PyTorch"},
    {"id": "coco", "name": "COCO JSON Format"},
    {"id": "coco-segmentation", "name": "COCO Instance Segmentation"},
    {"id": "pascal_voc", "name": "Pascal VOC XML"},
    {"id": "yolact", "name": "YOLACT Segmentation"},
    {"id": "tfrecord", "name": "TensorFlow TFRecord"},
    {"id": "coreml", "name": "Apple CoreML"},
]


def check_connection() -> Dict[str, Any]:
    """Verify Roboflow connectivity and return workspace/project info."""
    if not ROBOFLOW_API_KEY:
        return {"connected": False, "error": "ROBOFLOW_API_KEY not set in .env"}

    try:
        project = _get_rf_project()
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


def get_roboflow_models_and_versions() -> Dict[str, Any]:
    """Retrieve available Roboflow project dataset versions and supported model architectures."""
    if not ROBOFLOW_API_KEY:
        return {
            "connected": False,
            "error": "ROBOFLOW_API_KEY no configurada en .env",
            "versions": [],
            "supported_models": SUPPORTED_MODEL_TYPES,
        }

    try:
        project = _get_rf_project()

        versions_info = []
        try:
            versions = project.versions()
            for v in versions:
                v_num = str(getattr(v, "version", v))
                v_id = getattr(v, "id", f"{project.id}/{v_num}")
                versions_info.append({
                    "version": v_num,
                    "id": v_id,
                    "name": f"Versión {v_num}",
                })
        except Exception as ve:
            logger.warning(f"Could not list project versions: {ve}")

        return {
            "connected": True,
            "project_id": project.id,
            "project_name": project.name,
            "versions": versions_info,
            "supported_models": SUPPORTED_MODEL_TYPES,
        }
    except Exception as e:
        logger.error(f"Error fetching Roboflow models/versions: {e}")
        return {
            "connected": False,
            "error": str(e),
            "versions": [],
            "supported_models": SUPPORTED_MODEL_TYPES,
        }


def _normalize_segmentation(seg: Any, bbox: List[float]) -> List[List[float]]:
    """Ensure segmentation is formatted as valid COCO 2D list of float polygon coordinates."""
    if not seg:
        x, y, w, h = bbox
        return [[float(x), float(y), float(x + w), float(y), float(x + w), float(y + h), float(x), float(y + h)]]

    # Case 1: Flat 1D list of floats [x1, y1, x2, y2, ...]
    if isinstance(seg, list) and len(seg) > 0 and isinstance(seg[0], (int, float)):
        return [[float(v) for v in seg]]

    # Case 2: List of coordinate pairs [[x1, y1], [x2, y2], ...]
    if isinstance(seg, list) and len(seg) > 0 and isinstance(seg[0], list):
        if len(seg[0]) == 2 and isinstance(seg[0][0], (int, float)):
            flat = []
            for pt in seg:
                flat.extend([float(pt[0]), float(pt[1])])
            return [flat]
        # Case 3: List of polygons [[x1, y1, x2, y2, ...]] or [[[x1, y1], ...]]
        res = []
        for poly in seg:
            if isinstance(poly, list) and len(poly) > 0:
                if isinstance(poly[0], (int, float)):
                    res.append([float(v) for v in poly])
                elif isinstance(poly[0], list) and len(poly[0]) == 2:
                    flat = []
                    for pt in poly:
                        flat.extend([float(pt[0]), float(pt[1])])
                    res.append(flat)
        if res:
            return res

    x, y, w, h = bbox
    return [[float(x), float(y), float(x + w), float(y), float(x + w), float(y + h), float(x), float(y + h)]]


def build_coco_json(annotations_payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert the frontend annotation payload into a COCO-format JSON dict.
    """
    classes = annotations_payload.get("classes", [])
    annotations = annotations_payload.get("annotations", [])
    filename = annotations_payload.get("image_filename", "image.png")
    width = int(annotations_payload.get("image_width", 0))
    height = int(annotations_payload.get("image_height", 0))

    # Build COCO categories
    categories = []
    seen_cat_ids = set()
    for idx, cls in enumerate(classes):
        c_id = cls.get("id") if (isinstance(cls, dict) and cls.get("id") is not None) else (idx + 1)
        c_name = cls.get("name", cls.get("label", f"class_{c_id}")) if isinstance(cls, dict) else str(cls)
        try:
            c_id_int = int(c_id)
        except (ValueError, TypeError):
            c_id_int = idx + 1

        if c_id_int not in seen_cat_ids:
            seen_cat_ids.add(c_id_int)
            categories.append({
                "id": c_id_int,
                "name": str(c_name),
                "supercategory": "none",
            })

    # If no categories provided, create a default one
    if not categories:
        categories.append({
            "id": 1,
            "name": "default_class",
            "supercategory": "none",
        })

    # Build COCO image entry
    image_entry = {
        "id": 1,
        "file_name": os.path.basename(filename),
        "width": width,
        "height": height,
    }

    # Build COCO annotations
    coco_annotations = []
    for ann_idx, ann in enumerate(annotations):
        c_id = ann.get("class_id", ann.get("category_id", 1))
        try:
            c_id_int = int(c_id) if c_id is not None else 1
        except (ValueError, TypeError):
            c_id_int = 1

        raw_bbox = ann.get("bbox", [0, 0, 0, 0])
        x, y, w, h = raw_bbox if len(raw_bbox) == 4 else [0, 0, 0, 0]
        bbox = [float(x), float(y), max(1.0, float(w)), max(1.0, float(h))]

        coco_ann = {
            "id": int(ann.get("id", ann_idx + 1)),
            "image_id": 1,
            "category_id": c_id_int,
            "bbox": bbox,
            "area": float(ann.get("area", bbox[2] * bbox[3])),
            "segmentation": _normalize_segmentation(ann.get("segmentation"), bbox),
            "iscrowd": 0,
        }

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
    all_categories = {}  # Deduplicate by id
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
            "file_name": os.path.basename(filename),
            "width": width,
            "height": height,
        })

        for idx, cls in enumerate(img_data.get("classes", [])):
            c_id = cls.get("id") if (isinstance(cls, dict) and cls.get("id") is not None) else (idx + 1)
            c_name = cls.get("name", cls.get("label", f"class_{c_id}")) if isinstance(cls, dict) else str(cls)
            try:
                c_id_int = int(c_id)
            except (ValueError, TypeError):
                c_id_int = idx + 1

            all_categories[c_id_int] = {
                "id": c_id_int,
                "name": str(c_name),
                "supercategory": "none",
            }

        for ann in img_data.get("annotations", []):
            c_id = ann.get("class_id", ann.get("category_id", 1))
            try:
                c_id_int = int(c_id) if c_id is not None else 1
            except (ValueError, TypeError):
                c_id_int = 1

            coco_ann = {
                "id": annotation_id_counter,
                "image_id": image_id,
                "category_id": c_id_int,
                "bbox": ann.get("bbox", [0, 0, 0, 0]),
                "area": float(ann.get("area", 0)),
                "iscrowd": 0,
            }
            if "segmentation" in ann and ann["segmentation"]:
                coco_ann["segmentation"] = ann["segmentation"]
            else:
                x, y, w, h = ann.get("bbox", [0, 0, 0, 0])
                coco_ann["segmentation"] = [[x, y, x + w, y, x + w, y + h, x, y + h]]

            coco_annotations.append(coco_ann)
            annotation_id_counter += 1

    if not all_categories:
        all_categories[1] = {
            "id": 1,
            "name": "default_class",
            "supercategory": "none",
        }

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
    Upload a set of annotated images to Roboflow using project.upload().

    Args:
        images_data: list of annotation payloads (one per image)
        image_files: dict mapping filename -> raw image bytes
    """
    if not ROBOFLOW_API_KEY:
        return {"success": False, "error": "ROBOFLOW_API_KEY not set"}

    tmp_dir = None
    try:
        tmp_dir = tempfile.mkdtemp(prefix="sam3_roboflow_")
        project = _get_rf_project()

        uploaded_count = 0
        total_annotations = 0

        # Process and upload each image
        for img_data in images_data:
            filename = img_data.get("image_filename", "")
            if not filename or filename not in image_files:
                # Try finding matching file by basename if full path differs
                base = os.path.basename(filename)
                matching_file = next((k for k in image_files.keys() if os.path.basename(k) == base), None)
                if matching_file:
                    filename = matching_file
                elif image_files:
                    # Fallback to first available file if only 1 image in batch
                    filename = list(image_files.keys())[0]
                else:
                    logger.warning(f"Image file {filename} not provided in payload, skipping.")
                    continue

            img_bytes = image_files[filename]
            clean_name = os.path.basename(filename)
            img_path = os.path.join(tmp_dir, clean_name)
            with open(img_path, "wb") as f:
                f.write(img_bytes)

            # Auto-detect real image width and height if missing or zero
            if not img_data.get("image_width") or not img_data.get("image_height"):
                try:
                    from PIL import Image
                    with Image.open(io.BytesIO(img_bytes)) as pimg:
                        img_data["image_width"] = pimg.width
                        img_data["image_height"] = pimg.height
                except Exception as ie:
                    logger.warning(f"Could not auto-detect image dimensions for {clean_name}: {ie}")

            # Build single-image COCO JSON
            coco_dict = build_coco_json(img_data)

            # Ensure proper annotation filename matching base image name (e.g. image1.json instead of image1.png.json)
            name_base = os.path.splitext(clean_name)[0]
            ann_path = os.path.join(tmp_dir, f"{name_base}.json")
            with open(ann_path, "w") as f:
                json.dump(coco_dict, f, indent=2)

            logger.info(f"Uploading {clean_name} with annotation {name_base}.json to Roboflow ({len(coco_dict.get('annotations', []))} annotations)...")

            # Upload via official Roboflow project.upload method
            project.upload(
                image_path=img_path,
                annotation_path=ann_path,
                split="train",
                num_retry_uploads=3,
            )

            uploaded_count += 1
            total_annotations += len(coco_dict.get("annotations", []))

        # Cleanup temporary files
        shutil.rmtree(tmp_dir, ignore_errors=True)

        return {
            "success": True,
            "images_uploaded": uploaded_count,
            "annotations_uploaded": total_annotations,
        }

    except Exception as e:
        logger.error(f"Upload to Roboflow failed: {e}", exc_info=True)
        if tmp_dir and os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir, ignore_errors=True)
        return {"success": False, "error": str(e)}


def trigger_training(model_type: str = "yolov8", version: Optional[str] = None) -> Dict[str, Any]:
    """
    Trigger model training on Roboflow for a specified model architecture and dataset version.

    If version is provided (e.g. "1"), trains that specific version.
    If version is None or "new", generates a new expanded dataset version.
    """
    if not ROBOFLOW_API_KEY:
        return {"success": False, "error": "ROBOFLOW_API_KEY no configurada en .env"}

    try:
        project = _get_rf_project()

        version_obj = None
        settings = {
            "preprocessing": {
                "auto-orient": True,
                "resize": {"width": 640, "height": 640, "format": "Stretch to"}
            },
            "augmentation": {}
        }

        if version and version not in ("new", "auto", "latest"):
            logger.info(f"Targeting specified Roboflow version: {version}")
            try:
                version_obj = project.version(version)
            except Exception as ve:
                logger.warning(f"Could not fetch version {version} directly: {ve}")
                versions = project.versions()
                for v in versions:
                    if str(getattr(v, "version", "")) == str(version):
                        version_obj = v
                        break
            if not version_obj:
                return {"success": False, "error": f"La versión '{version}' no existe en Roboflow."}
        else:
            logger.info("Generating a new expanded dataset version in Roboflow...")
            try:
                version_obj = project.generate_version(settings=settings)
            except Exception as ge:
                logger.warning(f"Could not generate new version, trying latest version: {ge}")
                try:
                    versions = project.versions()
                    if versions:
                        version_obj = versions[-1]
                except Exception:
                    return {"success": False, "error": f"Error al acceder a versiones de Roboflow: {str(ge)}"}

        # Ensure version_obj is a Version instance with .train() method
        if not hasattr(version_obj, "train"):
            v_str = str(version_obj)
            logger.info(f"Resolving Version object for version string '{v_str}'...")
            version_obj = project.version(v_str)

        # Trigger training on the target version
        version_obj.train(model_type=model_type)

        version_num = getattr(version_obj, "version", str(version_obj))
        return {
            "success": True,
            "version": version_num,
            "model_type": model_type,
            "message": f"Entrenamiento ({model_type}) iniciado con éxito en Roboflow para la versión {version_num}.",
        }

    except Exception as e:
        logger.error(f"Training trigger failed: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


def export_dataset_version(
    model_format: str = "yolov8",
    version: Optional[str] = None,
    download_local: bool = True,
) -> Dict[str, Any]:
    """
    Generate/freeze a Roboflow dataset version snapshot and download it locally
    for local training on GPU without requiring Roboflow cloud training credits.
    """
    if not ROBOFLOW_API_KEY:
        return {"success": False, "error": "ROBOFLOW_API_KEY no configurada en .env"}

    try:
        project = _get_rf_project()
        version_obj = None

        settings = {
            "preprocessing": {
                "auto-orient": True,
                "resize": {"width": 640, "height": 640, "format": "Stretch to"}
            },
            "augmentation": {}
        }

        if version and version not in ("new", "auto", "latest"):
            logger.info(f"Obteniendo versión existente {version} de Roboflow...")
            v_str = str(version)
            version_obj = project.version(v_str)
        else:
            logger.info("Generando nueva versión congelada del dataset en Roboflow...")
            res = project.generate_version(settings=settings)
            v_str = str(res)
            version_obj = project.version(v_str)

        version_num = getattr(version_obj, "version", str(version_obj))

        # Extract workspace name and real project slug
        workspace_name = ROBOFLOW_WORKSPACE
        project_slug = getattr(project, "id", f"{workspace_name}/{ROBOFLOW_PROJECT}").rsplit("/", 1)[-1]

        # Generate exact python snippet requested by user
        key_var = "api_key"
        python_snippet = (
            f"!pip install roboflow\n\n"
            f"from roboflow import Roboflow\n\n"
            f'rf = Roboflow({key_var}="{ROBOFLOW_API_KEY}")\n'
            f'project = rf.workspace("{workspace_name}").project("{project_slug}")\n'
            f"version = project.version({version_num})\n"
            f'dataset = version.download("{model_format}")'
        )

        export_dir = os.path.abspath(os.path.join("datasets", f"roboflow_v{version_num}_{model_format}"))

        if download_local:
            try:
                os.makedirs(export_dir, exist_ok=True)
                logger.info(f"Descargando versión {version_num} ({model_format}) en {export_dir}...")
                dl_res = version_obj.download(model_format=model_format, location=export_dir, overwrite=True)
                export_dir = dl_res.location
            except Exception as dle:
                logger.warning(f"Could not download dataset locally: {dle}")

        data_yaml_path = os.path.join(export_dir, "data.yaml")
        cli_command = f"yolo task=detect mode=train model={model_format}n.pt data={data_yaml_path} epochs=50 device=0"

        return {
            "success": True,
            "version": version_num,
            "model_format": model_format,
            "workspace": workspace_name,
            "project_slug": project_slug,
            "api_key": ROBOFLOW_API_KEY,
            "export_dir": export_dir,
            "data_yaml": data_yaml_path if os.path.exists(data_yaml_path) else export_dir,
            "cli_command": cli_command,
            "python_snippet": python_snippet,
            "message": f"Versión unificada v{version_num} generada con éxito.",
        }

    except Exception as e:
        logger.error(f"Error exporting dataset version: {e}", exc_info=True)
        return {"success": False, "error": str(e)}
