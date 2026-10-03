"""
COCO JSON serialization utilities for histological annotations.
Replaces Roboflow dependency with lightweight, self-contained COCO export.
"""

from typing import Any, Dict, List, Optional


def normalize_segmentation(seg: Any, bbox: List[float]) -> List[List[float]]:
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
    """Convert the frontend annotation payload into a standard COCO-format JSON dict."""
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
                "name": c_name,
                "supercategory": "cell_or_structure",
            })

    image_obj = {
        "id": 1,
        "file_name": filename,
        "width": width,
        "height": height,
    }

    coco_annotations = []
    for idx, ann in enumerate(annotations):
        x = float(ann.get("x", 0))
        y = float(ann.get("y", 0))
        w = float(ann.get("width", ann.get("w", 0)))
        h = float(ann.get("height", ann.get("h", 0)))
        bbox = [x, y, w, h]

        cat_id = ann.get("class_id", ann.get("category_id", 1))
        try:
            cat_id_int = int(cat_id)
        except (ValueError, TypeError):
            cat_id_int = 1

        raw_seg = ann.get("segmentation", [])
        norm_seg = normalize_segmentation(raw_seg, bbox)

        area = float(ann.get("area", w * h))

        coco_annotations.append({
            "id": idx + 1,
            "image_id": 1,
            "category_id": cat_id_int,
            "bbox": bbox,
            "segmentation": norm_seg,
            "area": area,
            "iscrowd": 0,
            "scale": ann.get("scale", "micro"),
            "score": float(ann.get("score", 1.0)),
        })

    return {
        "info": {
            "description": "Histological Dataset (Meta SAM 3 / Cellpose / Gemini 3.5)",
            "version": "1.0",
            "year": 2026,
        },
        "images": [image_obj],
        "annotations": coco_annotations,
        "categories": categories,
    }


def build_multi_image_coco(images_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Combine annotations from multiple images into a unified multi-image COCO JSON dataset."""
    all_images = []
    all_annotations = []
    all_categories = []
    cat_name_to_id = {}
    ann_counter = 1

    for img_idx, img_payload in enumerate(images_data, start=1):
        filename = img_payload.get("image_filename", f"image_{img_idx}.png")
        width = int(img_payload.get("image_width", 0))
        height = int(img_payload.get("image_height", 0))

        all_images.append({
            "id": img_idx,
            "file_name": filename,
            "width": width,
            "height": height,
        })

        for cls in img_payload.get("classes", []):
            c_name = cls.get("name", cls.get("label", "")) if isinstance(cls, dict) else str(cls)
            if c_name and c_name not in cat_name_to_id:
                new_id = len(cat_name_to_id) + 1
                cat_name_to_id[c_name] = new_id
                all_categories.append({
                    "id": new_id,
                    "name": c_name,
                    "supercategory": "cell_or_structure",
                })

        for ann in img_payload.get("annotations", []):
            x = float(ann.get("x", 0))
            y = float(ann.get("y", 0))
            w = float(ann.get("width", ann.get("w", 0)))
            h = float(ann.get("height", ann.get("h", 0)))
            bbox = [x, y, w, h]

            cls_name = ann.get("class_name", ann.get("label", ""))
            if cls_name in cat_name_to_id:
                cat_id = cat_name_to_id[cls_name]
            else:
                cat_id = ann.get("class_id", 1)
                try:
                    cat_id = int(cat_id)
                except (ValueError, TypeError):
                    cat_id = 1

            raw_seg = ann.get("segmentation", [])
            norm_seg = normalize_segmentation(raw_seg, bbox)
            area = float(ann.get("area", w * h))

            all_annotations.append({
                "id": ann_counter,
                "image_id": img_idx,
                "category_id": cat_id,
                "bbox": bbox,
                "segmentation": norm_seg,
                "area": area,
                "iscrowd": 0,
                "scale": ann.get("scale", "micro"),
            })
            ann_counter += 1

    return {
        "info": {
            "description": "Multi-Image Histology Dataset",
            "version": "1.0",
            "year": 2026,
        },
        "images": all_images,
        "annotations": all_annotations,
        "categories": all_categories,
    }
