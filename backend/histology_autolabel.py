"""
Histology Automated Semantic Labeling Pipeline for Roboflow Dataset Generation.

Architecture:
1. Ontology Resolver: Loads candidate cellular classes & visual metadata dynamically from PDF ontologies.
2. Cellpose-SAM Instance Segmenter: Extracts high-precision polygon boundaries for cells and nuclei.
3. Quad-Foundation Model Ensemble:
   - CONCH (512d Vision-Language): Zero-shot text-image semantic matching
   - Virchow 2 (1280d ViT-Huge): Morphological granularity & fine texture
   - UNI (1024d ViT-Large): Dense tissue context representations
   - Lunit DINO (384d ViT-Small/8): Self-supervised representation from 33M H&E patches
4. Gemini Vision Adjudicator: Resolves ambiguous/low-confidence instances with multimodal visual review.
5. Review & Roboflow Exporter: Generates COCO datasets ready for manual UI review and direct Roboflow training.
"""

import io
import json
import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import cv2
from PIL import Image

try:
    from backend.cellpose_segmenter import run_cellpose_segmentation, is_cellpose_available
    from backend.pathology_models import (
        classify_with_ontology_ensemble,
        group_detections_by_class,
        filter_cellular_candidate_classes,
        get_pathology_models_status,
    )
    from backend.pdf_ontology import list_ontologies, load_ontology
    from backend.gemini_vision import validate_uncertain_detections_with_gemini
    from backend.roboflow_integration import build_coco_json, build_multi_image_coco, upload_dataset_to_roboflow
except ImportError:
    from cellpose_segmenter import run_cellpose_segmentation, is_cellpose_available
    from pathology_models import (
        classify_with_ontology_ensemble,
        group_detections_by_class,
        filter_cellular_candidate_classes,
        get_pathology_models_status,
    )
    from pdf_ontology import list_ontologies, load_ontology
    from gemini_vision import validate_uncertain_detections_with_gemini
    from roboflow_integration import build_coco_json, build_multi_image_coco, upload_dataset_to_roboflow

logger = logging.getLogger("sam3-backend.autolabel")


class HistologyAutoLabeler:
    """
    End-to-End Automated Histology Annotation Pipeline for Testicle & Artery (and any tissue ontology).
    Optimized for high-precision cell segmentation and robust foundation model classification.
    """

    def __init__(self, default_cellpose_model: str = "cpsam") -> None:
        self.default_cellpose_model = default_cellpose_model

    def resolve_ontology_classes(
        self,
        ontology_name: Optional[str] = None,
        raw_ontology: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
    ) -> Tuple[List[Dict[str, Any]], str]:
        """
        Dynamically extract candidate classes and domain context from active ontology.
        Strictly reads from ontology documents (PDF-extracted or uploaded JSON). Zero hardcoded rules.
        """
        candidate_classes: List[Dict[str, Any]] = []
        domain_title = "histología"

        if raw_ontology:
            if isinstance(raw_ontology, dict):
                candidate_classes = raw_ontology.get("structures", [])
                domain_title = raw_ontology.get("domain", raw_ontology.get("title", domain_title))
            elif isinstance(raw_ontology, list):
                candidate_classes = raw_ontology

        elif ontology_name and ontology_name.strip():
            target = ontology_name.strip()
            ont_doc = load_ontology(target)
            if not ont_doc:
                # Case-insensitive / substring matching in available ontologies
                all_onts = list_ontologies()
                for o in all_onts:
                    o_name = str(o.get("name", "")).lower()
                    if target.lower() in o_name or o_name.startswith(target.lower()):
                        ont_doc = load_ontology(o.get("name", ""))
                        break
            if ont_doc and "structures" in ont_doc:
                candidate_classes = ont_doc["structures"]
                domain_title = ont_doc.get("domain", ont_doc.get("title", target))

        if not candidate_classes:
            onts = list_ontologies()
            if onts:
                first_name = onts[0].get("name", "")
                ont_doc = load_ontology(first_name)
                if ont_doc and "structures" in ont_doc:
                    candidate_classes = ont_doc["structures"]
                    domain_title = ont_doc.get("domain", first_name)

        return candidate_classes, str(domain_title)



    def autolabel_single_image(
        self,
        image: Image.Image,
        image_filename: str = "image.png",
        ontology_name: Optional[str] = None,
        raw_ontology: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
        cellpose_model: Optional[str] = None,
        cell_diameter: Optional[float] = None,
        confidence_threshold: float = 0.50,
        uncertainty_threshold: float = 0.30,
        use_gemini_validation: bool = True,
        min_area: int = 15,
        include_macro_layers: bool = True,
    ) -> Dict[str, Any]:
        """
        Execute Dual-Scale High-Precision Automated Labeling for a histology image:
        Level 1: Continuous anatomical layers & compartments (Tunica Media, Adventitia, Lumen, Tubules).
        Level 2: Individual cellular instances (Endothelium, Smooth muscle, Spermatogonia, Leydig, etc.).
        """
        start_time = time.time()
        img_rgb = image.convert("RGB")
        w, h = img_rgb.size

        # 1. Resolve ontology classes
        classes, domain_title = self.resolve_ontology_classes(
            ontology_name=ontology_name, raw_ontology=raw_ontology
        )
        logger.info(
            f"Dual-Scale autolabeling '{image_filename}' ({w}x{h}) using {len(classes)} classes from '{domain_title}'."
        )

        all_combined_detections: List[Dict[str, Any]] = []

        # =========================================================================
        # LEVEL 1: Macro-Compartment & Tissue Layer Segmentation
        # =========================================================================
        macro_layers: List[Dict[str, Any]] = []
        if include_macro_layers:
            try:
                from backend.gemini_vision import detect_histological_macro_layers_gemini
            except ImportError:
                from gemini_vision import detect_histological_macro_layers_gemini

            grounded = detect_histological_macro_layers_gemini(
                image=img_rgb,
                organ_context=domain_title,
                ontology_structures=classes,
            )

            # Refine macro-layer polygon boundaries using image gradients / contours
            img_np = np.array(img_rgb)
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

            for layer in grounded:
                lx1, ly1, lx2, ly2 = [int(v) for v in layer["box"]]
                lx1, ly1 = max(0, lx1), max(0, ly1)
                lx2, ly2 = min(w, lx2), min(h, ly2)
                layer_w, layer_h = max(1, lx2 - lx1), max(1, ly2 - ly1)

                # Attempt organic contour extraction within the grounded box
                try:
                    crop_gray = gray[ly1:ly2, lx1:lx2]
                    if crop_gray.size > 100:
                        # Otsu thresholding on layer crop
                        _, thresh = cv2.threshold(crop_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                        # Morphological smoothing
                        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
                        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
                        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        if contours:
                            largest = max(contours, key=cv2.contourArea)
                            if cv2.contourArea(largest) >= (layer_w * layer_h * 0.10):
                                epsilon = 0.008 * cv2.arcLength(largest, True)
                                approx = cv2.approxPolyDP(largest, epsilon, True)
                                if len(approx) >= 3:
                                    pts = approx.reshape(-1, 2)
                                    # Shift back to full image coordinates
                                    pts[:, 0] += lx1
                                    pts[:, 1] += ly1
                                    layer["segmentation"] = [pts.flatten().astype(float).tolist()]
                except Exception as poly_err:
                    logger.debug(f"Contour refinement note for layer {layer.get('key')}: {poly_err}")

                macro_layers.append(layer)

            logger.info(f"Grounded {len(macro_layers)} macro-tissue compartments/layers.")
            all_combined_detections.extend(macro_layers)

        # =========================================================================
        # LEVEL 2: Cellular & Nuclear Instance Segmentation
        # =========================================================================
        cp_model = cellpose_model or self.default_cellpose_model
        seg_res = run_cellpose_segmentation(
            image_input=img_rgb,
            model_type=cp_model,
            diameter=cell_diameter,
            min_area=min_area,
        )
        raw_cell_detections = seg_res.get("detections", [])
        total_cells_segmented = len(raw_cell_detections)
        logger.info(f"Segmented {total_cells_segmented} cell/nucleus instances with Cellpose ({cp_model}).")

        classified_cell_detections: List[Dict[str, Any]] = []
        if raw_cell_detections:
            # Spatial Layer Attribution Prior
            # Assign cells to their containing anatomical layer as a strong biological prior
            for cell_det in raw_cell_detections:
                cbx, cby, cbw, cbh = cell_det.get("bbox", [0, 0, 1, 1])
                cx, cy = cbx + cbw / 2.0, cby + cbh / 2.0
                cell_det["containing_layer"] = None

                for layer in macro_layers:
                    lx1, ly1, lx2, ly2 = layer["box"]
                    if lx1 <= cx <= lx2 and ly1 <= cy <= ly2:
                        cell_det["containing_layer"] = layer["key"]
                        break

            # Quad-Foundation Ensemble Classification for cellular instances
            classified_cell_detections, uncertain_indices = classify_with_ontology_ensemble(
                image=img_rgb,
                detections=raw_cell_detections,
                ontology_classes=classes,
                confidence_threshold=confidence_threshold,
                uncertainty_threshold=uncertainty_threshold,
                is_histology=True,
            )

            # Dynamic Spatial Layer Attribution based on Ontology Parent Graph (Zero Hardcoded Names)
            # If the ontology document specifies that certain cell types belong to a parent layer,
            # cells located inside that layer prioritize those valid child entities.
            parent_children_map: Dict[str, List[Dict[str, Any]]] = {}
            for c in classes:
                p_key = str(c.get("parent", "") or "").strip()
                if p_key and p_key not in ("none", "null", ""):
                    parent_children_map.setdefault(p_key, []).append(c)

            for det in classified_cell_detections:
                layer_k = det.get("containing_layer")
                if layer_k and layer_k in parent_children_map:
                    valid_children = parent_children_map[layer_k]
                    valid_child_keys = {c.get("key") for c in valid_children if c.get("key")}
                    cur_key = det.get("class_key")
                    # If current prediction does not belong to the containing parent layer,
                    # re-assign to the primary ontology-defined child of this compartment
                    if cur_key not in valid_child_keys and valid_children:
                        best_child = valid_children[0]
                        det["category_id"] = best_child.get("key")
                        det["class_key"] = best_child.get("key")
                        det["class_label"] = best_child.get("label", best_child.get("name", best_child.get("key")))
                        det["color"] = best_child.get("color", "#8b5cf6")
                        det["spatial_parent_aligned"] = layer_k

            # Gemini Vision Validation on Ambiguous Instances
            if use_gemini_validation and uncertain_indices:
                uncertain_subset = [classified_cell_detections[idx] for idx in uncertain_indices]
                validated_subset = validate_uncertain_detections_with_gemini(
                    image=img_rgb,
                    uncertain_detections=uncertain_subset,
                    ontology_classes=classes,
                    organ_context=domain_title,
                )
                for local_idx, orig_idx in enumerate(uncertain_indices):
                    if local_idx < len(validated_subset):
                        classified_cell_detections[orig_idx] = validated_subset[local_idx]

            all_combined_detections.extend(classified_cell_detections)

        # =========================================================================
        # COMBINED RESULTS & COCO EXPORT GENERATION
        # =========================================================================
        groups = group_detections_by_class(
            classified_detections=all_combined_detections,
            candidate_classes=classes,
        )

        scores = [float(d.get("score", 0.0)) for d in all_combined_detections]
        conf_stats = {
            "mean": round(float(np.mean(scores)), 4) if scores else 0.0,
            "min": round(float(np.min(scores)), 4) if scores else 0.0,
            "max": round(float(np.max(scores)), 4) if scores else 0.0,
            "high_confidence_count": sum(1 for s in scores if s >= 0.70),
            "macro_layers_count": len(macro_layers),
            "cells_count": len(classified_cell_detections),
            "uncertain_count": sum(1 for d in all_combined_detections if d.get("classification_uncertain", False)),
        }

        # Build comprehensive category mapping (both macro-layers and cellular subtypes)
        category_map = {}
        class_id_counter = 1

        # Add ontology classes
        for c in classes:
            k = c.get("key")
            if k and k not in category_map:
                category_map[k] = {
                    "id": class_id_counter,
                    "name": c.get("label", c.get("name", k)),
                    "key": k,
                    "color": c.get("color", "#8b5cf6"),
                    "structure_type": c.get("structure_type", "cell"),
                }
                class_id_counter += 1

        # Format detection annotations
        formatted_annotations = []
        for det in all_combined_detections:
            ck = det.get("class_key", "default_class")
            c_info = category_map.get(ck)
            if c_info is None:
                c_info = {
                    "id": class_id_counter,
                    "name": det.get("class_label", ck),
                    "key": ck,
                    "color": det.get("color", "#8b5cf6"),
                    "structure_type": det.get("structure_type", "cell"),
                }
                category_map[ck] = c_info
                class_id_counter += 1

            det["class_id"] = c_info["id"]
            formatted_annotations.append(det)

        coco_payload = {
            "image_filename": image_filename,
            "width": w,
            "height": h,
            "annotations": formatted_annotations,
            "classes": list(category_map.values()),
        }

        exec_time = round(time.time() - start_time, 2)
        logger.info(
            f"Dual-scale autolabel completed: {len(macro_layers)} layers + {len(classified_cell_detections)} cells in {exec_time}s."
        )

        return {
            "success": True,
            "image_filename": image_filename,
            "width": w,
            "height": h,
            "total_detections": len(all_combined_detections),
            "macro_layers_count": len(macro_layers),
            "cells_count": len(classified_cell_detections),
            "groups": groups,
            "detections": all_combined_detections,
            "confidence_stats": conf_stats,
            "classes_used": classes,
            "domain_context": domain_title,
            "coco_payload": coco_payload,
            "execution_time_seconds": exec_time,
        }

    def autolabel_batch(
        self,
        images: List[Tuple[str, Image.Image]],
        ontology_name: Optional[str] = None,
        raw_ontology: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
        cellpose_model: Optional[str] = None,
        cell_diameter: Optional[float] = None,
        confidence_threshold: float = 0.50,
        use_gemini_validation: bool = True,
    ) -> Dict[str, Any]:
        """
        Process a batch of images and generate a unified multi-image dataset.
        """
        start_time = time.time()
        per_image_results = []
        multi_image_coco_list = []
        total_detections_all = 0

        for filename, img in images:
            logger.info(f"Batch autolabeling image: {filename}...")
            res = self.autolabel_single_image(
                image=img,
                image_filename=filename,
                ontology_name=ontology_name,
                raw_ontology=raw_ontology,
                cellpose_model=cellpose_model,
                cell_diameter=cell_diameter,
                confidence_threshold=confidence_threshold,
                use_gemini_validation=use_gemini_validation,
            )
            per_image_results.append(res)
            if res.get("coco_payload"):
                multi_image_coco_list.append(res["coco_payload"])
            total_detections_all += res.get("total_detections", 0)

        # Generate combined COCO JSON
        combined_coco = build_multi_image_coco(multi_image_coco_list) if multi_image_coco_list else {}

        return {
            "success": True,
            "total_images": len(images),
            "total_detections": total_detections_all,
            "per_image_results": per_image_results,
            "combined_coco": combined_coco,
            "execution_time_seconds": round(time.time() - start_time, 2),
        }

    def export_to_roboflow(
        self,
        labeled_results: Union[Dict[str, Any], List[Dict[str, Any]]],
        image_files: Dict[str, bytes],
    ) -> Dict[str, Any]:
        """
        Upload the autolabeled images & COCO annotations directly to Roboflow.
        """
        if isinstance(labeled_results, dict):
            if "per_image_results" in labeled_results:
                images_list = [
                    r["coco_payload"] for r in labeled_results["per_image_results"] if "coco_payload" in r
                ]
            elif "coco_payload" in labeled_results:
                images_list = [labeled_results["coco_payload"]]
            elif "images" in labeled_results:
                images_list = labeled_results["images"]
            else:
                images_list = [labeled_results]
        else:
            images_list = labeled_results

        return upload_dataset_to_roboflow(images_data=images_list, image_files=image_files)
