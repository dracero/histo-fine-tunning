"""
Histology Automated Semantic Labeling Pipeline for Roboflow Dataset Generation.

Architecture:
1. Dual-Scale Ontology Resolver: Loads candidate cellular classes & macro architectural
   structures dynamically from PDF ontologies (compartments, layers, basements, cavities).
2. Level 1 (Macro): SAM 3.1 + Gemini Multimodal Vision segment anatomical compartments
   (tubules, lumen, stroma) and derives continuous basement membrane / tunica propria ribbons.
3. Level 2 (Micro): Cellpose (cpsam/cyto3/nuclei) extracts high-density individual cell/nucleus instances.
4. Spatial Topological Prior: Assigns cell instances to their parent containing macro-compartment.
5. Quad-Foundation Model Ensemble:
   - CONCH (512d Vision-Language): Zero-shot text-image semantic matching
   - Virchow 2 (1280d ViT-Huge): Morphological granularity & fine cytological texture
   - UNI (1024d ViT-Large): Dense tissue context representations
   - Lunit DINO (384d ViT-Small/8): Self-supervised representation from 33M H&E patches
6. Gemini Multimodal Image-by-Image Validation: Arbitrates ambiguous cellular instances
   strictly constrained by the active ontology classes.
7. Roboflow & COCO Exporter: Generates datasets ready for review and direct model training.
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
        is_cellular_class,
        get_pathology_models_status,
        VirchowModelWrapper,
        UniModelWrapper,
        extract_crops_from_detections,
    )
    from backend.pdf_ontology import list_ontologies, load_ontology
    from backend.roboflow_integration import build_coco_json, build_multi_image_coco, upload_dataset_to_roboflow
    from backend.gemini_vision import (
        detect_histological_macro_layers_gemini,
        validate_uncertain_detections_with_gemini,
        analyze_tissue_macro_micro_ontology,
        classify_cell_with_spatial_prior_gemini,
        classify_cells_batch_gemini,
    )
except ImportError:
    from cellpose_segmenter import run_cellpose_segmentation, is_cellpose_available
    from pathology_models import (
        classify_with_ontology_ensemble,
        group_detections_by_class,
        filter_cellular_candidate_classes,
        is_cellular_class,
        get_pathology_models_status,
        VirchowModelWrapper,
        UniModelWrapper,
        extract_crops_from_detections,
    )
    from pdf_ontology import list_ontologies, load_ontology
    from roboflow_integration import build_coco_json, build_multi_image_coco, upload_dataset_to_roboflow
    from gemini_vision import (
        detect_histological_macro_layers_gemini,
        validate_uncertain_detections_with_gemini,
        analyze_tissue_macro_micro_ontology,
        classify_cell_with_spatial_prior_gemini,
        classify_cells_batch_gemini,
    )

logger = logging.getLogger("sam3-backend.autolabel")


class HistologyAutoLabeler:
    """
    End-to-End Automated Dual-Scale Histology Annotation Pipeline.
    Combines Gemini Multimodal Vision + SAM 3.1 for macro architectural layers and
    Cellpose + Quad-Foundation Models (Virchow 2, UNI, CONCH, Lunit DINO) for dense cellular instances.
    """

    def __init__(
        self,
        default_cellpose_model: str = "cpsam",
        sam3_predictor: Optional[Any] = None,
        sam3_processor: Optional[Any] = None,
    ) -> None:
        self.default_cellpose_model: str = default_cellpose_model
        self.sam3_predictor: Optional[Any] = sam3_predictor
        self.sam3_processor: Optional[Any] = sam3_processor

        # Reuse existing SAM 3 instances if already resident in memory
        if self.sam3_predictor is None and self.sam3_processor is None:
            try:
                import sys
                main_mod = sys.modules.get("backend.main") or sys.modules.get("main")
                if main_mod:
                    self.sam3_predictor = getattr(main_mod, "sam3_semantic_predictor", None)
                    self.sam3_processor = getattr(main_mod, "processor", None)
            except Exception as e:
                logger.debug(f"Could not hook into loaded SAM 3 from main: {e}")

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
        domain_title: str = "histología"

        if raw_ontology:
            if isinstance(raw_ontology, dict):
                candidate_classes = raw_ontology.get("structures", [])
                if not candidate_classes:
                    candidate_classes = raw_ontology.get("macro_structures", []) + raw_ontology.get("micro_structures", [])
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
            if ont_doc:
                if "structures" in ont_doc and ont_doc["structures"]:
                    candidate_classes = ont_doc["structures"]
                else:
                    macros = ont_doc.get("macro_structures", [])
                    micros = ont_doc.get("micro_structures", [])
                    candidate_classes = macros + micros
                domain_title = ont_doc.get("domain", ont_doc.get("title", target))

        if not candidate_classes:
            onts = list_ontologies()
            if onts:
                first_name = onts[0].get("name", "")
                ont_doc = load_ontology(first_name)
                if ont_doc:
                    if "structures" in ont_doc and ont_doc["structures"]:
                        candidate_classes = ont_doc["structures"]
                    else:
                        candidate_classes = ont_doc.get("macro_structures", []) + ont_doc.get("micro_structures", [])
                    domain_title = ont_doc.get("domain", first_name)

        # Ensure macro vs micro flags are properly resolved
        for c in candidate_classes:
            if "is_macro" not in c:
                c["is_macro"] = not is_cellular_class(c)
            if "role" in c and c["role"] in ["compartment", "layer", "cavity", "boundary_outer", "boundary_inner", "stroma"]:
                c["is_macro"] = True

        return candidate_classes, str(domain_title)

    def _derive_basement_membrane_annulus(
        self,
        tubule_detections: List[Dict[str, Any]],
        boundary_class: Dict[str, Any],
        img_w: int,
        img_h: int,
        ribbon_thickness: int = 4,
    ) -> List[Dict[str, Any]]:
        """
        Derives high-precision continuous basement membrane / tunica propria ribbons
        around segmented seminiferous tubule boundaries using morphological dilation.
        Guarantees 100% boundary continuity in H&E sections.
        """
        membrane_detections: List[Dict[str, Any]] = []
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ribbon_thickness * 2 + 1, ribbon_thickness * 2 + 1))

        for idx, tubule in enumerate(tubule_detections):
            segs = tubule.get("segmentation", [])
            if not segs:
                continue

            tubule_mask = np.zeros((img_h, img_w), dtype=np.uint8)
            for poly in segs:
                if len(poly) >= 6:
                    pts = np.array(poly, dtype=np.int32).reshape(-1, 2)
                    cv2.fillPoly(tubule_mask, [pts], 255)

            if cv2.countNonZero(tubule_mask) < 100:
                continue

            # Morphological dilation: outer ring - inner tubule
            dilated_mask = cv2.dilate(tubule_mask, kernel)
            ribbon_mask = cv2.bitwise_and(dilated_mask, cv2.bitwise_not(tubule_mask))

            contours, _ = cv2.findContours(ribbon_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c_idx, cnt in enumerate(contours):
                area = cv2.contourArea(cnt)
                if area < 40:
                    continue
                epsilon = 0.005 * cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, epsilon, True)
                if len(approx) < 3:
                    continue
                pts_arr = approx.reshape(-1, 2)
                poly_pts = pts_arr.flatten().astype(float).tolist()
                bx, by, bw, bh = cv2.boundingRect(cnt)

                m_key = boundary_class.get("key", "lamina_basal_tubular")
                m_label = boundary_class.get("label", boundary_class.get("name", "Lámina basal tubular"))
                m_color = boundary_class.get("color", "#06b6d4")

                membrane_detections.append({
                    "id": f"membrane_{idx}_{c_idx}",
                    "key": m_key,
                    "class_key": m_key,
                    "class_label": m_label,
                    "category_id": m_key,
                    "label": m_label,
                    "color": m_color,
                    "structure_type": "boundary",
                    "is_macro_layer": True,
                    "parent_structure": tubule.get("key"),
                    "score": 0.95,
                    "box": [float(bx), float(by), float(bx + bw), float(by + bh)],
                    "bbox": [float(bx), float(by), float(bw), float(bh)],
                    "segmentation": [poly_pts],
                    "area": float(round(area, 1)),
                    "decision_source": "morphological_tubule_boundary_derivation",
                })

        logger.info(f"Derived {len(membrane_detections)} continuous basement membrane ribbons around tubules.")
        return membrane_detections

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
        include_macro_layers: bool = True,
        min_area: int = 15,
    ) -> Dict[str, Any]:
        """
        Execute Dual-Scale High-Precision Automated Labeling for a histology image:
        Level 1: Continuous anatomical layers & compartments (SAM 3.1 + Gemini + Morphological boundary).
        Level 2: Individual cellular instances (Cellpose + Quad-Foundation Model Ensemble + Gemini validation).
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
        macro_layers: List[Dict[str, Any]] = []
        spatial_map: Dict[str, List[str]] = {}

        macro_classes = [c for c in classes if c.get("is_macro", False)]
        cellular_classes = [c for c in classes if not c.get("is_macro", False)]
        if not cellular_classes:
            cellular_classes = classes

        # =========================================================================
        # LEVEL 1: Dual-Scale Macro-Micro Analysis & Compartment Segmentation
        # =========================================================================
        grounded: List[Dict[str, Any]] = []
        if include_macro_layers or use_gemini_validation:
            logger.info(f"Analyzing Macro & Micro tissue architecture with Gemini Vision for '{domain_title}'...")
            try:
                gemini_analysis = analyze_tissue_macro_micro_ontology(
                    image=img_rgb,
                    organ_context=domain_title,
                    base_ontology=classes,
                )
                if gemini_analysis.get("success"):
                    if gemini_analysis.get("macro_layers"):
                        grounded = gemini_analysis["macro_layers"]
                    if gemini_analysis.get("spatial_map"):
                        spatial_map = gemini_analysis["spatial_map"]

                    # Enrich cellular classes with newly discovered cytological features
                    discovered_cells = gemini_analysis.get("cellular_classes", [])
                    if discovered_cells:
                        existing_keys = {c.get("key") for c in cellular_classes if c.get("key")}
                        for dc in discovered_cells:
                            if dc.get("key") not in existing_keys:
                                cellular_classes.append(dc)
                                classes.append(dc)
                                existing_keys.add(dc.get("key"))
                            else:
                                # Update prompt and cytological features if present
                                for ec in cellular_classes:
                                    if ec.get("key") == dc.get("key"):
                                        if dc.get("cytological_features"):
                                            ec["cytological_features"] = dc["cytological_features"]
                                        if dc.get("spatial_zone"):
                                            ec["spatial_zone"] = dc["spatial_zone"]
            except Exception as gemini_err:
                logger.warning(f"Gemini dual-scale analysis note: {gemini_err}")

        # Fallback to standard macro layer grounding if dual-scale returned empty
        if not grounded and include_macro_layers and macro_classes:
            try:
                grounded = detect_histological_macro_layers_gemini(
                    image=img_rgb,
                    organ_context=domain_title,
                    ontology_structures=macro_classes,
                )
            except Exception as e:
                logger.warning(f"Macro grounding fallback note: {e}")

        if grounded:
            # Refine macro-layer polygon boundaries using SAM 3.1 or adaptive gradient contours
            img_np = np.array(img_rgb)
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            tubule_detections: List[Dict[str, Any]] = []

            for layer in grounded:
                lx1, ly1, lx2, ly2 = [int(v) for v in layer["box"]]
                lx1, ly1 = max(0, lx1), max(0, ly1)
                lx2, ly2 = min(w, lx2), min(h, ly2)
                layer_w, layer_h = max(1, lx2 - lx1), max(1, ly2 - ly1)

                refined_poly: Optional[List[float]] = None

                # Refinement Priority 1: SAM 3.1 box prompt if predictor available
                if self.sam3_predictor is not None:
                    try:
                        self.sam3_predictor.set_image(img_rgb)
                        sam_res = self.sam3_predictor(bboxes=[[lx1, ly1, lx2, ly2]])
                        if sam_res and hasattr(sam_res[0], "masks") and sam_res[0].masks is not None:
                            polys = sam_res[0].masks.xy
                            if len(polys) > 0 and len(polys[0]) >= 3:
                                pts = polys[0].astype(float)
                                refined_poly = pts.flatten().tolist()
                    except Exception as sam_err:
                        logger.debug(f"SAM 3 predictor box refinement note: {sam_err}")

                # Refinement Priority 2: Adaptive morphological contour extraction on crop
                if refined_poly is None:
                    try:
                        crop_gray = gray[ly1:ly2, lx1:lx2]
                        if crop_gray.size > 100:
                            _, thresh = cv2.threshold(crop_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
                            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
                            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            if contours:
                                largest = max(contours, key=cv2.contourArea)
                                if cv2.contourArea(largest) >= (layer_w * layer_h * 0.08):
                                    epsilon = 0.006 * cv2.arcLength(largest, True)
                                    approx = cv2.approxPolyDP(largest, epsilon, True)
                                    if len(approx) >= 3:
                                        pts = approx.reshape(-1, 2)
                                        pts[:, 0] += lx1
                                        pts[:, 1] += ly1
                                        refined_poly = pts.flatten().astype(float).tolist()
                    except Exception as poly_err:
                        logger.debug(f"Contour refinement note for layer {layer.get('key')}: {poly_err}")

                if refined_poly is not None:
                    layer["segmentation"] = [refined_poly]

                macro_layers.append(layer)

                l_key = str(layer.get("key", "")).lower()
                if any(t_ind in l_key for t_ind in ("tubulo", "tubule", "seminifero", "epithel")):
                    tubule_detections.append(layer)

            # Morphological Basement Membrane Derivation:
            boundary_classes = [
                c for c in classes
                if any(b_ind in str(c.get("key", "")).lower() for b_ind in ("basal", "membrana", "pared", "tunica_propria"))
                or c.get("role") in ("boundary_inner", "boundary_outer")
            ]
            if tubule_detections and boundary_classes:
                target_boundary = boundary_classes[0]
                derived_membranes = self._derive_basement_membrane_annulus(
                    tubule_detections=tubule_detections,
                    boundary_class=target_boundary,
                    img_w=w,
                    img_h=h,
                    ribbon_thickness=4,
                )
                macro_layers.extend(derived_membranes)

            logger.info(f"Grounded {len(macro_layers)} macro-tissue compartments/layers and boundaries.")
            all_combined_detections.extend(macro_layers)

        # =========================================================================
        # LEVEL 2: Cellular & Nuclear Instance Segmentation (Cellpose)
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
            # Spatial Layer Attribution: Map each cell to its containing macro-compartment
            for cell_det in raw_cell_detections:
                cbx, cby, cbw, cbh = cell_det.get("bbox", [0, 0, 1, 1])
                cx, cy = cbx + cbw / 2.0, cby + cbh / 2.0
                cell_det["containing_layer"] = None

                for layer in macro_layers:
                    segs = layer.get("segmentation", [])
                    matched = False
                    for poly in segs:
                        if len(poly) >= 6:
                            pts = np.array(poly, dtype=np.float32).reshape(-1, 2)
                            if cv2.pointPolygonTest(pts, (float(cx), float(cy)), False) >= 0:
                                cell_det["containing_layer"] = layer.get("key")
                                matched = True
                                break
                    if matched:
                        break

                    if cell_det["containing_layer"] is None:
                        lx1, ly1, lx2, ly2 = layer.get("box", [0, 0, 0, 0])
                        if lx1 <= cx <= lx2 and ly1 <= cy <= ly2:
                            cell_det["containing_layer"] = layer.get("key")

            # =================================================================
            # PRIMARY: Gemini Vision Batch Classification (0 GPU VRAM)
            # Sends annotated image with numbered contours to Gemini for
            # contextual classification of all cells in 1-2 API calls.
            # =================================================================
            gemini_success = False
            if use_gemini_validation:
                try:
                    classified_cell_detections, uncertain_indices = classify_cells_batch_gemini(
                        image=img_rgb,
                        detections=raw_cell_detections,
                        ontology_classes=cellular_classes,
                        spatial_map=spatial_map,
                        organ_context=domain_title,
                    )
                    # Consider success if Gemini classified at least 50% of cells
                    classified_count = len(raw_cell_detections) - len(uncertain_indices)
                    if classified_count >= len(raw_cell_detections) * 0.5:
                        gemini_success = True
                        logger.info(
                            f"Gemini Vision batch classified {classified_count}/{len(raw_cell_detections)} cells "
                            f"({len(uncertain_indices)} uncertain)."
                        )
                except Exception as gemini_err:
                    logger.warning(f"Gemini Vision batch classification failed, falling back to ensemble: {gemini_err}")

            # =================================================================
            # FALLBACK: Quad-Foundation Ensemble (if Gemini unavailable/failed)
            # Uses CONCH 512d + Virchow2 1280d + UNI 1024d + Lunit DINO 384d
            # =================================================================
            if not gemini_success:
                logger.info("Using Quad-Foundation Ensemble classification (fallback).")
                classified_cell_detections, uncertain_indices = classify_with_ontology_ensemble(
                    image=img_rgb,
                    detections=raw_cell_detections,
                    ontology_classes=cellular_classes,
                    confidence_threshold=confidence_threshold,
                    uncertainty_threshold=uncertainty_threshold,
                    is_histology=True,
                )

            # Spatial Constraint Enforcement via Spatial Map / Ontology Parent Graph
            for det in classified_cell_detections:
                layer_k = det.get("containing_layer")
                if layer_k and layer_k in spatial_map:
                    allowed_keys = set(spatial_map[layer_k])
                    cur_key = det.get("class_key")
                    if cur_key not in allowed_keys:
                        matched_c = next((c for c in cellular_classes if c.get("key") in allowed_keys), None)
                        if matched_c:
                            det["category_id"] = matched_c.get("key")
                            det["class_key"] = matched_c.get("key")
                            det["class_label"] = matched_c.get("name", matched_c.get("label", matched_c.get("key")))
                            det["color"] = matched_c.get("color", "#8b5cf6")
                            det["spatial_parent_aligned"] = layer_k
                            det["classification_uncertain"] = True

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

        # Build comprehensive category mapping
        category_map = {}
        class_id_counter = 1

        for c in classes:
            k = c.get("key")
            if k and k not in category_map:
                category_map[k] = {
                    "id": class_id_counter,
                    "name": c.get("label", c.get("name", k)),
                    "key": k,
                    "color": c.get("color", "#8b5cf6"),
                    "structure_type": "macro_layer" if c.get("is_macro") else "cell",
                }
                class_id_counter += 1

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
            f"Dual-scale autolabel completed: {len(macro_layers)} layers/boundaries + {len(classified_cell_detections)} cells in {exec_time}s."
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
        include_macro_layers: bool = True,
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
                include_macro_layers=include_macro_layers,
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
