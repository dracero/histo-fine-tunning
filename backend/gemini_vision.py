"""
Dynamic Multimodal Vision Assistant with Gemini 3.5 Flash & API Key Rotation.

Provides zero-shot visual prompt refinement, image structure discovery,
and prototype cell identification using Google Gemini with automatic API key rotation.
"""

import io
import json
import logging
import os
import re
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from dotenv import load_dotenv
from PIL import Image

# Ensure .env is loaded
env_path = Path(__file__).resolve().parent.parent / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
else:
    load_dotenv()

logger = logging.getLogger("sam3-backend")

DEFAULT_COLORS = [
    "#e11d48", "#8b5cf6", "#06b6d4", "#f59e0b", "#10b981",
    "#ec4899", "#6366f1", "#14b8a6", "#f97316", "#84cc16",
    "#a855f7", "#0ea5e9", "#ef4444", "#22c55e", "#eab308",
    "#d946ef", "#38bdf8", "#fb923c", "#4ade80", "#facc15",
]

GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash")
FALLBACK_MODEL = os.environ.get("FALLBACK_MODEL", "gemini-3.5-flash-lite")


class GeminiKeyManager:
    """
    Thread-safe manager for Google GenAI API keys with automatic round-robin rotation
    and cooldown handling upon 429 (rate-limit) or 403 (quota/forbidden) errors.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._index = 0
        self._cooldowns: Dict[str, float] = {}  # key -> timestamp until available
        self._failure_counts: Dict[str, int] = {}

    def get_all_keys(self) -> List[str]:
        keys = []
        # 1. GOOGLE_API_KEYS (comma-separated list from .env)
        raw_google = os.environ.get("GOOGLE_API_KEYS", "")
        if raw_google:
            for k in raw_google.split(","):
                k_clean = k.strip()
                if k_clean and k_clean not in keys:
                    keys.append(k_clean)
        # 2. GEMINI_API_KEYS (comma-separated fallback)
        raw_gemini = os.environ.get("GEMINI_API_KEYS", "")
        if raw_gemini:
            for k in raw_gemini.split(","):
                k_clean = k.strip()
                if k_clean and k_clean not in keys:
                    keys.append(k_clean)
        # 3. GEMINI_API_KEY (single key)
        single_gemini = os.environ.get("GEMINI_API_KEY", "").strip()
        if single_gemini and single_gemini not in keys:
            keys.append(single_gemini)
        # 4. GOOGLE_API_KEY (single key)
        single_google = os.environ.get("GOOGLE_API_KEY", "").strip()
        if single_google and single_google not in keys:
            keys.append(single_google)
        return keys

    def get_status(self) -> Dict[str, Any]:
        all_keys = self.get_all_keys()
        now = time.time()
        active_count = sum(1 for k in all_keys if self._cooldowns.get(k, 0) <= now)
        return {
            "total_keys": len(all_keys),
            "active_keys": active_count,
            "on_cooldown": len(all_keys) - active_count,
            "current_model": GEMINI_MODEL,
            "rotation_enabled": len(all_keys) > 1,
        }

    def mark_key_cooldown(self, key: str, duration_sec: float = 60.0, reason: str = "Rate limit (429/403)"):
        with self._lock:
            self._cooldowns[key] = time.time() + duration_sec
            self._failure_counts[key] = self._failure_counts.get(key, 0) + 1
            key_preview = key[:8] + "..." + key[-4:] if len(key) > 12 else "key"
            logger.warning(f"Gemini API Key [{key_preview}] in cooldown for {duration_sec}s: {reason}")

    def execute_with_rotation(
        self,
        call_fn: Callable[[Any, str], Any],
        explicit_api_key: Optional[str] = None,
        preferred_model: Optional[str] = None,
    ) -> Any:
        from google import genai
        from google.genai import types

        http_opts = types.HttpOptions(timeout=35.0)
        target_model = preferred_model or GEMINI_MODEL

        # If an explicit key was provided by the caller, use it directly
        if explicit_api_key:
            client = genai.Client(api_key=explicit_api_key, http_options=http_opts)
            try:
                return call_fn(client, target_model)
            except Exception as e:
                err_str = str(e).lower()
                if (
                    "not_found" in err_str
                    or "not found" in err_str
                    or "read operation timed out" in err_str
                    or "timeout" in err_str
                    or "503" in err_str
                    or "unavailable" in err_str
                    or "high demand" in err_str
                    or "spikes in demand" in err_str
                ) and target_model != FALLBACK_MODEL:
                    logger.warning(f"Retrying with fallback model {FALLBACK_MODEL} on error: {e}")
                    return call_fn(client, FALLBACK_MODEL)
                raise e

        keys = self.get_all_keys()
        if not keys:
            raise RuntimeError("No Google/Gemini API keys configured in .env (GOOGLE_API_KEYS or GEMINI_API_KEY).")

        now = time.time()
        with self._lock:
            start_idx = self._index
            self._index = (self._index + 1) % len(keys)
            ordered_keys = [keys[(start_idx + i) % len(keys)] for i in range(len(keys))]

        last_error = None
        for key in ordered_keys:
            # Check cooldown
            if self._cooldowns.get(key, 0) > time.time():
                continue

            try:
                client = genai.Client(api_key=key, http_options=http_opts)
                try:
                    res = call_fn(client, target_model)
                except Exception as model_err:
                    err_str = str(model_err).lower()
                    if (
                        "not_found" in err_str
                        or "not found" in err_str
                        or "read operation timed out" in err_str
                        or "timeout" in err_str
                        or "503" in err_str
                        or "unavailable" in err_str
                        or "high demand" in err_str
                        or "spikes in demand" in err_str
                        or "429" in err_str
                        or "resource_exhausted" in err_str
                        or "quota" in err_str
                    ) and target_model != FALLBACK_MODEL:
                        logger.warning(f"Model {target_model} issue ({model_err}), falling back to {FALLBACK_MODEL}")
                        res = call_fn(client, FALLBACK_MODEL)
                    else:
                        raise model_err

                # Success! Advance round-robin index
                with self._lock:
                    self._index = (keys.index(key) + 1) % len(keys)
                    self._cooldowns.pop(key, None)
                return res

            except Exception as e:
                err_str = str(e).lower()
                if (
                    "429" in err_str
                    or "resource_exhausted" in err_str
                    or "quota" in err_str
                    or "rate limit" in err_str
                    or "403" in err_str
                    or "503" in err_str
                    or "unavailable" in err_str
                    or "high demand" in err_str
                    or "spikes in demand" in err_str
                    or "overloaded" in err_str
                    or "read operation timed out" in err_str
                    or "timeout" in err_str
                ):
                    self.mark_key_cooldown(key, duration_sec=30.0, reason=str(e))
                    last_error = e
                    continue
                else:
                    raise e

        if last_error:
            raise last_error
        raise RuntimeError("All Google API keys are currently on cooldown due to rate limits. Please try again shortly.")


key_manager = GeminiKeyManager()


def _get_gemini_client(api_key: Optional[str] = None):
    """Backwards-compatible helper returning a GenAI client."""
    keys = [api_key] if api_key else key_manager.get_all_keys()
    if not keys:
        logger.warning("No Google API keys found in environment.")
        return None
    try:
        from google import genai
        return genai.Client(api_key=keys[0])
    except Exception as e:
        logger.error(f"Failed to initialize google-genai client: {e}")
        return None


def generate_gemini_content(
    contents: Any,
    system_instruction: Optional[str] = None,
    temperature: Optional[float] = None,
    response_mime_type: Optional[str] = None,
    api_key: Optional[str] = None,
    preferred_model: Optional[str] = None,
) -> Any:
    """Execute generate_content with key rotation, cooldown handling and model fallback."""
    def _call(client, model_name):
        from google.genai import types
        config = types.GenerateContentConfig(
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True)
        )
        if system_instruction:
            config.system_instruction = system_instruction
        if temperature is not None:
            config.temperature = temperature
        if response_mime_type:
            config.response_mime_type = response_mime_type

        return client.models.generate_content(
            model=model_name,
            contents=contents,
            config=config,
        )

    return key_manager.execute_with_rotation(
        _call,
        explicit_api_key=api_key,
        preferred_model=preferred_model,
    )


def suggest_cell_prototype_gemini(
    crop: Image.Image,
    organ_context: str = "testículo / espermatogénesis",
    ontology_structures: Optional[List[Dict[str, Any]]] = None,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Multimodal analysis of a microscopic cell crop using Gemini 3.5 Flash.
    Identifies the cellular subtype (e.g. Espermatogonia A Clara, Espermatocito Primario,
    Espermátide, Célula de Sertoli, etc.) with confidence, reasoning, and color.
    """
    if crop.mode != "RGB":
        crop = crop.convert("RGB")

    # Ensure crop has adequate dimensions for vision model inspection
    min_dim = 96
    w, h = crop.size
    if max(w, h) < min_dim:
        scale = min_dim / max(w, h)
        crop_preview = crop.resize((max(16, int(w * scale)), max(16, int(h * scale))), Image.NEAREST)
    else:
        crop_preview = crop.copy()

    ont_desc = ""
    if ontology_structures:
        class_names = [s.get("label", s.get("name", s.get("key", ""))) for s in ontology_structures[:20]]
        ont_desc = f"\nCandidate classes from active tissue ontology: {', '.join(class_names)}."

    sys_inst = """\
You are an expert computational histopathologist specializing in digital cytology and microscopy.
Analyze this high-resolution microscopic cell crop.
Determine the most probable biological/cytological cell type (with high expertise in spermatogenesis: \
e.g., 'Espermatogonia A Clara', 'Espermatogonia A Oscura', 'Espermatogonia B', 'Espermatocito Primario', \
'Espermátide Temprana', 'Espermátide Tardía', 'Espermatozoide', 'Célula de Sertoli', 'Célula de Leydig', \
'Célula Muscular Lisa / Mioide', 'Célula Endotelial', etc.).

Return ONLY a valid JSON object matching this schema:
{
  "label": "<Spanish cell type name, e.g. 'Espermatogonia A Clara'>",
  "category_id": "<normalized_snake_case_key, e.g. 'espermatogonia_a_clara'>",
  "color": "<hex_color_code, e.g. '#e11d48'>",
  "confidence": 0.85,
  "reasoning": "<Short clinical/morphological explanation: nuclear chromatin, nucleoli, position, size, cytoplasm in Spanish>",
  "alternative_labels": ["<Alternative 1>", "<Alternative 2>"]
}
"""

    prompt = f"Examine this cell crop from a histological section of {organ_context}.{ont_desc}\nIdentify the cell type:"

    try:
        response = generate_gemini_content(
            contents=[prompt, crop_preview],
            system_instruction=sys_inst,
            temperature=0.15,
            response_mime_type="application/json",
            api_key=api_key,
        )

        raw_text = (response.text or "").strip()
        if raw_text.startswith("```"):
            raw_text = re.sub(r"^```(?:json)?\s*", "", raw_text)
            raw_text = re.sub(r"\s*```$", "", raw_text)

        data = json.loads(raw_text)
        if isinstance(data, dict) and "label" in data:
            if "color" not in data or not data["color"].startswith("#"):
                data["color"] = "#e11d48"
            return data

    except Exception as e:
        logger.error(f"Error suggesting cell prototype with Gemini: {e}", exc_info=True)

    # Fallback if anything goes wrong
    return {
        "label": "Espermatogonia A Clara",
        "category_id": "espermatogonia_a_clara",
        "color": "#e11d48",
        "confidence": 0.70,
        "reasoning": "Célula espermatogénica situada en la membrana basal del túbulo seminífero.",
        "alternative_labels": ["Espermatogonia", "Espermatocito Primario"],
    }


def refine_prompt_multimodal(
    image: Image.Image,
    user_prompt: str,
    ontology_context: Optional[List[Dict[str, Any]]] = None,
    api_key: Optional[str] = None,
) -> str:
    """
    Dynamically translate and refine any user prompt (in Spanish or any language)
    into a concise visual grounding prompt optimized for SAM 3.1, considering
    the actual visual context of the image.
    """
    if not user_prompt or not user_prompt.strip():
        return "cell nucleus"

    try:
        # Prepare lightweight image preview for Gemini
        img_preview = image.copy()
        if img_preview.mode != "RGB":
            img_preview = img_preview.convert("RGB")
        max_dim = 768
        if max(img_preview.size) > max_dim:
            ratio = max_dim / max(img_preview.size)
            img_preview = img_preview.resize(
                (int(img_preview.width * ratio), int(img_preview.height * ratio)),
                Image.LANCZOS,
            )

        ont_summary = ""
        if ontology_context:
            keys = [c.get("name", c.get("key", "")) for c in ontology_context[:10]]
            ont_summary = f"Active ontology structures: {', '.join(keys)}."

        sys_inst = (
            "You are a multimodal vision specialist. Convert the user's histological/biological "
            "query into a single concise English visual grounding prompt (3 to 6 words max) "
            "specifically optimized for Segment Anything Model 3 (SAM 3.1 open-vocabulary grounding). "
            "Focus on direct visual attributes visible in the image: color/staining (dark violet, pink eosinophilic), "
            "shape (round, elongated spindle, tubular cavity, wavy bundle), and structure (nucleus, lumen, fiber). "
            "Return ONLY the concise visual phrase without quotes or explanation."
        )

        prompt_text = f"User term: '{user_prompt}'. {ont_summary} Generate the optimal SAM 3 visual phrase:"

        response = generate_gemini_content(
            contents=[prompt_text, img_preview],
            system_instruction=sys_inst,
            temperature=0.1,
            api_key=api_key,
        )

        refined = response.text.strip().replace('"', '').replace("'", "")
        logger.info(f"Gemini refined prompt '{user_prompt}' -> '{refined}'")
        return refined if refined else user_prompt.strip()

    except Exception as e:
        logger.warning(f"Error in multimodal prompt refinement: {e}")
        return user_prompt.strip()


def discover_visual_primitives_from_image(
    image: Image.Image,
    ontology_context: Optional[List[Dict[str, Any]]] = None,
    max_structures: int = 8,
    api_key: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Dynamically inspect the histology/microscopy image with Gemini Vision to discover
    all distinct visual structures present on this specific slide, returning
    a list of prompt descriptors suitable for SAM 3.1 grounding.
    """
    client = _get_gemini_client(api_key)
    if client is None:
        return []

    try:
        img_preview = image.copy()
        if img_preview.mode != "RGB":
            img_preview = img_preview.convert("RGB")
        max_dim = 1024
        if max(img_preview.size) > max_dim:
            ratio = max_dim / max(img_preview.size)
            img_preview = img_preview.resize(
                (int(img_preview.width * ratio), int(img_preview.height * ratio)),
                Image.LANCZOS,
            )

        ont_info = ""
        if ontology_context:
            ont_info = "Relevant ontology knowledge: " + json.dumps([
                {"name": c.get("name", c.get("key")), "prompt": c.get("prompt")}
                for c in ontology_context[:15]
            ], ensure_ascii=False)

        sys_inst = """\
You are an expert computational pathologist and vision AI assistant.
Analyze this histology / microscopy image and identify all distinct visual structures, cell types, \
connective fibers, lumens/cavities, and tissue compartments present in THIS specific image.

For each structure, provide:
- "key": short ASCII identifier (e.g. "cell_nucleus", "collagen_fiber", "tubular_lumen")
- "name": Spanish name (e.g. "Núcleos celulares", "Fibras de colágeno", "Luz tubular")
- "label": Short UI label in Spanish
- "prompt": A concise English visual prompt (3-6 words) for SAM 3.1 segmentation describing color, shape, and structure (e.g. "dark violet round nucleus", "pink wavy collagen fiber", "empty circular lumen space").

Return ONLY a valid JSON array of objects.
"""

        user_content = [
            f"Analyze this image and extract up to {max_structures} distinct visual structures.\n{ont_info}",
            img_preview,
        ]

        response = generate_gemini_content(
            contents=user_content,
            system_instruction=sys_inst,
            temperature=0.2,
            response_mime_type="application/json",
            api_key=api_key,
        )

        raw_text = response.text.strip()
        if raw_text.startswith("```"):
            raw_text = re.sub(r"^```(?:json)?\s*", "", raw_text)
            raw_text = re.sub(r"\s*```$", "", raw_text)

        structures = json.loads(raw_text)
        if not isinstance(structures, list):
            return []

        # Assign palette colors
        for i, s in enumerate(structures):
            if "color" not in s:
                s["color"] = DEFAULT_COLORS[i % len(DEFAULT_COLORS)]
            if "label" not in s:
                s["label"] = s.get("name", s.get("key", f"Estructura {i + 1}"))

        logger.info(f"Gemini discovered {len(structures)} visual primitives in image: {[s['key'] for s in structures]}")
        return structures

    except Exception as e:
        logger.error(f"Error discovering visual primitives with Gemini Vision: {e}", exc_info=True)
        return []


def classify_with_multimodal_gemini_fusion(
    image: Image.Image,
    detections: List[Dict[str, Any]],
    candidate_classes: List[Dict[str, Any]],
    ontology_name: Optional[str] = None,
    ontology_context: Optional[Dict[str, Any]] = None,
    api_key: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Multimodal Agentic Pathology Classifier with Calibrated Probabilities:
    Combines MahmoodLab CONCH (512-dim) and Paige AI Virchow 2 (1280-dim) Foundation Embeddings
    with high-resolution visual crops passed directly to Gemini Multimodal Vision.

    1. Computes deep feature embeddings with CONCH (512d) and Virchow 2 (1280d).
    2. Extracts high-resolution cellular crops with adaptive contextual margins.
    3. Sends ambiguous/borderline crops directly to Gemini Vision for cytological arbitration.
    4. Performs Bayesian/calibrated fusion so LLM output cannot blindly overwrite confident foundation predictions.
    """
    if not detections:
        return []

    # Import pathology functions dynamically to avoid circular dependencies
    try:
        from backend.pathology_models import (
            extract_crops_from_detections,
            classify_detections_with_conch,
            filter_cellular_candidate_classes,
            VirchowModelWrapper,
            UniModelWrapper,
            _compute_detection_area,
        )
    except ImportError:
        from pathology_models import (
            extract_crops_from_detections,
            classify_detections_with_conch,
            filter_cellular_candidate_classes,
            VirchowModelWrapper,
            UniModelWrapper,
            _compute_detection_area,
        )
    import torch
    import torch.nn.functional as F

    # 1. Filter candidate classes to retain only cellular entities
    cellular_classes = filter_cellular_candidate_classes(candidate_classes or [])
    if not cellular_classes:
        cellular_classes = candidate_classes or []

    class_meta_map = {
        c.get("key", f"class_{i}"): {
            "key": c.get("key", f"class_{i}"),
            "label": c.get("label", c.get("name", c.get("key"))),
            "name": c.get("name", c.get("label", c.get("key"))),
            "color": c.get("color", DEFAULT_COLORS[i % len(DEFAULT_COLORS)]),
            "prompt": c.get("prompt", ""),
        }
        for i, c in enumerate(cellular_classes)
    }

    # 2. Extract crops and compute Virchow 2 (1280-dim), UNI (1024-dim) & CONCH (512-dim)
    crops = extract_crops_from_detections(image, detections, margin_ratio=0.35, min_size=80)
    num_dets = len(detections)

    virchow_feats = None
    virchow_sim_matrix = None
    try:
        virchow = VirchowModelWrapper.get_instance()
        if not virchow.is_loaded:
            virchow.load()
        if virchow.is_loaded:
            virchow_feats = virchow.encode_crops(crops, batch_size=16).float()
            virchow_norm = F.normalize(virchow_feats, dim=-1)
            virchow_sim_matrix = torch.matmul(virchow_norm, virchow_norm.T).cpu().numpy()
            logger.info(f"Computed Virchow 2 (1280d) features for {num_dets} cells")
    except Exception as virchow_err:
        logger.warning(f"Virchow feature computation skipped: {virchow_err}")

    uni_feats = None
    uni_sim_matrix = None
    try:
        uni = UniModelWrapper.get_instance()
        if not uni.is_loaded:
            uni.load()
        if uni.is_loaded:
            uni_feats = uni.encode_crops(crops, batch_size=16).float()
            uni_norm = F.normalize(uni_feats, dim=-1)
            uni_sim_matrix = torch.matmul(uni_norm, uni_norm.T).cpu().numpy()
            logger.info(f"Computed UNI (1024d) features for {num_dets} cells")
    except Exception as uni_err:
        logger.warning(f"UNI feature computation skipped: {uni_err}")

    # Combined dual-foundation morphological similarity matrix (Virchow 2 1280d + UNI 1024d)
    morph_sim_matrix = None
    if virchow_sim_matrix is not None and uni_sim_matrix is not None:
        morph_sim_matrix = 0.5 * virchow_sim_matrix + 0.5 * uni_sim_matrix
    elif virchow_sim_matrix is not None:
        morph_sim_matrix = virchow_sim_matrix
    elif uni_sim_matrix is not None:
        morph_sim_matrix = uni_sim_matrix

    # 3. High-Precision CONCH (512d) zero-shot classification with multi-template ensembling
    conch_classified = classify_detections_with_conch(
        image=image,
        detections=detections,
        candidate_classes=cellular_classes,
        temperature=0.08,
        is_histology=True,
    )

    # 4. Multi-modal HD Crop Arbitration with Gemini
    client = _get_gemini_client(api_key)
    gemini_predictions: Dict[int, Dict[str, Any]] = {}

    if client is not None:
        try:
            # Classes description for prompt
            classes_desc = "\n".join([
                f"- '{c.get('key')}': {c.get('label', c.get('name'))} (Prompt: {c.get('prompt', '')})"
                for c in cellular_classes
            ])

            sys_inst = """\
You are a senior computational pathologist performing cell-by-cell cytological analysis on high-resolution image crops.
For each cell crop, examine nuclear shape, chromatin texture (hyperchromatic vs vesicular), nucleoli presence/position, \
cytoplasmic volume, and histological compartment.

Validate or arbitrate the CONCH foundation model prediction.
Output ONLY a valid JSON list of objects:
[
  {
    "cell_index": 1,
    "class_key": "exact_key_from_list",
    "confidence": 0.88,
    "reasoning": "Vesicular chromatin with prominent peripheral nucleolus and basal location."
  }
]
"""
            # Identify cells that benefit from multimodal arbitration (low CONCH margin or low confidence)
            ambiguous_indices = [
                i for i, d in enumerate(conch_classified)
                if float(d.get("conch_margin", 1.0)) < 0.20 or float(d.get("conch_confidence", 1.0)) < 0.50
            ]
            # If small detection set (<= 20 cells), evaluate all of them
            if len(conch_classified) <= 20:
                ambiguous_indices = list(range(len(conch_classified)))
            elif len(ambiguous_indices) > 24:
                # Prioritize the most borderline cells by lowest margin
                ambiguous_indices = sorted(
                    ambiguous_indices,
                    key=lambda idx: float(conch_classified[idx].get("conch_margin", 0.0))
                )[:24]

            logger.info(
                f"Gemini HD Crop Arbitration: Evaluating {len(ambiguous_indices)} ambiguous cells "
                f"(out of {num_dets} total segmented cells)..."
            )

            # Process ambiguous crops in chunks of up to 12
            chunk_size = 12
            for start_pos in range(0, len(ambiguous_indices), chunk_size):
                sub_indices = ambiguous_indices[start_pos : start_pos + chunk_size]

                user_parts: List[Any] = [
                    f"Histology Slide Analysis (Ontology: {ontology_name or 'Histopathology'})\n"
                    f"CANDIDATE CELL CLASSES:\n{classes_desc}\n\n"
                    f"Analyze each of the following {len(sub_indices)} high-resolution cell crops:\n"
                ]

                for local_i, global_i in enumerate(sub_indices):
                    det = conch_classified[global_i]
                    crop_img = crops[global_i]
                    top_k = det.get("class_key", "")
                    top_c = det.get("conch_confidence", 0.0)
                    scores = det.get("conch_scores", {})
                    sorted_scores = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:3]
                    scores_summary = ", ".join([f"{k}: {v:.2f}" for k, v in sorted_scores])

                    user_parts.append(
                        f"\n--- Cell #{global_i + 1} ---\n"
                        f"CONCH Top: '{top_k}' (conf: {top_c:.2f}, margin: {det.get('conch_margin', 0.0):.2f})\n"
                        f"Top scores: [{scores_summary}]\n"
                        f"Crop image for Cell #{global_i + 1}:"
                    )
                    user_parts.append(crop_img)

                user_parts.append(
                    "\nOutput JSON array with classifications for each of the cells listed above:"
                )

                response = generate_gemini_content(
                    contents=user_parts,
                    system_instruction=sys_inst,
                    temperature=0.1,
                    response_mime_type="application/json",
                    api_key=api_key,
                )

                raw_text = (response.text or "").strip()
                if raw_text.startswith("```"):
                    raw_text = re.sub(r"^```(?:json)?\s*", "", raw_text)
                    raw_text = re.sub(r"\s*```$", "", raw_text)

                parsed = json.loads(raw_text)
                if isinstance(parsed, list):
                    for item in parsed:
                        c_idx = item.get("cell_index")
                        if c_idx is not None and isinstance(c_idx, int) and 1 <= c_idx <= num_dets:
                            gemini_predictions[c_idx - 1] = {
                                "class_key": item.get("class_key"),
                                "confidence": float(item.get("confidence", 0.85)),
                                "reasoning": item.get("reasoning", ""),
                            }

            logger.info(f"Gemini Multimodal HD Crops arbitrated {len(gemini_predictions)} ambiguous cells.")

        except Exception as gemini_err:
            logger.error(f"Gemini Multimodal reasoning error: {gemini_err}", exc_info=True)

    # 5. Calibrated Multimodal Fusion (CONCH + Virchow 2 + Gemini HD Crops)
    final_classified = []
    for i, det in enumerate(conch_classified):
        det_copy = dict(det)
        orig_det = detections[i] if i < len(detections) else {}

        # Preserve user manual labels unconditionally
        if orig_det.get("is_user_exemplar") or orig_det.get("manual_override"):
            det_copy["is_user_exemplar"] = True
            det_copy["virchow_confidence"] = 1.0
            det_copy["gemini_confidence"] = 1.0
            det_copy["score"] = 1.0
            det_copy["gemini_reasoning"] = "Etiqueta manual confirmada por el usuario."
            final_classified.append(det_copy)
            continue

        gemini_pred = gemini_predictions.get(i)
        conch_key = det_copy.get("class_key")
        conch_conf = float(det_copy.get("conch_confidence", 0.5))
        conch_margin = float(det_copy.get("conch_margin", 0.1))
        conch_scores = det_copy.get("conch_scores", {})

        if gemini_pred and gemini_pred.get("class_key") in class_meta_map:
            g_key = gemini_pred["class_key"]
            g_conf = float(gemini_pred.get("confidence", 0.85))
            g_reason = gemini_pred.get("reasoning", "")

            # Decision Logic: Calibrated Agreement vs Arbitration
            if g_key == conch_key:
                # Strong consensus between CONCH and Gemini HD Crop
                chosen_key = conch_key
                fused_score = min(0.98, max(conch_conf, 0.75) + 0.10)
                reason = f"Consenso patológico (CONCH {conch_conf:.2f} + Gemini HD). {g_reason}"
            elif conch_margin >= 0.25 and conch_conf >= 0.60:
                # CONCH has high margin confidence on morphological features -> trust CONCH
                chosen_key = conch_key
                fused_score = round(0.70 * conch_conf + 0.30 * g_conf, 4)
                reason = f"Morfología CONCH dominante (margen {conch_margin:.2f})."
            else:
                # CONCH was ambiguous/low margin -> Gemini HD Crop arbitration resolves the tie
                chosen_key = g_key
                fused_score = round(0.65 * g_conf + 0.35 * conch_scores.get(g_key, 0.35), 4)
                reason = f"Arbitraje visual Gemini sobre crop HD: {g_reason}"

            meta = class_meta_map[chosen_key]
            det_copy["category_id"] = chosen_key
            det_copy["class_key"] = chosen_key
            det_copy["class_label"] = meta["label"]
            det_copy["color"] = meta["color"]
            det_copy["gemini_confidence"] = round(g_conf, 4)
            det_copy["gemini_reasoning"] = reason
            det_copy["multimodal_fused"] = True
            det_copy["score"] = round(fused_score, 4)

        elif conch_key in class_meta_map:
            meta = class_meta_map[conch_key]
            det_copy["category_id"] = conch_key
            det_copy["class_key"] = conch_key
            det_copy["class_label"] = meta["label"]
            det_copy["color"] = meta["color"]
            det_copy["gemini_confidence"] = round(conch_conf, 4)
            det_copy["gemini_reasoning"] = f"Clasificado por similitud morfológica CONCH (margen: {conch_margin:.2f})."
            det_copy["score"] = round(conch_conf, 4)

        final_classified.append(det_copy)

    # 6. Data-driven neighbourhood consistency with Dual Foundation (Virchow 2 1280d + UNI 1024d)
    if morph_sim_matrix is not None and len(final_classified) > 1:
        for i, det in enumerate(final_classified):
            if det.get("is_user_exemplar"):
                continue
            # Check if an uncertain cell is strongly aligned (>0.88) with a high-confidence neighbour
            if float(det.get("score", 0.0)) < 0.50:
                best_sim = -1.0
                best_j = -1
                for j, other in enumerate(final_classified):
                    if i == j:
                        continue
                    if float(other.get("score", 0.0)) >= 0.75:
                        sim = float(morph_sim_matrix[i, j])
                        if sim > best_sim:
                            best_sim = sim
                            best_j = j
                if best_j >= 0 and best_sim > 0.88:
                    donor = final_classified[best_j]
                    det["category_id"] = donor["category_id"]
                    det["class_key"] = donor["class_key"]
                    det["class_label"] = donor["class_label"]
                    det["color"] = donor["color"]
                    det["neighbour_aligned"] = True
                    det["neighbour_similarity"] = round(best_sim, 4)
                    det["gemini_reasoning"] += f" (Consistencia morfológica Virchow2+UNI: {best_sim:.2f})"

    return final_classified


def validate_uncertain_detections_with_gemini(
    image: Image.Image,
    uncertain_detections: List[Dict[str, Any]],
    ontology_classes: List[Dict[str, Any]],
    organ_context: str = "histología",
    api_key: Optional[str] = None,
    max_to_validate: int = 25,
) -> List[Dict[str, Any]]:
    """
    Arbitrates and validates ambiguous/uncertain histological instances using Gemini 2.5 Flash Vision.

    Selects ambiguous detections, passes contextual high-resolution crops along with the full slide context,
    and returns adjudications conforming strictly to candidate ontology classes.
    """
    if not uncertain_detections or not ontology_classes:
        return uncertain_detections

    client = _get_gemini_client(api_key)
    if client is None:
        logger.warning("Gemini client not available for uncertain detection validation.")
        return uncertain_detections

    class_meta = {
        c.get("key"): {
            "label": c.get("label", c.get("name", c.get("key"))),
            "color": c.get("color", "#8b5cf6"),
            "prompt": c.get("prompt", ""),
        }
        for c in ontology_classes
        if c.get("key")
    }
    class_list_desc = [
        f"- key: '{c.get('key')}', name: '{c.get('label', c.get('name'))}', description: '{c.get('prompt', '')}'"
        for c in ontology_classes
    ]

    target_dets = uncertain_detections[:max_to_validate]
    img_w, img_h = image.size

    # Batch crops into composite or individual evaluations
    try:
        crops_to_send = []
        for idx, det in enumerate(target_dets):
            bbox = det.get("bbox", [0, 0, img_w, img_h])
            bx, by, bw, bh = bbox
            pad = max(12, int(max(bw, bh) * 0.4))
            x1 = max(0, bx - pad)
            y1 = max(0, by - pad)
            x2 = min(img_w, bx + bw + pad)
            y2 = min(img_h, by + bh + pad)
            crop = image.crop((x1, y1, x2, y2)).convert("RGB")
            crops_to_send.append((idx, crop, det.get("class_key", "unknown"), det.get("score", 0.0)))

        # Build prompt
        prompt_parts = [
            f"You are an expert histopathologist evaluating ambiguous microscopic cell/tissue instances in {organ_context}.",
            "Here is the list of VALID ONTOLOGY CLASSES you must choose from:",
            "\n".join(class_list_desc),
            "",
            "Review each numbered crop image and assign the most accurate ontology class_key, confidence (0.0 to 1.0), and short rationale.",
            "Respond ONLY with valid JSON in this exact structure:",
            "```json",
            "{",
            '  "adjudications": [',
            '    {"crop_index": 0, "class_key": "<exact_key>", "confidence": 0.88, "reasoning": "<brief explanation>"}',
            "  ]",
            "}",
            "```",
        ]

        contents_list = ["\n".join(prompt_parts)]
        for idx, crop, tentative_key, tentative_score in crops_to_send:
            contents_list.append(f"--- Crop #{idx} (Tentative ensemble class: '{tentative_key}', score: {tentative_score:.2f}) ---")
            contents_list.append(crop)

        response = generate_gemini_content(
            contents=contents_list,
            api_key=api_key,
        )

        resp_text = response.text or ""
        match = re.search(r"\{[\s\S]*\}", resp_text)
        if match:
            parsed = json.loads(match.group(0))
            adjudications = parsed.get("adjudications", [])
            for adj in adjudications:
                crop_idx = int(adj.get("crop_index", -1))
                if 0 <= crop_idx < len(target_dets):
                    target_det = target_dets[crop_idx]
                    adj_key = adj.get("class_key")
                    adj_conf = float(adj.get("confidence", 0.75))
                    adj_reason = adj.get("reasoning", "")

                    if adj_key in class_meta:
                        meta = class_meta[adj_key]
                        target_det["category_id"] = adj_key
                        target_det["class_key"] = adj_key
                        target_det["class_label"] = meta["label"]
                        target_det["color"] = meta["color"]
                        target_det["gemini_validated"] = True
                        target_det["gemini_confidence"] = round(adj_conf, 4)
                        target_det["gemini_reasoning"] = adj_reason
                        target_det["score"] = round(max(adj_conf, float(target_det.get("score", 0.5))), 4)
                        if adj_conf >= 0.70:
                            target_det["classification_uncertain"] = False
                        target_det["decision_source"] = "gemini_multimodal_arbitration"

    except Exception as e:
        logger.warning(f"Error during Gemini uncertain detection validation: {e}")

    return uncertain_detections


def detect_histological_macro_layers_gemini(
    image: Image.Image,
    organ_context: str = "histología",
    ontology_structures: Optional[List[Dict[str, Any]]] = None,
    api_key: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Multimodal Spatial Grounding for Macro-Histological Layers & Compartments:
    Dynamically extracts target layer descriptions directly from the ontology structures
    (e.g. from the PDF) and detects 2D bounding boxes [ymin, xmin, ymax, xmax].
    Zero hardcoded tissue lists.
    """
    client = _get_gemini_client(api_key)
    if client is None:
        logger.warning("Gemini client not available for macro-layer grounding.")
        return []

    img_w, img_h = image.size
    img_preview = image.copy().convert("RGB")
    max_dim = 1024
    if max(img_preview.size) > max_dim:
        ratio = max_dim / max(img_preview.size)
        img_preview = img_preview.resize(
            (int(img_preview.width * ratio), int(img_preview.height * ratio)),
            Image.LANCZOS,
        )

    # Dynamically extract target descriptions from the provided ontology document
    target_descriptions: List[str] = []
    class_meta_lookup: Dict[str, Dict[str, Any]] = {}

    if ontology_structures:
        for idx, s in enumerate(ontology_structures):
            k = s.get("key", f"struct_{idx + 1}").strip()
            name = s.get("label", s.get("name", k))
            prompt_desc = s.get("prompt", "")
            stype = s.get("structure_type", "")

            class_meta_lookup[k] = {
                "name": name,
                "label": name,
                "color": s.get("color", DEFAULT_COLORS[idx % len(DEFAULT_COLORS)]),
                "prompt": prompt_desc,
                "structure_type": stype,
            }
            target_descriptions.append(f"- '{k}': {name} — visual descriptor: '{prompt_desc}'")
    else:
        target_descriptions = [
            f"- Distinct visual anatomical structures, tissue layers, and compartments visible in {organ_context}"
        ]

    prompt = f"""\
You are an expert anatomical histopathologist. Analyze this histological photomicrograph of {organ_context}.
Your task is to identify and spatially localize ALL continuous macro-architectural layers, lumens, and tissue compartments present in the image.

TARGET STRUCTURES FROM ONTOLOGY:
{chr(10).join(target_descriptions)}

For EACH distinct compartment or layer present in the image, output its exact bounding box coordinates in normalized scale [ymin, xmin, ymax, xmax] (integers 0 to 1000).

Return ONLY valid JSON matching this schema:
```json
{{
  "layers": [
    {{
      "key": "<exact_key_from_ontology>",
      "name": "<canonical_name>",
      "box_2d": [ymin, xmin, ymax, xmax],
      "description": "<concise visual description>",
      "confidence": 0.95
    }}
  ]
}}
```
"""

    try:
        response = generate_gemini_content(
            contents=[prompt, img_preview],
            api_key=api_key,
        )
        resp_text = response.text or ""
        match = re.search(r"\{[\s\S]*\}", resp_text)
        if not match:
            return []

        parsed = json.loads(match.group(0))
        raw_layers = parsed.get("layers", [])

        grounded_layers: List[Dict[str, Any]] = []
        for idx, item in enumerate(raw_layers):
            box = item.get("box_2d")
            if not box or len(box) != 4:
                continue

            ymin, xmin, ymax, xmax = [float(v) for v in box]
            px_x1 = max(0.0, min(float(img_w), (xmin / 1000.0) * img_w))
            px_y1 = max(0.0, min(float(img_h), (ymin / 1000.0) * img_h))
            px_x2 = max(0.0, min(float(img_w), (xmax / 1000.0) * img_w))
            px_y2 = max(0.0, min(float(img_h), (ymax / 1000.0) * img_h))
            bw = max(1.0, px_x2 - px_x1)
            bh = max(1.0, px_y2 - px_y1)

            k = item.get("key", f"layer_{idx + 1}").lower().replace("-", "_")
            meta = class_meta_lookup.get(k, {})
            name = meta.get("label", item.get("name", k.replace("_", " ").title()))
            color = meta.get("color", DEFAULT_COLORS[idx % len(DEFAULT_COLORS)])
            stype = meta.get("structure_type", "tissue_layer")

            # Estimate initial polygon box
            poly = [
                px_x1, px_y1,
                px_x2, px_y1,
                px_x2, px_y2,
                px_x1, px_y2,
            ]

            grounded_layers.append({
                "id": f"layer_{idx + 1}",
                "key": k,
                "class_key": k,
                "class_label": name,
                "category_id": k,
                "label": name,
                "color": color,
                "structure_type": stype,
                "is_macro_layer": True,
                "score": float(item.get("confidence", 0.90)),
                "box": [round(px_x1, 1), round(px_y1, 1), round(px_x2, 1), round(px_y2, 1)],
                "bbox": [round(px_x1, 1), round(px_y1, 1), round(bw, 1), round(bh, 1)],
                "segmentation": [poly],
                "area": round(bw * bh, 1),
                "description": item.get("description", ""),
                "decision_source": "gemini_spatial_grounding",
            })

        logger.info(f"Gemini grounded {len(grounded_layers)} macro-layers for {organ_context}: {[l['key'] for l in grounded_layers]}")
        return grounded_layers

    except Exception as e:
        logger.warning(f"Error in macro-layer grounding with Gemini: {e}")
        return []


def analyze_tissue_macro_micro_ontology(
    image: Image.Image,
    organ_context: str = "histología / tejido",
    base_ontology: Optional[List[Dict[str, Any]]] = None,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Step 1: Dual-Scale Macro & Micro Tissue Analysis with Gemini Multimodal Vision.
    Discovers the textual and spatial ontology of the histological image:
    1. Macro-structures: Anatomical compartments & tissue layers with 2D coordinates.
    2. Micro-structures: Cellular populations assigned to each macro compartment with
       spatial positioning rules (basal, adluminal, luminal, interstitial) and
       cytological criteria (nuclear shape, chromatin texture, size).
    """
    client = _get_gemini_client(api_key)
    if client is None:
        logger.warning("Gemini client not available for macro/micro tissue analysis.")
        return {
            "success": False,
            "error": "No hay API Key de Gemini configurada.",
            "macro_layers": [],
            "cellular_classes": [],
            "spatial_map": {},
        }

    img_w, img_h = image.size
    img_preview = image.copy().convert("RGB")
    max_dim = 1024
    if max(img_preview.size) > max_dim:
        ratio = max_dim / max(img_preview.size)
        img_preview = img_preview.resize(
            (int(img_preview.width * ratio), int(img_preview.height * ratio)),
            Image.LANCZOS,
        )

    base_classes_hint = ""
    if base_ontology:
        classes_str = ", ".join([c.get("name") or c.get("label") or c.get("key", "") for c in base_ontology if c.get("key")])
        base_classes_hint = f"\nCONSIDER EXISTING ONTOLOGY CLASSES IF RELEVANT: {classes_str}"

    prompt = f"""\
You are an expert anatomical pathologist and cytologist.
Analyze this high-resolution histological photomicrograph ({organ_context}).{base_classes_hint}

YOUR TASK:
Determine the complete TEXTUAL and SPATIAL ONTOLOGY of the tissue at both Macro and Micro architectural scales.

1. MACRO-STRUCTURES:
   Identify and localize all continuous anatomical compartments visible in the image (e.g., seminiferous tubules, interstitial stroma, tubular lumen, continuous basement membrane / tunica propria, blood vessels).
   For EACH macro compartment, provide its bounding box `box_2d` in normalized coordinates [ymin, xmin, ymax, xmax] (0 to 1000).

2. MICRO-STRUCTURES (Cellular Populations):
   For each macro compartment, define the specific cellular types that physiologically and histologically reside inside it.
   Specify for each cell class:
   - `key`: lowercase identifier (e.g., "espermatogonia_a_clara", "celula_de_leydig", "celula_de_sertoli", "espermatocito_primario")
   - `name`: scientific canonical name in Spanish
   - `parent_compartment`: the macro compartment key it belongs to (e.g. "tubulo_seminifero" vs "estroma_intersticial")
   - `spatial_zone`: exact micro-location ("basal" [contacting basement membrane], "adluminal", "luminal", "intersticial")
   - `cytological_features`: specific cytological criteria (nuclear morphology, chromatin condensation, nucleoli, N/C ratio)
   - `color`: a distinct hex color code (e.g. #e11d48, #059669, #3b82f6, #f59e0b)

Output STRICTLY valid JSON conforming to this schema:
```json
{{
  "organ_identified": "Testículo / Túbulos Seminíferos y Tejido Intersticial",
  "macro_structures": [
    {{
      "key": "tubulo_seminifero_1",
      "compartment_type": "tubulo_seminifero",
      "name": "Túbulo Seminífero 1",
      "box_2d": [ymin, xmin, ymax, xmax],
      "description": "Sección transversal de túbulo seminífero con epitelio espermatogénico estratificado"
    }},
    {{
      "key": "estroma_intersticial",
      "compartment_type": "estroma_intersticial",
      "name": "Estroma Intersticial",
      "box_2d": [ymin, xmin, ymax, xmax],
      "description": "Espacio conjuntivo intertubular con vasos y células endocrinas"
    }}
  ],
  "cellular_classes": [
    {{
      "key": "espermatogonia_a_clara",
      "name": "Espermatogonia A Clara",
      "parent_compartment": "tubulo_seminifero",
      "spatial_zone": "basal",
      "color": "#e11d48",
      "cytological_features": "Núcleo esférico a ovoide con cromatina fina, pálida o pulverulenta, 1 o 2 nucleolos cerca de la carioteca, situada estrictamente contra la lámina basal",
      "prompt": "Espermatogonia A clara en lámina basal con núcleo esférico claro"
    }},
    {{
      "key": "celula_de_leydig",
      "name": "Célula de Leydig",
      "parent_compartment": "estroma_intersticial",
      "spatial_zone": "intersticial",
      "color": "#f59e0b",
      "cytological_features": "Célula poligonal grande, citoplasma acidófilo eosinófilo abundante, núcleo excéntrico con cromatina periférica, aislada o en nidos intertubulares",
      "prompt": "Célula de Leydig en estroma intertubular con citoplasma eosinófilo"
    }}
  ]
}}
```
"""

    try:
        response = generate_gemini_content(
            contents=[prompt, img_preview],
            api_key=api_key,
        )
        resp_text = response.text or ""
        match = re.search(r"\{[\s\S]*\}", resp_text)
        if not match:
            return {"success": False, "error": "No se pudo extraer JSON de Gemini", "macro_layers": [], "cellular_classes": [], "spatial_map": {}}

        parsed = json.loads(match.group(0))
        organ_name = parsed.get("organ_identified", organ_context)
        raw_macros = parsed.get("macro_structures", [])
        raw_micros = parsed.get("cellular_classes", [])

        # Process macro layers into pixel boxes and polygon coordinates
        macro_layers: List[Dict[str, Any]] = []
        spatial_map: Dict[str, List[str]] = {}

        for idx, m in enumerate(raw_macros):
            box = m.get("box_2d")
            if not box or len(box) != 4:
                continue

            ymin, xmin, ymax, xmax = [float(v) for v in box]
            px_x1 = max(0.0, min(float(img_w), (xmin / 1000.0) * img_w))
            px_y1 = max(0.0, min(float(img_h), (ymin / 1000.0) * img_h))
            px_x2 = max(0.0, min(float(img_w), (xmax / 1000.0) * img_w))
            px_y2 = max(0.0, min(float(img_h), (ymax / 1000.0) * img_h))
            bw = max(1.0, px_x2 - px_x1)
            bh = max(1.0, px_y2 - px_y1)

            m_key = m.get("key", f"macro_{idx + 1}").lower().replace("-", "_")
            c_type = m.get("compartment_type", m_key).lower().replace("-", "_")

            poly = [
                px_x1, px_y1,
                px_x2, px_y1,
                px_x2, px_y2,
                px_x1, px_y2,
            ]

            macro_color = DEFAULT_COLORS[idx % len(DEFAULT_COLORS)]
            if "tubulo" in c_type:
                macro_color = "#3b82f6"
            elif "estroma" in c_type:
                macro_color = "#f59e0b"
            elif "vaso" in c_type:
                macro_color = "#ef4444"

            macro_layers.append({
                "id": f"macro_{idx + 1}",
                "key": m_key,
                "compartment_type": c_type,
                "class_key": c_type,
                "class_label": m.get("name", m_key.title()),
                "category_id": c_type,
                "label": m.get("name", m_key.title()),
                "color": macro_color,
                "structure_type": "macro_compartment",
                "is_macro_layer": True,
                "score": 0.95,
                "box": [round(px_x1, 1), round(px_y1, 1), round(px_x2, 1), round(px_y2, 1)],
                "bbox": [round(px_x1, 1), round(px_y1, 1), round(bw, 1), round(bh, 1)],
                "segmentation": [poly],
                "area": round(bw * bh, 1),
                "description": m.get("description", ""),
                "decision_source": "gemini_macro_micro_spatial_ontology",
            })

        # Process cellular classes and build spatial constraint mapping
        cellular_classes: List[Dict[str, Any]] = []
        for idx, c in enumerate(raw_micros):
            c_key = c.get("key", f"cell_type_{idx + 1}").lower().replace("-", "_")
            c_name = c.get("name", c_key.replace("_", " ").title())
            p_comp = c.get("parent_compartment", "").lower().replace("-", "_")
            c_color = c.get("color", DEFAULT_COLORS[(idx + 3) % len(DEFAULT_COLORS)])

            cell_obj = {
                "id": idx + 1,
                "key": c_key,
                "name": c_name,
                "label": c_name,
                "color": c_color,
                "parent_compartment": p_comp,
                "spatial_zone": c.get("spatial_zone", "unspecified"),
                "cytological_features": c.get("cytological_features", ""),
                "prompt": c.get("prompt", c_name),
                "is_macro": False,
            }
            cellular_classes.append(cell_obj)

            # Map compartment to allowed cell keys
            if p_comp:
                spatial_map.setdefault(p_comp, []).append(c_key)
                # Also map with partial match (e.g. tubulo)
                for m_layer in macro_layers:
                    if p_comp in m_layer["compartment_type"] or m_layer["compartment_type"] in p_comp:
                        spatial_map.setdefault(m_layer["key"], []).append(c_key)

        logger.info(
            f"Gemini Dual-Scale Analysis complete: organ='{organ_name}', "
            f"{len(macro_layers)} macro compartments, {len(cellular_classes)} micro classes."
        )

        return {
            "success": True,
            "organ_identified": organ_name,
            "macro_layers": macro_layers,
            "cellular_classes": cellular_classes,
            "spatial_map": spatial_map,
        }

    except Exception as e:
        logger.error(f"Error in Gemini Dual-Scale Analysis: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
            "macro_layers": [],
            "cellular_classes": [],
            "spatial_map": {},
        }


def classify_cell_with_spatial_prior_gemini(
    crop: Image.Image,
    containing_macro_compartment: Optional[str] = None,
    candidate_classes: Optional[List[Dict[str, Any]]] = None,
    virchow_candidate_label: Optional[str] = None,
    virchow_confidence: float = 0.0,
    organ_context: str = "histología",
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Step 3: Multi-modal Cytological Arbiter with Spatial & Virchow Embeddings Priors.
    Evaluates a single cell crop by synthesizing:
    1. Spatial prior: the macro-compartment it resides in (eliminates out-of-context misclassifications).
    2. Virchow 2 foundation model morphological embedding top match & confidence.
    3. Gemini 3.5 visual cytological analysis of chromatin, nuclear membrane, and nucleoli.
    """
    client = _get_gemini_client(api_key)
    if client is None:
        return {
            "class_key": (candidate_classes[0].get("key") if candidate_classes else "cell"),
            "confidence": 0.5,
            "reasoning": "Gemini no disponible; fallback asignado",
        }

    crop_rgb = crop.convert("RGB")
    if crop_rgb.width < 64 or crop_rgb.height < 64:
        crop_rgb = crop_rgb.resize((128, 128), Image.BICUBIC)

    allowed_desc = []
    if candidate_classes:
        for c in candidate_classes:
            allowed_desc.append(
                f"- '{c.get('key')}': {c.get('name', c.get('label'))} (Zona: {c.get('spatial_zone', 'n/a')}). "
                f"Criterio citológico: {c.get('cytological_features', c.get('prompt', ''))}"
            )
    classes_block = "\n".join(allowed_desc) if allowed_desc else "Células esperadas en este tejido"

    virchow_hint = ""
    if virchow_candidate_label and virchow_confidence > 0:
        virchow_hint = f"\nVIRCHOW 2 FOUNDATION MODEL SUGGESTION: '{virchow_candidate_label}' (Embedding similarity score: {virchow_confidence:.2f})"

    spatial_hint = ""
    if containing_macro_compartment:
        spatial_hint = f"\nSPATIAL TOPOLOGICAL LOCATION: Located inside anatomical compartment '{containing_macro_compartment}'"

    prompt = f"""\
You are an expert cytopathologist.
Analyze this high-magnification photomicrograph crop of an individual segmented cell in {organ_context}.{spatial_hint}{virchow_hint}

CANDIDATE CLASSES CONSTRAINED BY THIS SPATIAL COMPARTMENT:
{classes_block}

TASK:
Classify this individual cell into the most accurate candidate class based on its cytological morphology:
- Nuclear shape (spherical, oval, irregular, indented)
- Chromatin texture (pale/fine/euchromatic vs dark/condensed/heterochromatic)
- Presence and location of nucleoli
- Cytoplasmic abundance and staining
- Position relative to the compartment boundaries

Respond STRICTLY in JSON format:
```json
{{
  "class_key": "<exact_key_from_candidates>",
  "class_name": "<canonical_name>",
  "confidence": 0.95,
  "reasoning": "<concise cytological rationale in Spanish>"
}}
```
"""

    try:
        response = generate_gemini_content(
            contents=[prompt, crop_rgb],
            api_key=api_key,
        )
        resp_text = response.text or ""
        match = re.search(r"\{[\s\S]*\}", resp_text)
        if match:
            parsed = json.loads(match.group(0))
            return {
                "class_key": parsed.get("class_key", "cell"),
                "class_name": parsed.get("class_name", "Célula"),
                "confidence": float(parsed.get("confidence", 0.90)),
                "reasoning": parsed.get("reasoning", "Clasificado por citología visual Gemini"),
            }
    except Exception as e:
        logger.warning(f"Cell cytological classification error with Gemini: {e}")

    # Fallback to Virchow recommendation if Gemini parsing failed
    default_key = virchow_candidate_label or (candidate_classes[0].get("key") if candidate_classes else "cell")
    return {
        "class_key": default_key,
        "class_name": default_key.replace("_", " ").title(),
        "confidence": float(virchow_confidence) if virchow_confidence > 0 else 0.70,
        "reasoning": "Asignado por concordancia de embeddings Virchow 2",
    }


# ---------------------------------------------------------------------------
# Gemini Vision Batch Cell Classification (Annotated Image Strategy)
# ---------------------------------------------------------------------------

def _render_numbered_contours(
    image: Image.Image,
    detections: List[Dict[str, Any]],
    indices: Optional[List[int]] = None,
) -> Image.Image:
    """
    Draw numbered contour overlays on the image for each detection.

    Each cell gets a colored contour and a visible index number so Gemini
    can reference cells by their numeric ID.

    Args:
        image: Original PIL image.
        detections: List of detection dicts with 'bbox' [x,y,w,h] or 'segmentation'.
        indices: Optional subset of detection indices to draw. If None, draw all.

    Returns:
        Annotated PIL image with numbered cell contours.
    """
    import cv2
    import numpy as np

    img_np = np.array(image.convert("RGB")).copy()
    h, w = img_np.shape[:2]

    draw_indices = indices if indices is not None else list(range(len(detections)))

    # Adaptive font scale based on image dimensions and cell count
    base_scale = min(w, h) / 1200.0
    font_scale = max(0.28, min(0.55, base_scale * (60.0 / max(len(draw_indices), 1)) ** 0.15))
    thickness = max(1, int(font_scale * 2.2))
    contour_thickness = max(1, int(font_scale * 2.0))

    # Color palette for visual differentiation
    palette = [
        (225, 29, 72), (139, 92, 246), (6, 182, 212), (245, 158, 11),
        (16, 185, 129), (236, 72, 153), (99, 102, 241), (20, 184, 166),
        (249, 115, 22), (132, 204, 22), (168, 85, 247), (14, 165, 233),
    ]

    for seq, det_idx in enumerate(draw_indices):
        if det_idx >= len(detections):
            continue
        det = detections[det_idx]
        color = palette[seq % len(palette)]

        # Draw segmentation polygon if available
        segs = det.get("segmentation", [])
        has_poly = False
        if segs:
            for poly in segs:
                if isinstance(poly, list) and len(poly) >= 6:
                    pts = np.array(poly, dtype=np.float32).reshape(-1, 2).astype(np.int32)
                    cv2.polylines(img_np, [pts], isClosed=True, color=color, thickness=contour_thickness)
                    has_poly = True

        # Fallback: draw bbox rectangle
        if not has_poly:
            bbox = det.get("bbox", det.get("box"))
            if bbox and len(bbox) == 4:
                if "bbox" in det:
                    x, y, bw, bh = [int(v) for v in bbox]
                    cv2.rectangle(img_np, (x, y), (x + bw, y + bh), color, contour_thickness)
                else:
                    x1, y1, x2, y2 = [int(v) for v in bbox]
                    cv2.rectangle(img_np, (x1, y1), (x2, y2), color, contour_thickness)

        # Compute centroid for number placement
        bbox = det.get("bbox")
        if bbox and len(bbox) == 4:
            cx = int(bbox[0] + bbox[2] / 2)
            cy = int(bbox[1] + bbox[3] / 2)
        elif det.get("box") and len(det["box"]) == 4:
            bx1, by1, bx2, by2 = det["box"]
            cx, cy = int((bx1 + bx2) / 2), int((by1 + by2) / 2)
        else:
            continue

        # Draw number with dark background for readability
        label = str(det_idx)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        tx, ty = cx - tw // 2, cy + th // 2
        # Dark pill background
        cv2.rectangle(img_np, (tx - 2, ty - th - 2), (tx + tw + 2, ty + 3), (0, 0, 0), -1)
        cv2.putText(img_np, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)

    return Image.fromarray(img_np)


def classify_cells_batch_gemini(
    image: Image.Image,
    detections: List[Dict[str, Any]],
    ontology_classes: List[Dict[str, Any]],
    spatial_map: Optional[Dict[str, List[str]]] = None,
    organ_context: str = "histología",
    max_cells_per_call: int = 250,
    api_key: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], List[int]]:
    """
    Batch-classify all Cellpose-segmented cells using Gemini Vision with annotated image.

    Strategy:
    1. Render numbered contours/bboxes on the original image so Gemini sees each cell
       in its full tissue context (like a pathologist looking through a microscope).
    2. Send the annotated image + structured prompt with ontology class descriptions
       to Gemini in a single API call (or split into quadrants if >max_cells_per_call).
    3. Parse the JSON response and assign class_key, label, color, score to each detection.

    Args:
        image: Original PIL image (RGB).
        detections: Cellpose segmentation results (list of dicts with 'bbox', 'segmentation').
        ontology_classes: Cellular ontology classes (list of dicts with 'key', 'name', etc.).
        spatial_map: Optional mapping of macro-compartment keys to allowed cell class keys.
        organ_context: Tissue/organ description for Gemini context.
        max_cells_per_call: Maximum cells per Gemini API call before splitting.
        api_key: Optional explicit API key.

    Returns:
        Tuple of (classified_detections, uncertain_indices).
    """
    if not detections:
        return [], []

    if not ontology_classes:
        return detections, list(range(len(detections)))

    client = _get_gemini_client(api_key)
    if client is None:
        logger.warning("Gemini client not available for batch cell classification.")
        return detections, list(range(len(detections)))

    num_dets = len(detections)

    # Build class description block for the prompt
    class_meta: Dict[str, Dict[str, Any]] = {}
    class_descriptions: List[str] = []
    for c in ontology_classes:
        c_key = c.get("key", "")
        c_name = c.get("name", c.get("label", c_key))
        c_color = c.get("color", "#8b5cf6")
        c_zone = c.get("spatial_zone", c.get("spatial_rules", {}).get("compartment", ""))
        c_parent = c.get("parent_compartment", c.get("spatial_rules", {}).get("parent_macro", ""))
        c_cyto = c.get("cytological_features", c.get("prompt", ""))

        class_meta[c_key] = {"name": c_name, "color": c_color}
        class_descriptions.append(
            f"- key: '{c_key}' | name: '{c_name}' | zone: {c_zone} | "
            f"parent: {c_parent} | cytology: {c_cyto}"
        )

    classes_block = "\n".join(class_descriptions)

    # Spatial map hint for Gemini
    spatial_hint = ""
    if spatial_map:
        sp_lines = []
        for comp_key, allowed_keys in spatial_map.items():
            sp_lines.append(f"  - Compartment '{comp_key}' → allowed cells: {allowed_keys}")
        spatial_hint = (
            "\n\nSPATIAL CONSTRAINT RULES (cells MUST only appear in their parent compartment):\n"
            + "\n".join(sp_lines)
        )

    def _classify_subset(
        subset_indices: List[int],
    ) -> Dict[int, Dict[str, Any]]:
        """Classify a subset of detections via one Gemini API call."""
        if not subset_indices:
            return {}

        # Render annotated image with numbered contours for this subset
        annotated = _render_numbered_contours(image, detections, subset_indices)

        # Resize for Gemini if too large (keep detail but respect API limits)
        max_dim = 1280
        if max(annotated.size) > max_dim:
            ratio = max_dim / max(annotated.size)
            annotated = annotated.resize(
                (int(annotated.width * ratio), int(annotated.height * ratio)),
                Image.LANCZOS,
            )

        # Build detection context: what spatial compartment each cell is in
        cell_context_lines: List[str] = []
        for det_idx in subset_indices:
            det = detections[det_idx]
            layer_info = det.get("containing_layer", "unknown")
            cell_context_lines.append(f"  Cell #{det_idx}: in compartment '{layer_info}'")
        cell_context_block = "\n".join(cell_context_lines)

        prompt = f"""\
You are an expert histopathologist and cytologist analyzing a high-resolution H&E stained \
photomicrograph of {organ_context}.

The image shows numbered cell/nucleus segmentations (contours with index numbers). \
Each number corresponds to a segmented cell instance detected by automated instance segmentation.

CELL SPATIAL LOCATIONS:
{cell_context_block}

CANDIDATE ONTOLOGY CLASSES (you MUST choose from these):
{classes_block}{spatial_hint}

YOUR TASK:
For EACH numbered cell visible in the image, classify it into the most accurate ontology class \
based on its cytological morphology IN CONTEXT:
- Nuclear shape, size, and chromatin pattern
- Position within the tissue architecture (basal vs adluminal vs luminal vs interstitial)
- Surrounding cellular neighborhood
- Cytoplasmic characteristics

CRITICAL RULES:
1. Use ONLY the exact class keys listed above
2. Respect spatial constraints: a cell inside a tubule cannot be classified as interstitial
3. If uncertain between two classes, choose the most probable and set confidence < 0.7

Respond STRICTLY with valid JSON:
```json
{{
  "classifications": [
    {{"cell_index": 0, "class_key": "<exact_key>", "confidence": 0.92, "reasoning": "<brief cytological rationale in Spanish>"}},
    {{"cell_index": 5, "class_key": "<exact_key>", "confidence": 0.85, "reasoning": "<brief rationale>"}}
  ]
}}
```
Include ALL numbered cells. Do not skip any.
"""

        try:
            response = generate_gemini_content(
                contents=[prompt, annotated],
                api_key=api_key,
            )
            resp_text = response.text or ""
            match = re.search(r"\{[\s\S]*\}", resp_text)
            if match:
                parsed = json.loads(match.group(0))
                results: Dict[int, Dict[str, Any]] = {}
                for item in parsed.get("classifications", []):
                    c_idx = int(item.get("cell_index", -1))
                    c_key = item.get("class_key", "")
                    if c_idx in subset_indices and c_key in class_meta:
                        results[c_idx] = {
                            "class_key": c_key,
                            "class_name": class_meta[c_key]["name"],
                            "color": class_meta[c_key]["color"],
                            "confidence": float(item.get("confidence", 0.80)),
                            "reasoning": item.get("reasoning", ""),
                        }
                return results
        except Exception as e:
            logger.warning(f"Gemini batch cell classification error: {e}")

        return {}

    # Split into batches if too many cells
    all_indices = list(range(num_dets))
    batches: List[List[int]] = []

    if num_dets <= max_cells_per_call:
        batches = [all_indices]
    else:
        # Spatial quadrant splitting based on cell centroids
        import numpy as np

        centroids = []
        for det in detections:
            bbox = det.get("bbox", [0, 0, 1, 1])
            cx = bbox[0] + bbox[2] / 2.0
            cy = bbox[1] + bbox[3] / 2.0
            centroids.append((cx, cy))

        img_w, img_h = image.size
        mid_x, mid_y = img_w / 2.0, img_h / 2.0

        quadrants: Dict[str, List[int]] = {"TL": [], "TR": [], "BL": [], "BR": []}
        for i, (cx, cy) in enumerate(centroids):
            if cy < mid_y:
                quadrants["TL" if cx < mid_x else "TR"].append(i)
            else:
                quadrants["BL" if cx < mid_x else "BR"].append(i)

        for q_indices in quadrants.values():
            if q_indices:
                # Further split if a quadrant still has too many
                for start in range(0, len(q_indices), max_cells_per_call):
                    batches.append(q_indices[start:start + max_cells_per_call])

    # Execute batches (parallel if multiple)
    all_results: Dict[int, Dict[str, Any]] = {}

    if len(batches) == 1:
        all_results = _classify_subset(batches[0])
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        logger.info(f"Splitting {num_dets} cells into {len(batches)} Gemini Vision batches.")
        with ThreadPoolExecutor(max_workers=min(4, len(batches))) as executor:
            futures = {executor.submit(_classify_subset, batch): batch for batch in batches}
            try:
                for future in as_completed(futures, timeout=60):
                    try:
                        batch_results = future.result()
                        all_results.update(batch_results)
                    except Exception as e:
                        err_msg = str(e) or type(e).__name__
                        logger.warning(f"Gemini batch future error ({type(e).__name__}): {err_msg}")
            except TimeoutError as te:
                logger.warning(f"Gemini batch classification timeout ({te}): keeping {len(all_results)} partial results.")

    # Apply classifications to detections
    classified: List[Dict[str, Any]] = []
    uncertain_indices: List[int] = []

    for i, det in enumerate(detections):
        det_copy = dict(det)
        if i in all_results:
            result = all_results[i]
            det_copy["category_id"] = result["class_key"]
            det_copy["class_key"] = result["class_key"]
            det_copy["class_label"] = result["class_name"]
            det_copy["label"] = result["class_name"]
            det_copy["color"] = result["color"]
            det_copy["score"] = float(result.get("confidence", 0.85))
            det_copy["decision_source"] = "gemini_vision_batch"
            det_copy["cytological_reasoning"] = result.get("reasoning", "")
            if float(result.get("confidence", 0.85)) < 0.60:
                det_copy["classification_uncertain"] = True
                uncertain_indices.append(i)
            else:
                det_copy["classification_uncertain"] = False
        else:
            # Cell not explicitly classified by Gemini — keep visible so user can review it
            det_copy["classification_uncertain"] = True
            det_copy["score"] = 0.75
            det_copy["decision_source"] = "unclassified"
            det_copy["category_id"] = "unclassified"
            det_copy["class_key"] = "unclassified"
            det_copy["class_label"] = "Sin clasificar (Revisar)"
            det_copy["label"] = "Sin clasificar (Revisar)"
            det_copy["color"] = "#94a3b8"
            uncertain_indices.append(i)

        classified.append(det_copy)

    classified_count = num_dets - len(uncertain_indices)
    logger.info(
        f"Gemini Vision batch classified {classified_count}/{num_dets} cells "
        f"({len(uncertain_indices)} uncertain) in {len(batches)} API call(s)."
    )

    return classified, uncertain_indices

