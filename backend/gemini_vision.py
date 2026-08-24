"""
Dynamic Multimodal Vision Assistant with Gemini 2.5 Flash.

Provides zero-shot visual prompt refinement and image structure discovery
without any hardcoded prompt lists or static translation dictionaries.
"""

import io
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional
from PIL import Image

logger = logging.getLogger("sam3-backend")

DEFAULT_COLORS = [
    "#e11d48", "#8b5cf6", "#06b6d4", "#f59e0b", "#10b981",
    "#ec4899", "#6366f1", "#14b8a6", "#f97316", "#84cc16",
    "#a855f7", "#0ea5e9", "#ef4444", "#22c55e", "#eab308",
    "#d946ef", "#38bdf8", "#fb923c", "#4ade80", "#facc15",
]


def _get_gemini_client(api_key: Optional[str] = None):
    """Initialize Google GenAI client using provided or environment API key."""
    key = api_key or os.environ.get("GEMINI_API_KEY")
    if not key:
        logger.warning("GEMINI_API_KEY is not configured in environment.")
        return None
    try:
        from google import genai
        return genai.Client(api_key=key)
    except Exception as e:
        logger.error(f"Failed to initialize google-genai client: {e}")
        return None


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

    client = _get_gemini_client(api_key)
    if client is None:
        # Fallback to the raw prompt if Gemini client is not configured
        return user_prompt.strip()

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

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[prompt_text, img_preview],
            config={
                "system_instruction": sys_inst,
                "temperature": 0.1,
            },
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

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=user_content,
            config={
                "system_instruction": sys_inst,
                "temperature": 0.2,
                "response_mime_type": "application/json",
            },
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
