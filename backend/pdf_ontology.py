"""
PDF Ontology Extraction Pipeline.

Extracts text and images from academic PDFs, then uses Gemini LLM to generate
a domain-specific ontology with visual prompts optimized for SAM3.

Usage:
    from pdf_ontology import extract_pdf_content, generate_ontology, save_ontology
"""

import io
import json
import logging
import os
import re
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

logger = logging.getLogger("sam3-backend")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ONTOLOGIES_DIR = Path(__file__).resolve().parent.parent / "datasets" / "ontologies"
PDF_IMAGES_DIR = Path(__file__).resolve().parent.parent / "datasets" / "pdf_images"

ONTOLOGIES_DIR.mkdir(parents=True, exist_ok=True)
PDF_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

# Palette for auto-assigned ontology colors (distinguishable, not too light)
DEFAULT_COLORS = [
    "#e11d48", "#8b5cf6", "#06b6d4", "#f59e0b", "#10b981",
    "#ec4899", "#6366f1", "#14b8a6", "#f97316", "#84cc16",
    "#a855f7", "#0ea5e9", "#ef4444", "#22c55e", "#eab308",
    "#d946ef", "#38bdf8", "#fb923c", "#4ade80", "#facc15",
]

# System prompt for ontology extraction — domain-agnostic, multi-scale with spatial rules
ONTOLOGY_SYSTEM_PROMPT = """\
You are an expert computational histopathologist and spatial ontology engineer.
Your task is to analyze academic histology, pathology, or microscopy texts (in any language, typically Spanish or English) and produce a structured, domain-agnostic Multi-Scale Histological & Spatial Ontology.

You must dissect the tissue into two complementary architectural scales and define the spatial topological rules between them:

1. MACRO STRUCTURES (Architectural landmarks & boundaries — targeted for SAM 3.1):
   - Compartments, tubules, follicles, acini, glands, vessels, layers, basement membranes, lumen cavities, interstitium/stroma.
   - SAM 3.1 excels at zero-shot boundary, cavity, and layer segmentation.
   - Each macro structure must include:
     * "key": snake_case identifier (e.g. "tubulo_seminifero", "luz_tubular", "membrana_basal", "espacio_intersticial", "foliculo_tiroideo", "coloide", "capsula_bowman", "espacio_urinario")
     * "name": Canonical Spanish name
     * "name_en": English name
     * "role": One of ["boundary_outer", "boundary_inner", "cavity", "compartment", "layer", "stroma"]
     * "target_engine": "sam3"
     * "prompt": Direct visual English prompt for SAM 3 (3-8 words, direct visual nouns: e.g. "circular seminiferous tubule cross section", "empty central lumen space cavity", "thin eosinophilic basement membrane ring")
     * "description": Short description of visual histological appearance.

2. MICRO STRUCTURES (Cells, nuclei, and micro-entities — targeted for Cellpose-SAM):
   - Specific cell types, nuclear morphology, or specialized micro-elements.
   - Cellpose excels at dense cell/nuclei boundary segmentation using topological gradient flows.
   - Each micro structure must include:
     * "key": snake_case identifier (e.g. "espermatogonia", "espermatocito_primario", "espermatozoide", "celula_leydig", "celula_sertoli", "podocito", "celula_folicular")
     * "name": Canonical Spanish name
     * "name_en": English name
     * "target_engine": "cellpose"
     * "prompt": Visual English prompt for morphology (e.g. "small round dark nucleus at basement membrane", "elongated condensed sperm head with flagellum in lumen")
     * "expected_diameter_px": Approximate nuclear/cellular diameter in standard 20x/40x microscopy (e.g. 15 to 40)
     * "spatial_rules": Explicit spatial constraints and anatomical distribution rules:
       - "compartment": The specific histological compartment (e.g. "basal", "adluminal", "luminal", "interstitial", "cortex", "medulla")
       - "parent_macro": The key of the parent macro structure it belongs to
       - "forbidden_in": Array of macro keys where this entity CANNOT physically exist (e.g. spermatogonia cannot be in ["luz_tubular", "espacio_intersticial"]; spermatozoa cannot be in ["membrana_basal", "espacio_intersticial"])
       - "relative_radial_position": Array of [min, max] where 0.0 is center/lumen and 1.0 is the outer basement membrane (e.g. [0.85, 1.0] for basal cells, [0.0, 0.35] for luminal cells, or null if non-radial)
       - "rule_description": Explicit validation rule in Spanish (e.g. "Debe situarse en la periferia adherida a la membrana basal; estrictamente prohibido en la luz tubular")

3. ROOT METADATA:
   - "tissue_name": The identified tissue or organ (e.g., "Testículo", "Tiroides", "Riñón", "Piel", "Hígado", etc.)
   - "summary": Brief 1-2 sentence overview of the structural architecture.

CRITICAL INSTRUCTIONS:
- Generate this structure for ANY tissue or organ described in the text (testis, kidney, liver, brain, thyroid, intestine, bone, etc.).
- NEVER make the system specific to only one organ; generalize based strictly on the provided text and figures.
- Return ONLY valid JSON adhering to the specified schema — no markdown backticks, no commentary.

EXAMPLE JSON OUTPUT:
{
  "tissue_name": "Testículo",
  "summary": "Estructura tubular con túbulos seminíferos rodeados de membrana basal e intersticio con células de Leydig.",
  "macro_structures": [
    {
      "key": "tubulo_seminifero",
      "name": "Túbulo seminífero",
      "name_en": "Seminiferous tubule",
      "role": "compartment",
      "target_engine": "sam3",
      "prompt": "circular seminiferous tubule cross section",
      "description": "Unidad funcional tubular delimitada por la lámina propia"
    },
    {
      "key": "membrana_basal",
      "name": "Membrana basal tubular",
      "name_en": "Basement membrane",
      "role": "boundary_outer",
      "target_engine": "sam3",
      "prompt": "thin basement membrane ring surrounding tubule",
      "description": "Lámina basal que delimita el epitelio seminífero del intersticio"
    },
    {
      "key": "luz_tubular",
      "name": "Luz del túbulo",
      "name_en": "Tubular lumen",
      "role": "cavity",
      "target_engine": "sam3",
      "prompt": "empty central tubular lumen cavity",
      "description": "Cavidad central donde se liberan los espermatozoides maduros"
    },
    {
      "key": "espacio_intersticial",
      "name": "Espacio intersticial",
      "name_en": "Interstitial stroma",
      "role": "stroma",
      "target_engine": "sam3",
      "prompt": "interstitial connective tissue between tubules",
      "description": "Tejido conectivo laxo con vasos sanguíneos y células endocrinas"
    }
  ],
  "micro_structures": [
    {
      "key": "espermatogonia",
      "name": "Espermatogonia",
      "name_en": "Spermatogonium",
      "target_engine": "cellpose",
      "prompt": "small dark round nucleus at basement membrane",
      "expected_diameter_px": 20,
      "spatial_rules": {
        "compartment": "basal",
        "parent_macro": "tubulo_seminifero",
        "forbidden_in": ["luz_tubular", "espacio_intersticial"],
        "relative_radial_position": [0.85, 1.0],
        "rule_description": "Ubicada exclusivamente apoyada en la membrana basal en la periferia tubular; nunca en la luz central"
      }
    },
    {
      "key": "espermatozoide",
      "name": "Espermatozoide",
      "name_en": "Spermatozoon",
      "target_engine": "cellpose",
      "prompt": "small dense elongated condensed nucleus with flagellum",
      "expected_diameter_px": 12,
      "spatial_rules": {
        "compartment": "luminal",
        "parent_macro": "luz_tubular",
        "forbidden_in": ["membrana_basal", "espacio_intersticial"],
        "relative_radial_position": [0.0, 0.3],
        "rule_description": "Ubicado exclusivamente en la luz central adluminal; prohibido en la membrana basal"
      }
    },
    {
      "key": "celula_leydig",
      "name": "Célula de Leydig",
      "name_en": "Leydig cell",
      "target_engine": "cellpose",
      "prompt": "large polygonal cell cluster in interstitial space",
      "expected_diameter_px": 28,
      "spatial_rules": {
        "compartment": "interstitial",
        "parent_macro": "espacio_intersticial",
        "forbidden_in": ["tubulo_seminifero", "luz_tubular"],
        "relative_radial_position": null,
        "rule_description": "Exclusiva del tejido conectivo intersticial fuera de los túbulos"
      }
    }
  ]
}
"""

ONTOLOGY_USER_PROMPT_TEMPLATE = """\
Analyze the following academic text and figures.
1. Identify the specific tissue/organ.
2. Extract the MACRO structures (anatomical compartments, boundaries, lumens, layers) optimized for SAM 3.1 open-vocabulary segmentation.
3. Extract the MICRO structures (cells, nuclei) optimized for Cellpose-SAM, and define explicit SPATIAL RULES (compartment, parent macro, forbidden zones, relative radial distribution) governing where each cell can and cannot physically appear.

TEXT:
{text}
"""


# ---------------------------------------------------------------------------
# PDF Content Extraction (pymupdf)
# ---------------------------------------------------------------------------

def extract_pdf_content(
    pdf_bytes: bytes,
    filename: str,
    min_image_size: int = 60,
    max_images: int = 50,
) -> Dict[str, Any]:
    """
    Extract text and embedded images from a PDF file.

    Args:
        pdf_bytes: Raw PDF file bytes.
        filename: Original filename for labeling.
        min_image_size: Minimum width/height in pixels to keep an image.
        max_images: Maximum number of images to extract.

    Returns:
        Dict with keys: text, pages, images (list of dicts), filename, pdf_id.
    """
    try:
        import fitz  # pymupdf
    except ImportError:
        raise ImportError(
            "pymupdf is required for PDF extraction. Install with: pip install pymupdf"
        )

    pdf_id = hashlib.md5(pdf_bytes[:4096] + filename.encode()).hexdigest()[:12]
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    all_text: List[str] = []
    page_texts: List[Dict[str, Any]] = []
    extracted_images: List[Dict[str, Any]] = []

    images_dir = PDF_IMAGES_DIR / pdf_id
    images_dir.mkdir(parents=True, exist_ok=True)

    seen_xrefs: set = set()
    seen_hashes: set = set()
    image_count = 0

    for page_num in range(len(doc)):
        page = doc[page_num]
        page_text = page.get_text("text")
        all_text.append(page_text)
        page_texts.append({"page": page_num + 1, "text": page_text})

        if image_count >= max_images:
            continue

        # Extract images from the page
        for img_index, img_info in enumerate(page.get_images(full=True)):
            if image_count >= max_images:
                break

            xref = img_info[0]
            if xref in seen_xrefs:
                # Already processed or skipped this xref from another page/reference
                continue

            try:
                saved_via_pil = False
                width, height = 0, 0
                raw_hash = None
                pixel_hash = None

                # Primary extraction: direct raw stream via doc.extract_image + PIL
                try:
                    base_image = doc.extract_image(xref)
                    if base_image and "image" in base_image:
                        raw_bytes = base_image["image"]
                        raw_hash = hashlib.sha256(raw_bytes).hexdigest()
                        if raw_hash in seen_hashes:
                            seen_xrefs.add(xref)
                            continue

                        pil_img = Image.open(io.BytesIO(raw_bytes))
                        width, height = pil_img.size

                        if width >= min_image_size and height >= min_image_size:
                            pixel_hash = hashlib.sha256(pil_img.tobytes()).hexdigest()
                            if pixel_hash in seen_hashes:
                                seen_xrefs.add(xref)
                                continue

                            if pil_img.mode != "RGB":
                                pil_img = pil_img.convert("RGB")
                            img_filename = f"{pdf_id}_p{page_num + 1}_img{img_index + 1}.png"
                            img_path = images_dir / img_filename
                            pil_img.save(img_path, format="PNG")
                            saved_via_pil = True
                            seen_hashes.add(raw_hash)
                            seen_hashes.add(pixel_hash)
                            seen_xrefs.add(xref)
                        else:
                            # Too small, skip and mark xref
                            seen_xrefs.add(xref)
                except Exception as extract_err:
                    logger.debug(f"doc.extract_image failed for xref={xref}: {extract_err}")

                # Secondary extraction: fallback to PyMuPDF Pixmap if raw stream failed
                if not saved_via_pil:
                    pix = fitz.Pixmap(doc, xref)

                    # Skip very small images (icons, decorations)
                    if pix.width < min_image_size or pix.height < min_image_size:
                        seen_xrefs.add(xref)
                        pix = None
                        continue

                    pix_hash = hashlib.sha256(pix.samples).hexdigest()
                    if pix_hash in seen_hashes:
                        seen_xrefs.add(xref)
                        pix = None
                        continue

                    # Convert CMYK / RGBA to RGB if necessary
                    if pix.n >= 4 or pix.alpha:
                        try:
                            pix = fitz.Pixmap(fitz.csRGB, pix)
                        except Exception as conv_err:
                            logger.warning(f"Error converting image xref={xref} to RGB: {conv_err}")

                    conv_hash = hashlib.sha256(pix.samples).hexdigest()
                    if conv_hash in seen_hashes:
                        seen_xrefs.add(xref)
                        pix = None
                        continue

                    img_filename = f"{pdf_id}_p{page_num + 1}_img{img_index + 1}.png"
                    img_path = images_dir / img_filename

                    pix.save(str(img_path))
                    width, height = pix.width, pix.height
                    seen_hashes.add(pix_hash)
                    seen_hashes.add(conv_hash)
                    seen_xrefs.add(xref)
                    pix = None

                # Try to find a caption near the image
                caption = _find_image_caption(page_text, page_num + 1, img_index)

                extracted_images.append({
                    "filename": img_filename,
                    "path": str(img_path),
                    "page": page_num + 1,
                    "width": width,
                    "height": height,
                    "caption": caption,
                    "pdf_id": pdf_id,
                })
                image_count += 1

            except Exception as e:
                seen_xrefs.add(xref)
                logger.warning(f"Failed to extract image xref={xref} from page {page_num + 1}: {e}")
                continue

    # FALLBACK: If no embedded raster images were found in the document,
    # render PDF pages as high-resolution images so the user has images to segment.
    if (image_count < min(len(doc), 2) or image_count == 0) and len(doc) > 0:
        logger.info(f"Rendering PDF pages as fallback/additional images for {filename} (embedded count was {image_count})...")
        max_page_renders = min(len(doc), 15)
        for page_num in range(max_page_renders):
            try:
                page = doc[page_num]
                pix = page.get_pixmap(dpi=150)

                if pix.n >= 4 or pix.alpha:
                    try:
                        pix = fitz.Pixmap(fitz.csRGB, pix)
                    except Exception as pix_err:
                        logger.warning(f"Failed to convert rendered page {page_num + 1} pixmap to csRGB: {pix_err}")

                img_filename = f"{pdf_id}_p{page_num + 1}_render.png"
                img_path = images_dir / img_filename
                pix.save(str(img_path))

                # Only add if not already in extracted_images
                if not any(im["filename"] == img_filename for im in extracted_images):
                    extracted_images.append({
                        "filename": img_filename,
                        "path": str(img_path),
                        "page": page_num + 1,
                        "width": pix.width,
                        "height": pix.height,
                        "caption": f"Página {page_num + 1} (Vista completa)",
                        "pdf_id": pdf_id,
                    })
                    image_count += 1
                pix = None
            except Exception as render_err:
                logger.warning(f"Failed to render page {page_num + 1} for {filename}: {render_err}")

    doc.close()

    full_text = "\n\n".join(all_text)

    # Persist extracted text and metadata to disk for fast retrieval & CRUD
    try:
        text_file = images_dir / "extracted_text.txt"
        with open(text_file, "w", encoding="utf-8") as f:
            f.write(full_text)

        metadata_file = images_dir / "metadata.json"
        metadata = {
            "pdf_id": pdf_id,
            "filename": filename,
            "total_pages": len(page_texts),
            "total_images": len(extracted_images),
            "text_length": len(full_text),
            "images": extracted_images,
        }
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.warning(f"Failed to persist PDF metadata/text for {pdf_id}: {e}")

    return {
        "pdf_id": pdf_id,
        "filename": filename,
        "text": full_text,
        "pages": page_texts,
        "images": extracted_images,
        "total_pages": len(page_texts),
        "total_images": len(extracted_images),
        "text_length": len(full_text),
    }


def _find_image_caption(page_text: str, page_num: int, img_index: int) -> Optional[str]:
    """Heuristic: try to find figure captions like 'Figura X', 'Fig. X', 'Figure X'."""
    patterns = [
        r"(?i)((?:figura|fig\.?|figure)\s*\d+[^.\n]{0,200}\.)",
        r"(?i)((?:imagen|image)\s*\d+[^.\n]{0,200}\.)",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, page_text)
        if matches and img_index < len(matches):
            return matches[img_index].strip()
    return None


# ---------------------------------------------------------------------------
# Ontology Generation (Gemini API)
# ---------------------------------------------------------------------------

def generate_ontology_with_gemini(
    extracted_text: str,
    api_key: Optional[str] = None,
    model_name: Optional[str] = None,
    max_text_chars: int = 35000,
    pdf_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Send extracted PDF text (and optional page images) to Gemini 3.5 Flash and get back a structured ontology.
    Uses automatic key rotation across GOOGLE_API_KEYS if api_key is None.

    Args:
        extracted_text: Full text extracted from PDF.
        api_key: Optional Gemini API key (defaults to key_manager rotation pool).
        model_name: Gemini model to use (defaults to GEMINI_MODEL / 'gemini-3.5-flash').
        max_text_chars: Max characters to send (optimized to 35,000 for fast responses).
        pdf_id: Optional PDF ID to load page images if text is very short/scanned.

    Returns:
        List of ontology structure dicts.
    """
    try:
        from backend.gemini_vision import generate_gemini_content, GEMINI_MODEL
    except ImportError:
        from gemini_vision import generate_gemini_content, GEMINI_MODEL

    effective_model = model_name or os.environ.get("GEMINI_MODEL", GEMINI_MODEL)

    # Truncate if very long
    text_for_llm = extracted_text[:max_text_chars]
    if len(extracted_text) > max_text_chars:
        logger.info(
            f"Text truncated from {len(extracted_text)} to {max_text_chars} chars for LLM"
        )

    user_prompt = ONTOLOGY_USER_PROMPT_TEMPLATE.format(
        text=text_for_llm if text_for_llm.strip() else "(Documento escaneado / sin texto extraído directamente. Analizar imágenes adjuntas.)"
    )

    contents: List[Any] = [user_prompt]

    # Multimodal: Send representative PDF images to Gemini alongside text.
    # This allows the LLM to see actual histological structures in figures
    # and generate more accurate visual prompts for SAM 3.
    if pdf_id:
        img_dir = PDF_IMAGES_DIR / pdf_id
        if img_dir.exists():
            img_files = sorted(
                list(img_dir.glob("*.png")) + list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.jpeg"))
            )
            # Filter non-image auxiliary files
            img_files = [f for f in img_files if f.name not in ("metadata.json", "extracted_text.txt")]
            # Sample up to 3 representative images to ensure fast latency with Gemini 3.5 Flash (<30s)
            max_gemini_images = 3
            if len(img_files) > max_gemini_images:
                step = len(img_files) / max_gemini_images
                selected_files = [img_files[int(i * step)] for i in range(max_gemini_images)]
            else:
                selected_files = img_files

            attached_count = 0
            for img_path in selected_files:
                try:
                    pil_im = Image.open(img_path)
                    if pil_im.mode != "RGB":
                        pil_im = pil_im.convert("RGB")
                    # Resize large images to 768px max dim for optimal speed/quality trade-off
                    max_dim = 768
                    if max(pil_im.size) > max_dim:
                        ratio = max_dim / max(pil_im.size)
                        new_size = (int(pil_im.width * ratio), int(pil_im.height * ratio))
                        pil_im = pil_im.resize(new_size, Image.LANCZOS)
                    contents.append(pil_im)
                    attached_count += 1
                    logger.info(f"Attached image {img_path.name} to Gemini multimodal prompt")
                except Exception as img_err:
                    logger.warning(f"Error loading image {img_path} for Gemini vision prompt: {img_err}")
            if attached_count > 0:
                logger.info(f"Attached {attached_count} images to Gemini prompt for pdf_id={pdf_id}")

    raw_text = None
    last_err = None

    # Attempt 1: Multimodal with attached images
    try:
        logger.info(f"Attempting Gemini ontology generation with model '{effective_model}' (multimodal)...")
        response = generate_gemini_content(
            contents=contents,
            system_instruction=ONTOLOGY_SYSTEM_PROMPT,
            temperature=0.2,
            response_mime_type="application/json",
            api_key=api_key or None,
            preferred_model=effective_model,
        )
        if response and response.text:
            raw_text = response.text.strip()
    except Exception as e:
        last_err = e
        logger.warning(f"Gemini multimodal generation failed on '{effective_model}': {e}")

    # Attempt 2: Text-only payload if multimodal failed
    if not raw_text:
        try:
            logger.info(f"Attempting Gemini ontology generation with model '{effective_model}' (text-only)...")
            response = generate_gemini_content(
                contents=[user_prompt],
                system_instruction=ONTOLOGY_SYSTEM_PROMPT,
                temperature=0.2,
                response_mime_type="application/json",
                api_key=api_key or None,
                preferred_model=effective_model,
            )
            if response and response.text:
                raw_text = response.text.strip()
        except Exception as e:
            last_err = e
            logger.warning(f"Gemini text-only generation failed on '{effective_model}': {e}")

    structures = []
    detected_tissue = "Tejido Histológico"
    if raw_text:
        # Parse JSON — handle potential markdown fences
        if raw_text.startswith("```"):
            raw_text = re.sub(r"^```(?:json)?\s*", "", raw_text)
            raw_text = re.sub(r"\s*```$", "", raw_text)

        try:
            parsed = json.loads(raw_text)
            if isinstance(parsed, dict):
                detected_tissue = parsed.get("tissue_name", "Tejido Histológico")
                macro_items = parsed.get("macro_structures", [])
                micro_items = parsed.get("micro_structures", [])

                for s in macro_items:
                    s["is_macro"] = True
                    s.setdefault("target_engine", "sam3")
                    s.setdefault("role", "compartment")
                    s.setdefault("spatial_rules", {})
                    s["tissue_name"] = detected_tissue

                for s in micro_items:
                    s["is_macro"] = False
                    s.setdefault("target_engine", "cellpose")
                    s.setdefault("role", "cell")
                    if "spatial_rules" not in s or not isinstance(s["spatial_rules"], dict):
                        s["spatial_rules"] = {}
                    s["tissue_name"] = detected_tissue

                structures = macro_items + micro_items

            elif isinstance(parsed, list):
                for s in parsed:
                    is_macro = s.get("is_macro")
                    if is_macro is None:
                        key_lower = (str(s.get("key", "")) + " " + str(s.get("name", "")) + " " + str(s.get("prompt", ""))).lower()
                        is_macro = any(w in key_lower for w in ["tubulo", "tubule", "lumen", "luz", "membrana", "membrane", "intersticio", "stroma", "capsula", "layer", "capa", "corteza", "medula", "foliculo", "acino"])
                    s["is_macro"] = bool(is_macro)
                    s.setdefault("target_engine", "sam3" if is_macro else "cellpose")
                    s.setdefault("spatial_rules", {})
                    s["tissue_name"] = detected_tissue
                    structures.append(s)
        except Exception as parse_err:
            logger.warning(f"Failed to parse LLM JSON: {parse_err}. Raw text: {raw_text[:200]}")

    # Attempt 3: Heuristic extraction if LLM returned empty or failed
    if not structures:
        logger.warning(f"LLM ontology generation returned no structures (last err: {last_err}). Using heuristic multi-scale extraction.")
        structures = [
            # Macro structures (SAM 3.1)
            {
                "key": "luz_tubular",
                "name": "Luz tubular o cavidad",
                "name_en": "Tubular lumen cavity",
                "prompt": "empty tubular lumen cavity central space",
                "color": "#06b6d4",
                "is_macro": True,
                "role": "cavity",
                "target_engine": "sam3",
                "spatial_rules": {},
                "tissue_name": "Tejido Histológico General"
            },
            {
                "key": "membrana_basal",
                "name": "Membrana basal o límite estructural",
                "name_en": "Basement membrane boundary",
                "prompt": "structural basement membrane boundary ring",
                "color": "#8b5cf6",
                "is_macro": True,
                "role": "boundary_outer",
                "target_engine": "sam3",
                "spatial_rules": {},
                "tissue_name": "Tejido Histológico General"
            },
            {
                "key": "espacio_intersticial",
                "name": "Estroma o tejido intersticial",
                "name_en": "Interstitial stroma",
                "prompt": "interstitial connective tissue fiber stroma",
                "color": "#10b981",
                "is_macro": True,
                "role": "stroma",
                "target_engine": "sam3",
                "spatial_rules": {},
                "tissue_name": "Tejido Histológico General"
            },
            # Micro structures (Cellpose-SAM + Spatial Rules)
            {
                "key": "celula_basal",
                "name": "Célula basal (periferia)",
                "name_en": "Basal cell nucleus",
                "prompt": "round dark cell nucleus at basement membrane",
                "color": "#e11d48",
                "is_macro": False,
                "role": "cell",
                "target_engine": "cellpose",
                "expected_diameter_px": 20,
                "spatial_rules": {
                    "compartment": "basal",
                    "parent_macro": "membrana_basal",
                    "forbidden_in": ["luz_tubular", "espacio_intersticial"],
                    "relative_radial_position": [0.85, 1.0],
                    "rule_description": "Ubicada adyacente a la membrana basal; estrictamente prohibida en la luz tubular o intersticio"
                },
                "tissue_name": "Tejido Histológico General"
            },
            {
                "key": "elemento_adluminal",
                "name": "Elemento adluminal / celular apical",
                "name_en": "Adluminal cell element",
                "prompt": "differentiated cellular element near lumen",
                "color": "#f59e0b",
                "is_macro": False,
                "role": "cell",
                "target_engine": "cellpose",
                "expected_diameter_px": 14,
                "spatial_rules": {
                    "compartment": "luminal",
                    "parent_macro": "luz_tubular",
                    "forbidden_in": ["membrana_basal"],
                    "relative_radial_position": [0.0, 0.4],
                    "rule_description": "Ubicado en el polo apical o luz; prohibido en la lámina o membrana basal"
                },
                "tissue_name": "Tejido Histológico General"
            },
            {
                "key": "celula_intersticial",
                "name": "Célula intersticial / estromal",
                "name_en": "Interstitial stroma cell",
                "prompt": "spindle or polygonal interstitial cell nucleus",
                "color": "#ec4899",
                "is_macro": False,
                "role": "cell",
                "target_engine": "cellpose",
                "expected_diameter_px": 26,
                "spatial_rules": {
                    "compartment": "interstitial",
                    "parent_macro": "espacio_intersticial",
                    "forbidden_in": ["luz_tubular"],
                    "relative_radial_position": None,
                    "rule_description": "Exclusiva del estroma exterior conectivo; prohibida en cavidades luminales"
                },
                "tissue_name": "Tejido Histológico General"
            },
        ]

    # Assign colors and labels if missing
    for i, struct in enumerate(structures):
        if "color" not in struct:
            struct["color"] = DEFAULT_COLORS[i % len(DEFAULT_COLORS)]
        # Ensure label field (use name)
        if "label" not in struct:
            struct["label"] = struct.get("name", struct.get("key", f"Clase {i + 1}"))

    logger.info(f"Generated ontology with {len(structures)} structures for tissue '{detected_tissue}'")
    return structures


HISTOLOGY_KEYWORDS = {
    # Spanish
    "histo", "histologia", "histología", "histopatologia", "histopatología",
    "tejido", "tejidos", "célula", "celula", "células", "celulas", "núcleo", "nucleo",
    "núcleos", "nucleos", "citoplasma", "microscop", "microscopio", "microscopia",
    "microscopía", "tinción", "tincion", "tinciones", "biopsia", "patología", "patologia",
    "epitelio", "estroma", "túbulo", "tubulo", "túbulos", "tubulos", "fibra", "fibras",
    "glándula", "glandula", "glándulas", "glandulas", "lumen", "membrana", "arteria",
    "vena", "capilar", "endotelio", "miocito", "neurona", "linfocito", "macrófago",
    "macrofago", "hematoxilina", "eosina", "h&e", "intersticio", "basal", "acino",
    "corte", "frotis", "lámina", "lamina", "portaobjeto", "endotelial",
    # English
    "histology", "histopathology", "tissue", "tissues", "cell", "cells", "cellular",
    "nucleus", "nuclei", "cytoplasm", "microscopy", "microscopic", "microscope",
    "stain", "staining", "biopsy", "pathology", "epithelium", "epithelial",
    "stroma", "stromal", "tubule", "tubules", "fiber", "fibers", "gland", "glands",
    "artery", "vein", "capillary", "endothelium", "myocyte", "neuron", "lymphocyte",
    "macrophage", "hematoxylin", "eosin", "interstitial", "basement", "acini", "acinar",
    "slide", "section", "smear", "endothelial"
}


def is_histology_ontology(ontology: Optional[Dict[str, Any]]) -> bool:
    """
    Determine if an ontology is related to histology/microscopy.
    Returns True if histological features/keywords are detected or explicitly flagged,
    False if the ontology is for non-histological domains.
    """
    if not ontology or not isinstance(ontology, dict):
        return False

    # Explicit flag takes priority if set
    if "is_histology" in ontology and isinstance(ontology["is_histology"], bool):
        return ontology["is_histology"]

    # Gather texts to search across
    domain = str(ontology.get("domain", "")).lower()
    source_pdf = str(ontology.get("source_pdf", "")).lower()

    combined_text_parts = [domain, source_pdf]

    structures = ontology.get("structures", []) or ontology.get("prompts", [])
    for s in structures:
        if isinstance(s, dict):
            combined_text_parts.append(str(s.get("key", "")).lower())
            combined_text_parts.append(str(s.get("name", "")).lower())
            combined_text_parts.append(str(s.get("name_en", "")).lower())
            combined_text_parts.append(str(s.get("label", "")).lower())
            combined_text_parts.append(str(s.get("prompt", "")).lower())
            combined_text_parts.append(str(s.get("description", "")).lower())

    full_text = " ".join(combined_text_parts)

    # Check for keyword matches
    for kw in HISTOLOGY_KEYWORDS:
        if kw in full_text:
            return True

    return False


# ---------------------------------------------------------------------------
# Ontology Storage
# ---------------------------------------------------------------------------

def build_ontology_document(
    pdf_id: str,
    filename: str,
    structures: List[Dict[str, Any]],
    extracted_images: List[Dict[str, Any]],
    domain_name: Optional[str] = None,
    tissue_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a complete ontology document for storage with dual-scale macro/micro and spatial rules."""
    if domain_name is None:
        # Derive from filename
        base = Path(filename).stem
        domain_name = re.sub(r"[^a-zA-Z0-9_]", "_", base).lower()

    detected_tissue = tissue_name
    if not detected_tissue and structures:
        for s in structures:
            if s.get("tissue_name"):
                detected_tissue = s["tissue_name"]
                break

    macro_structures = [s for s in structures if s.get("is_macro")]
    micro_structures = [s for s in structures if not s.get("is_macro")]

    doc = {
        "domain": domain_name,
        "tissue_name": detected_tissue or "Tejido Histológico General",
        "source_pdf": filename,
        "pdf_id": pdf_id,
        "macro_structures": macro_structures,
        "micro_structures": micro_structures,
        "spatial_rules": [
            {
                "micro_key": s.get("key"),
                "micro_name": s.get("name") or s.get("label"),
                **s.get("spatial_rules", {})
            }
            for s in micro_structures if s.get("spatial_rules")
        ],
        "structures": structures,
        "extracted_images": extracted_images,
        "prompts": [
            {
                "key": s.get("key", f"class_{i + 1}"),
                "prompt": s.get("prompt", s.get("name", s.get("key", f"structure {i + 1}"))),
                "label": s.get("label", s.get("name", s.get("key", f"Clase {i + 1}"))),
                "color": s.get("color", DEFAULT_COLORS[i % len(DEFAULT_COLORS)]),
                "is_macro": s.get("is_macro", False),
                "target_engine": s.get("target_engine", "sam3" if s.get("is_macro") else "cellpose"),
                "spatial_rules": s.get("spatial_rules", {}),
            }
            for i, s in enumerate(structures)
        ],
    }
    doc["is_histology"] = is_histology_ontology(doc)
    return doc


def save_ontology(ontology: Dict[str, Any]) -> str:
    """Save ontology JSON to disk. Returns the file path."""
    ONTOLOGIES_DIR.mkdir(parents=True, exist_ok=True)
    domain = ontology.get("domain", "unnamed")
    filepath = ONTOLOGIES_DIR / f"{domain}.json"
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(ontology, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved ontology to {filepath}")
    return str(filepath)


def merge_ontology_structures(
    existing_ontology: Dict[str, Any],
    new_structures: List[Dict[str, Any]],
    new_pdf_id: str,
    new_filename: str,
    new_images: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Incrementally merge new structures into an existing ontology.

    - Structures with the same ``key`` are updated (new prompt/name/rules win).
    - Structures with new keys are appended.
    - ``source_pdfs`` accumulates all PDF sources.
    - ``extracted_images`` from the new PDF are appended (deduped by filename).
    - Preserves macro_structures, micro_structures, and spatial_rules.
    """
    # Build a lookup of existing structures by key
    existing_by_key: Dict[str, Dict[str, Any]] = {
        s["key"]: s for s in existing_ontology.get("structures", [])
    }

    added_count = 0
    updated_count = 0

    for ns in new_structures:
        key = ns["key"]
        if key in existing_by_key:
            # Merge: update prompt/name/label/color from new, keep parent if set
            existing_by_key[key]["prompt"] = ns.get("prompt", existing_by_key[key].get("prompt"))
            existing_by_key[key]["name"] = ns.get("name", existing_by_key[key].get("name"))
            existing_by_key[key]["name_en"] = ns.get("name_en", existing_by_key[key].get("name_en"))
            existing_by_key[key]["label"] = ns.get("label", ns.get("name", existing_by_key[key].get("label")))
            if "parent" in ns:
                existing_by_key[key]["parent"] = ns["parent"]
            if "is_macro" in ns:
                existing_by_key[key]["is_macro"] = ns["is_macro"]
            if "target_engine" in ns:
                existing_by_key[key]["target_engine"] = ns["target_engine"]
            if "role" in ns:
                existing_by_key[key]["role"] = ns["role"]
            if "spatial_rules" in ns:
                existing_by_key[key]["spatial_rules"] = ns["spatial_rules"]
            if "expected_diameter_px" in ns:
                existing_by_key[key]["expected_diameter_px"] = ns["expected_diameter_px"]
            updated_count += 1
        else:
            # Assign a new color from the palette
            color_idx = len(existing_by_key)
            if "color" not in ns:
                ns["color"] = DEFAULT_COLORS[color_idx % len(DEFAULT_COLORS)]
            if "label" not in ns:
                ns["label"] = ns.get("name", ns.get("key", f"Clase {color_idx + 1}"))
            existing_by_key[key] = ns
            added_count += 1

    merged_structures = list(existing_by_key.values())

    # Accumulate source PDFs
    source_pdfs: List[Dict[str, str]] = existing_ontology.get("source_pdfs", [])
    # Migrate legacy single source_pdf field
    if not source_pdfs and existing_ontology.get("source_pdf"):
        source_pdfs.append({
            "pdf_id": existing_ontology.get("pdf_id", "unknown"),
            "filename": existing_ontology["source_pdf"],
        })
    # Add new source if not already present
    if not any(sp.get("pdf_id") == new_pdf_id for sp in source_pdfs):
        source_pdfs.append({"pdf_id": new_pdf_id, "filename": new_filename})

    # Merge images (deduplicate by filename)
    existing_images = existing_ontology.get("extracted_images", [])
    existing_img_filenames = {im.get("filename") for im in existing_images}
    for img in (new_images or []):
        if img.get("filename") not in existing_img_filenames:
            img_copy = dict(img)
            if "pdf_id" not in img_copy or not img_copy["pdf_id"]:
                img_copy["pdf_id"] = new_pdf_id
            existing_images.append(img_copy)
            existing_img_filenames.add(img.get("filename"))

    # Rebuild the ontology document
    existing_ontology["structures"] = merged_structures
    existing_ontology["macro_structures"] = [s for s in merged_structures if s.get("is_macro")]
    existing_ontology["micro_structures"] = [s for s in merged_structures if not s.get("is_macro")]
    existing_ontology["spatial_rules"] = [
        {
            "micro_key": s.get("key"),
            "micro_name": s.get("name") or s.get("label"),
            **s.get("spatial_rules", {})
        }
        for s in merged_structures if not s.get("is_macro") and s.get("spatial_rules")
    ]
    if "tissue_name" not in existing_ontology or not existing_ontology["tissue_name"]:
        for s in merged_structures:
            if s.get("tissue_name"):
                existing_ontology["tissue_name"] = s["tissue_name"]
                break

    existing_ontology["source_pdfs"] = source_pdfs
    existing_ontology["extracted_images"] = existing_images
    # Keep the legacy source_pdf pointing to the latest
    existing_ontology["source_pdf"] = new_filename
    existing_ontology["pdf_id"] = new_pdf_id

    # Regenerate prompts
    existing_ontology["prompts"] = [
        {
            "key": s.get("key", f"class_{i + 1}"),
            "prompt": s.get("prompt", s.get("name", s.get("key", f"structure {i + 1}"))),
            "label": s.get("label", s.get("name", s.get("key", f"Clase {i + 1}"))),
            "color": s.get("color", DEFAULT_COLORS[i % len(DEFAULT_COLORS)]),
            "is_macro": s.get("is_macro", False),
            "target_engine": s.get("target_engine", "sam3" if s.get("is_macro") else "cellpose"),
            "spatial_rules": s.get("spatial_rules", {}),
        }
        for i, s in enumerate(merged_structures)
    ]
    existing_ontology["is_histology"] = is_histology_ontology(existing_ontology)

    logger.info(
        f"Merge complete: {added_count} new + {updated_count} updated = "
        f"{len(merged_structures)} total structures from {len(source_pdfs)} PDFs, "
        f"{len(existing_images)} total images"
    )
    return existing_ontology


def load_ontology(name: str) -> Optional[Dict[str, Any]]:
    """Load an ontology by domain name."""
    filepath = ONTOLOGIES_DIR / f"{name}.json"
    if not filepath.exists():
        return None
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "is_histology" not in data:
        data["is_histology"] = is_histology_ontology(data)
    return data


def list_ontologies() -> List[Dict[str, Any]]:
    """List all saved ontologies with summary info."""
    ONTOLOGIES_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    for filepath in sorted(ONTOLOGIES_DIR.glob("*.json")):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            results.append({
                "name": filepath.stem,
                "domain": data.get("domain", filepath.stem),
                "source_pdf": data.get("source_pdf", "unknown"),
                "is_histology": is_histology_ontology(data),
                "num_structures": len(data.get("structures", [])),
                "num_images": len(data.get("extracted_images", [])),
                "num_prompts": len(data.get("prompts", [])),
            })
        except Exception as e:
            logger.warning(f"Failed to read ontology {filepath}: {e}")
    return results


def update_ontology_structures(
    name: str, structures: List[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    """Update the structures (and regenerate prompts and spatial rules) of a saved ontology."""
    ontology = load_ontology(name)
    if ontology is None:
        return None

    ontology["structures"] = structures
    ontology["macro_structures"] = [s for s in structures if s.get("is_macro")]
    ontology["micro_structures"] = [s for s in structures if not s.get("is_macro")]
    ontology["spatial_rules"] = [
        {
            "micro_key": s.get("key"),
            "micro_name": s.get("name") or s.get("label"),
            **s.get("spatial_rules", {})
        }
        for s in structures if not s.get("is_macro") and s.get("spatial_rules")
    ]
    ontology["is_histology"] = is_histology_ontology(ontology)

    # Regenerate prompts from updated structures
    ontology["prompts"] = [
        {
            "key": s.get("key", f"class_{i + 1}"),
            "prompt": s.get("prompt", s.get("name", s.get("key", f"structure {i + 1}"))),
            "label": s.get("label", s.get("name", s.get("key", f"Clase {i + 1}"))),
            "color": s.get("color", DEFAULT_COLORS[i % len(DEFAULT_COLORS)]),
            "is_macro": s.get("is_macro", False),
            "target_engine": s.get("target_engine", "sam3" if s.get("is_macro") else "cellpose"),
            "spatial_rules": s.get("spatial_rules", {}),
        }
        for i, s in enumerate(structures)
    ]

    save_ontology(ontology)
    return ontology


def get_ontology_prompts(name: str) -> Optional[List[Dict[str, str]]]:
    """
    Get the prompts list from an ontology, formatted for SAM3 AUTO_SEGMENT_PROMPTS.
    Returns list of dicts with keys: key, prompt, label, color.
    """
    ontology = load_ontology(name)
    if ontology is None:
        return None
    return ontology.get("prompts", [])


def get_pdf_image_path(pdf_id: str, filename: str) -> Optional[str]:
    """Get the absolute path to an extracted PDF image, resolving across merged PDF subdirectories."""
    # 1. Direct path in requested pdf_id
    if pdf_id and pdf_id != "unknown":
        path = PDF_IMAGES_DIR / pdf_id / filename
        if path.exists():
            return str(path)

    # 2. Extract actual pdf_id from filename prefix if filename format is {real_pdf_id}_...
    if "_" in filename:
        prefix = filename.split("_")[0]
        if prefix and prefix != pdf_id:
            path = PDF_IMAGES_DIR / prefix / filename
            if path.exists():
                return str(path)

    # 3. Fallback: Search across all subdirectories of PDF_IMAGES_DIR
    if PDF_IMAGES_DIR.exists():
        for sub_dir in PDF_IMAGES_DIR.iterdir():
            if sub_dir.is_dir():
                cand = sub_dir / filename
                if cand.exists():
                    return str(cand)

    return None


def get_extracted_text(pdf_id: str) -> Optional[str]:
    """Get the full extracted text for a given pdf_id from disk."""
    text_file = PDF_IMAGES_DIR / pdf_id / "extracted_text.txt"
    if text_file.exists():
        with open(text_file, "r", encoding="utf-8") as f:
            return f.read()
    return None


def get_pdf_metadata(pdf_id: str) -> Optional[Dict[str, Any]]:
    """Load metadata.json for a specific pdf_id, with fallback to saved ontologies or disk scanning."""
    meta_file = PDF_IMAGES_DIR / pdf_id / "metadata.json"
    if meta_file.exists():
        try:
            with open(meta_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Error loading metadata for {pdf_id}: {e}")

    # Fallback 1: Check if any saved ontology references this pdf_id
    if ONTOLOGIES_DIR.exists():
        for f in ONTOLOGIES_DIR.glob("*.json"):
            try:
                with open(f, "r", encoding="utf-8") as ont_f:
                    ont_data = json.load(ont_f)
                    if ont_data.get("pdf_id") == pdf_id and "extracted_images" in ont_data:
                        meta = {
                            "pdf_id": pdf_id,
                            "filename": ont_data.get("source_pdf", f"{pdf_id}.pdf"),
                            "total_images": len(ont_data["extracted_images"]),
                            "images": ont_data["extracted_images"],
                        }
                        save_pdf_metadata(pdf_id, meta)
                        return meta
            except Exception as e:
                logger.warning(f"Error reading ontology {f}: {e}")

    # Fallback 2: Scan images directory directly
    img_dir = PDF_IMAGES_DIR / pdf_id
    if img_dir.exists() and img_dir.is_dir():
        image_files = sorted(list(img_dir.glob("*.png")) + list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.jpeg")))
        if image_files:
            images = []
            for img_path in image_files:
                if img_path.name == "metadata.json" or img_path.name == "extracted_text.txt":
                    continue
                try:
                    with Image.open(img_path) as im:
                        w, h = im.size
                except Exception:
                    w, h = 0, 0
                
                # Extract page number if in format {pdf_id}_p{page}_img{n}
                page_match = re.search(r"_p(\d+)_", img_path.name)
                page_num = int(page_match.group(1)) if page_match else 1

                images.append({
                    "filename": img_path.name,
                    "path": str(img_path.resolve()),
                    "page": page_num,
                    "width": w,
                    "height": h,
                    "caption": None,
                })

            meta = {
                "pdf_id": pdf_id,
                "filename": f"{pdf_id}.pdf",
                "total_images": len(images),
                "images": images,
            }
            save_pdf_metadata(pdf_id, meta)
            return meta

    return None


def save_pdf_metadata(pdf_id: str, metadata: Dict[str, Any]) -> None:
    """Save updated metadata.json for a specific pdf_id."""
    meta_file = PDF_IMAGES_DIR / pdf_id / "metadata.json"
    meta_file.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def add_pdf_image(
    pdf_id: str,
    image_bytes: bytes,
    original_filename: str,
    caption: Optional[str] = None
) -> Dict[str, Any]:
    """
    Add a new image to a PDF's image collection (CRUD: Create).
    """
    images_dir = PDF_IMAGES_DIR / pdf_id
    images_dir.mkdir(parents=True, exist_ok=True)

    # Open image to verify and get dimensions
    img = Image.open(io.BytesIO(image_bytes))
    if img.mode != "RGB":
        img = img.convert("RGB")

    metadata = get_pdf_metadata(pdf_id) or {
        "pdf_id": pdf_id,
        "filename": "custom_dataset",
        "total_pages": 1,
        "total_images": 0,
        "text_length": 0,
        "images": [],
    }

    clean_name = re.sub(r"[^a-zA-Z0-9_.-]", "_", original_filename)
    stored_filename = f"{pdf_id}_custom_{len(metadata.get('images', [])) + 1}_{clean_name}"
    if not stored_filename.lower().endswith((".png", ".jpg", ".jpeg")):
        stored_filename += ".png"

    dest_path = images_dir / stored_filename
    img.save(dest_path, format="PNG")

    img_info = {
        "filename": stored_filename,
        "path": str(dest_path),
        "page": "Custom / Upload",
        "width": img.width,
        "height": img.height,
        "caption": caption or f"Imagen agregada manualmente: {original_filename}",
        "custom": True,
    }

    # Avoid duplicating if already present
    existing_filenames = {im["filename"] for im in metadata.get("images", [])}
    if stored_filename not in existing_filenames:
        metadata.setdefault("images", []).append(img_info)
    metadata["total_images"] = len(metadata["images"])
    save_pdf_metadata(pdf_id, metadata)

    return img_info


def update_pdf_image_metadata(
    pdf_id: str,
    filename: str,
    caption: Optional[str] = None,
    label: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    Update caption/label of an extracted PDF image (CRUD: Update).
    """
    metadata = get_pdf_metadata(pdf_id)
    if not metadata or "images" not in metadata:
        return None

    target_img = None
    for img in metadata["images"]:
        if img["filename"] == filename:
            if caption is not None:
                img["caption"] = caption
            if label is not None:
                img["label"] = label
            target_img = img
            break

    if target_img:
        save_pdf_metadata(pdf_id, metadata)

    return target_img


def delete_pdf_image(pdf_id: str, filename: str) -> bool:
    """
    Delete an extracted PDF image from disk and metadata (CRUD: Delete).
    """
    images_dir = PDF_IMAGES_DIR / pdf_id
    img_path = images_dir / filename
    if img_path.exists():
        try:
            img_path.unlink()
        except Exception as e:
            logger.warning(f"Failed to delete image file {img_path}: {e}")

    metadata = get_pdf_metadata(pdf_id)
    if metadata and "images" in metadata:
        initial_len = len(metadata["images"])
        metadata["images"] = [img for img in metadata["images"] if img["filename"] != filename]
        if len(metadata["images"]) < initial_len:
            metadata["total_images"] = len(metadata["images"])
            save_pdf_metadata(pdf_id, metadata)
            return True

    return False


def validate_spatial_rules(
    detections: List[Dict[str, Any]],
    macro_annotations: List[Dict[str, Any]],
    spatial_rules_lookup: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Validate cell/micro detections against segmented macro-structure boundaries.

    Args:
        detections: List of cell detections, each with 'box' [x1, y1, x2, y2] or 'bbox' or 'centroid'
                    and 'class_key' / 'category_id' / 'label'.
        macro_annotations: List of macro detections with 'class_key' / 'label' and 'segmentation' (polygon points).
        spatial_rules_lookup: Dict mapping micro class_key -> spatial_rules dict.

    Returns:
        Summary dict with validated_count, violations_count, and detailed violations list.
    """
    import cv2
    import numpy as np

    # Build spatial index of macro polygons
    macro_polys_by_key: Dict[str, List[np.ndarray]] = {}
    for macro in macro_annotations:
        m_key = str(macro.get("class_key") or macro.get("category_id") or macro.get("key") or macro.get("label") or "").strip().lower()
        if not m_key:
            continue
        segs = macro.get("segmentation") or []
        if isinstance(segs, list):
            for poly in segs:
                if isinstance(poly, list) and len(poly) >= 6:
                    pts = np.array(poly, dtype=np.float32).reshape(-1, 2)
                    macro_polys_by_key.setdefault(m_key, []).append(pts)

    violations = []
    valid_count = 0

    for idx, det in enumerate(detections):
        d_key = str(det.get("class_key") or det.get("category_id") or det.get("key") or det.get("label") or "").strip().lower()
        rule = spatial_rules_lookup.get(d_key) or {}
        if not rule:
            # Check by matching substring
            for rk, rv in spatial_rules_lookup.items():
                if rk in d_key or d_key in rk:
                    rule = rv
                    break

        forbidden = [str(k).strip().lower() for k in rule.get("forbidden_in", [])]

        # Compute centroid
        if "centroid" in det and isinstance(det["centroid"], (list, tuple)) and len(det["centroid"]) == 2:
            cx, cy = float(det["centroid"][0]), float(det["centroid"][1])
        elif "box" in det and isinstance(det["box"], (list, tuple)) and len(det["box"]) == 4:
            x1, y1, x2, y2 = [float(v) for v in det["box"]]
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        elif "bbox" in det and isinstance(det["bbox"], (list, tuple)) and len(det["bbox"]) == 4:
            x, y, w, h = [float(v) for v in det["bbox"]]
            cx, cy = x + w / 2.0, y + h / 2.0
        else:
            valid_count += 1
            continue

        det_violation = None

        # Check forbidden regions: cell centroid inside forbidden macro polygon
        for forb_key in forbidden:
            if forb_key in macro_polys_by_key:
                for poly_pts in macro_polys_by_key[forb_key]:
                    dist = cv2.pointPolygonTest(poly_pts, (cx, cy), False)
                    if dist >= 0:
                        det_violation = {
                            "detection_index": idx,
                            "class_key": d_key,
                            "label": det.get("label", d_key),
                            "centroid": [round(cx, 1), round(cy, 1)],
                            "violation_type": "forbidden_compartment",
                            "forbidden_macro": forb_key,
                            "reason": f"Célula '{det.get('label', d_key)}' detectada dentro de la macroestructura prohibida '{forb_key}' ({rule.get('rule_description', 'violación anatómica')})"
                        }
                        break
            if det_violation:
                break

        if det_violation:
            violations.append(det_violation)
        else:
            valid_count += 1

    return {
        "total_evaluated": len(detections),
        "valid_count": valid_count,
        "violations_count": len(violations),
        "violations": violations,
    }


def derive_spatial_map_and_rules(
    ontology_doc: Optional[Dict[str, Any]]
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, List[str]], Dict[str, List[str]]]:
    """
    Parses ontology structures and spatial rules to build comprehensive topological lookup maps:
    1. spatial_rules_lookup: micro_key -> {compartment, parent_macro, forbidden_in, relative_radial_position, rule_description, name, color}
    2. spatial_map: compartment/macro -> list of allowed micro_keys
    3. forbidden_map: compartment/macro -> list of strictly forbidden micro_keys

    Includes biological fallbacks for testicular and general tissue architecture if partial ontology.
    """
    spatial_rules_lookup: Dict[str, Dict[str, Any]] = {}
    spatial_map: Dict[str, List[str]] = {}
    forbidden_map: Dict[str, List[str]] = {}

    if not ontology_doc:
        return _apply_testicular_fallbacks(spatial_rules_lookup, spatial_map, forbidden_map)

    # 1. Parse root-level spatial_rules list if present
    raw_rules = ontology_doc.get("spatial_rules", [])
    if isinstance(raw_rules, list):
        for r in raw_rules:
            if not isinstance(r, dict):
                continue
            key = str(r.get("micro_key") or r.get("key") or "").strip().lower()
            if key:
                spatial_rules_lookup[key] = {
                    "compartment": str(r.get("compartment", "")).strip().lower(),
                    "parent_macro": str(r.get("parent_macro", "")).strip().lower(),
                    "forbidden_in": [str(x).strip().lower() for x in r.get("forbidden_in", []) if x],
                    "relative_radial_position": r.get("relative_radial_position"),
                    "rule_description": str(r.get("rule_description", "")).strip(),
                    "name": str(r.get("micro_name") or r.get("name", key)).strip(),
                }

    # 2. Parse micro_structures and general structures
    structures_pool = []
    if isinstance(ontology_doc.get("micro_structures"), list):
        structures_pool.extend(ontology_doc["micro_structures"])
    if isinstance(ontology_doc.get("structures"), list):
        structures_pool.extend([s for s in ontology_doc["structures"] if not s.get("is_macro")])

    for s in structures_pool:
        if not isinstance(s, dict):
            continue
        key = str(s.get("key") or s.get("id") or "").strip().lower()
        if not key:
            continue

        s_rules = s.get("spatial_rules") or {}
        if not isinstance(s_rules, dict):
            s_rules = {}

        existing = spatial_rules_lookup.get(key, {})
        compartment = str(s_rules.get("compartment") or s.get("spatial_zone") or existing.get("compartment", "")).strip().lower()
        parent_macro = str(s_rules.get("parent_macro") or s.get("parent_compartment") or existing.get("parent_macro", "")).strip().lower()
        
        forbidden_raw = s_rules.get("forbidden_in") or existing.get("forbidden_in") or []
        forbidden_in = [str(x).strip().lower() for x in forbidden_raw if x]
        radial = s_rules.get("relative_radial_position") or existing.get("relative_radial_position")
        rule_desc = str(s_rules.get("rule_description") or existing.get("rule_description", "")).strip()

        spatial_rules_lookup[key] = {
            "compartment": compartment,
            "parent_macro": parent_macro,
            "forbidden_in": forbidden_in,
            "relative_radial_position": radial,
            "rule_description": rule_desc,
            "name": str(s.get("name") or s.get("label") or existing.get("name", key)),
            "color": str(s.get("color") or existing.get("color", "#8b5cf6")),
            "prompt": str(s.get("prompt") or s.get("cytological_features", "")),
        }

    # 3. Populate spatial_map (allowed) and forbidden_map
    for key, rule in spatial_rules_lookup.items():
        p_macro = rule.get("parent_macro", "")
        comp = rule.get("compartment", "")
        forb_list = rule.get("forbidden_in", [])

        # Allowed in parent macro
        if p_macro:
            spatial_map.setdefault(p_macro, [])
            if key not in spatial_map[p_macro]:
                spatial_map[p_macro].append(key)

        # Allowed in specific compartment
        if comp:
            spatial_map.setdefault(comp, [])
            if key not in spatial_map[comp]:
                spatial_map[comp].append(key)

        # Prohibited in forbidden_in macro/compartments
        for forb in forb_list:
            if forb:
                forbidden_map.setdefault(forb, [])
                if key not in forbidden_map[forb]:
                    forbidden_map[forb].append(key)

    # 4. Check if testicular fallbacks are needed to ensure complete protection
    return _apply_testicular_fallbacks(spatial_rules_lookup, spatial_map, forbidden_map)


def _apply_testicular_fallbacks(
    spatial_rules_lookup: Dict[str, Dict[str, Any]],
    spatial_map: Dict[str, List[str]],
    forbidden_map: Dict[str, List[str]],
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, List[str]], Dict[str, List[str]]]:
    """Ensures absolute architectural constraints for testicular tissues."""
    testis_germ_cells = [
        "espermatogonia_a_clara", "espermatogonia_a_oscura", "espermatogonia_b",
        "espermatocito_primario", "espermatocito_secundario",
        "espermatide_temprana", "espermatide_tardia", "espermatozoide",
        "celula_sertoli", "sertoli", "espermatogonia", "espermatocito", "espermatide"
    ]
    interstitial_cells = ["celula_leydig", "leydig", "celula_intersticial", "celula_peritubular"]

    # 1. Tubule must strictly forbid Leydig cells
    tubule_aliases = ["tubulo_seminifero", "tubulo", "tubule", "epitelio_germinal", "seminiferous_tubule"]
    for t_alias in tubule_aliases:
        forbidden_map.setdefault(t_alias, [])
        for lc in ["celula_leydig", "leydig", "celula_intersticial"]:
            if lc not in forbidden_map[t_alias]:
                forbidden_map[t_alias].append(lc)

    # 2. Interstitial space must strictly forbid all germ cells and Sertoli
    interstitial_aliases = ["espacio_intersticial", "intersticio", "interstitium", "interstitial_space", "estroma_intersticial"]
    for i_alias in interstitial_aliases:
        forbidden_map.setdefault(i_alias, [])
        spatial_map.setdefault(i_alias, [])
        for gc in testis_germ_cells:
            if gc not in forbidden_map[i_alias]:
                forbidden_map[i_alias].append(gc)
        for ic in interstitial_cells:
            if ic not in spatial_map[i_alias]:
                spatial_map[i_alias].append(ic)

    # 3. Lumen must forbid spermatogonia, spermatocytes, Sertoli, and Leydig
    lumen_aliases = ["luz_tubular", "luz", "lumen", "tubular_lumen"]
    for l_alias in lumen_aliases:
        forbidden_map.setdefault(l_alias, [])
        spatial_map.setdefault(l_alias, [])
        for non_luminal in [
            "espermatogonia_a_clara", "espermatogonia_a_oscura", "espermatogonia_b",
            "espermatocito_primario", "espermatocito_secundario", "celula_sertoli",
            "celula_leydig", "leydig", "celula_peritubular"
        ]:
            if non_luminal not in forbidden_map[l_alias]:
                forbidden_map[l_alias].append(non_luminal)
        for lum in ["espermatozoide", "espermatide_tardia", "espermatide"]:
            if lum not in spatial_map[l_alias]:
                spatial_map[l_alias].append(lum)

    return spatial_rules_lookup, spatial_map, forbidden_map


def enforce_spatial_rules_on_detections(
    detections: List[Dict[str, Any]],
    spatial_rules_lookup: Dict[str, Dict[str, Any]],
    spatial_map: Dict[str, List[str]],
    forbidden_map: Dict[str, List[str]],
    macro_annotations: Optional[List[Dict[str, Any]]] = None,
    class_meta: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Tuple[List[Dict[str, Any]], int, List[Dict[str, Any]]]:
    """
    Enforces topological constraints on segmented cells. If a cell classification violates
    anatomical boundaries (e.g. Leydig cell inside seminiferous tubule or spermatogonia in
    interstitial space), it reassigns it to the most biologically valid class for that layer.

    Returns:
        (updated_detections, corrections_count, corrections_log)
    """
    import cv2
    import numpy as np

    # Build spatial index of macro polygons if provided
    macro_polys_by_key: Dict[str, List[np.ndarray]] = {}
    if macro_annotations:
        for macro in macro_annotations:
            m_key = str(macro.get("class_key") or macro.get("category_id") or macro.get("key") or macro.get("label") or "").strip().lower()
            if not m_key:
                continue
            segs = macro.get("segmentation") or []
            if isinstance(segs, list):
                for poly in segs:
                    if isinstance(poly, list) and len(poly) >= 6:
                        pts = np.array(poly, dtype=np.float32).reshape(-1, 2)
                        macro_polys_by_key.setdefault(m_key, []).append(pts)

    meta_lookup = class_meta or {}
    corrections_log: List[Dict[str, Any]] = []
    corrections_count = 0

    for idx, det in enumerate(detections):
        d_key = str(det.get("class_key") or det.get("category_id") or "").strip().lower()
        if not d_key or d_key == "unclassified":
            continue

        # 1. Determine cell centroid
        cx, cy = 0.0, 0.0
        has_centroid = False
        if "centroid" in det and isinstance(det["centroid"], (list, tuple)) and len(det["centroid"]) == 2:
            cx, cy = float(det["centroid"][0]), float(det["centroid"][1])
            has_centroid = True
        elif "bbox" in det and isinstance(det["bbox"], (list, tuple)) and len(det["bbox"]) == 4:
            x, y, w, h = [float(v) for v in det["bbox"]]
            cx, cy = x + w / 2.0, y + h / 2.0
            has_centroid = True
        elif "box" in det and isinstance(det["box"], (list, tuple)) and len(det["box"]) == 4:
            x1, y1, x2, y2 = [float(v) for v in det["box"]]
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            has_centroid = True

        # 2. Determine containing anatomical compartment
        comp = str(det.get("containing_layer") or det.get("compartment") or "").strip().lower()

        # If geometric macro polygons exist and centroid is available, check polygon containment
        if macro_polys_by_key and has_centroid:
            # Order of evaluation: lumen (inner cavity) -> basement membrane -> tubule -> interstitium
            matched_macro = None
            if "luz_tubular" in macro_polys_by_key:
                for poly in macro_polys_by_key["luz_tubular"]:
                    if cv2.pointPolygonTest(poly, (cx, cy), False) >= 0:
                        matched_macro = "luz_tubular"
                        break

            if not matched_macro and "membrana_basal" in macro_polys_by_key:
                for poly in macro_polys_by_key["membrana_basal"]:
                    if cv2.pointPolygonTest(poly, (cx, cy), False) >= 0:
                        matched_macro = "membrana_basal"
                        break

            if not matched_macro and "tubulo_seminifero" in macro_polys_by_key:
                for poly in macro_polys_by_key["tubulo_seminifero"]:
                    if cv2.pointPolygonTest(poly, (cx, cy), False) >= 0:
                        matched_macro = "tubulo_seminifero"
                        break

            if not matched_macro and "espacio_intersticial" in macro_polys_by_key:
                for poly in macro_polys_by_key["espacio_intersticial"]:
                    if cv2.pointPolygonTest(poly, (cx, cy), False) >= 0:
                        matched_macro = "espacio_intersticial"
                        break

            if matched_macro:
                comp = matched_macro
                det["containing_layer"] = comp
                det["compartment"] = comp

        if not comp:
            continue

        # 3. Check for anatomical rule violation
        rule = spatial_rules_lookup.get(d_key, {})
        forbidden_in = rule.get("forbidden_in", [])
        forbidden_in_comp = forbidden_map.get(comp, [])

        is_violated = (
            comp in forbidden_in
            or d_key in forbidden_in_comp
            or any(f in comp for f in forbidden_in)
            or any(k in d_key for k in forbidden_in_comp)
        )

        if is_violated:
            # Reassign to an allowed class in this compartment
            replacement_key = None
            allowed_in_comp = spatial_map.get(comp, [])

            if "interstic" in comp:
                # In interstitium: prioritize Leydig or peritubular
                for pref in ["celula_leydig", "leydig", "celula_peritubular", "celula_intersticial"]:
                    if pref in allowed_in_comp or pref in meta_lookup:
                        replacement_key = pref
                        break
                if not replacement_key:
                    replacement_key = "celula_leydig"

            elif "luz" in comp or "lumen" in comp:
                # In lumen: prioritize spermatozoa or late spermatid
                for pref in ["espermatozoide", "espermatide_tardia", "espermatide"]:
                    if pref in allowed_in_comp or pref in meta_lookup:
                        replacement_key = pref
                        break
                if not replacement_key:
                    replacement_key = "espermatozoide"

            elif "basal" in comp or "membrana" in comp:
                # At boundary/basal: prioritize spermatogonia or Sertoli
                for pref in ["espermatogonia_a_clara", "espermatogonia_a_oscura", "celula_sertoli", "celula_peritubular"]:
                    if pref in allowed_in_comp or pref in meta_lookup:
                        replacement_key = pref
                        break
                if not replacement_key:
                    replacement_key = "espermatogonia_a_clara"

            elif "tubulo" in comp or "tubule" in comp:
                # Inside tubule: if Leydig, reassign to basal germ cell or intermediate spermatocyte
                for pref in ["espermatogonia_a_clara", "espermatocito_primario", "celula_sertoli"]:
                    if pref in allowed_in_comp or pref in meta_lookup:
                        replacement_key = pref
                        break
                if not replacement_key:
                    replacement_key = "espermatogonia_a_clara"

            if replacement_key and replacement_key != d_key:
                target_meta = meta_lookup.get(replacement_key) or spatial_rules_lookup.get(replacement_key, {})
                rep_name = target_meta.get("name") or target_meta.get("label") or replacement_key.replace("_", " ").title()
                rep_color = target_meta.get("color") or "#10b981"
                orig_label = det.get("label") or d_key

                reason = (
                    f"Violación ontológica: '{orig_label}' prohibida en '{comp}'. "
                    f"Reasignada automáticamente a '{rep_name}' ({rule.get('rule_description', 'estrato válido')})."
                )

                det["original_class_key"] = d_key
                det["original_label"] = orig_label
                det["class_key"] = replacement_key
                det["category_id"] = replacement_key
                det["label"] = rep_name
                det["class_label"] = rep_name
                det["color"] = rep_color
                det["spatial_corrected"] = True
                det["spatial_compartment"] = comp
                det["correction_reason"] = reason

                # Append to cytological reasoning
                prev_reason = det.get("cytological_reasoning") or det.get("reasoning") or ""
                det["cytological_reasoning"] = f"[{comp.upper()}] {prev_reason} | ⚡ {reason}".strip()

                corrections_count += 1
                corrections_log.append({
                    "detection_index": idx,
                    "original_class": d_key,
                    "corrected_class": replacement_key,
                    "compartment": comp,
                    "reason": reason,
                })

    return detections, corrections_count, corrections_log


