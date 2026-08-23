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

# Palette for auto-assigned ontology colors (distinguishable, not too light)
DEFAULT_COLORS = [
    "#e11d48", "#8b5cf6", "#06b6d4", "#f59e0b", "#10b981",
    "#ec4899", "#6366f1", "#14b8a6", "#f97316", "#84cc16",
    "#a855f7", "#0ea5e9", "#ef4444", "#22c55e", "#eab308",
    "#d946ef", "#38bdf8", "#fb923c", "#4ade80", "#facc15",
]

# System prompt for ontology extraction — domain-agnostic
ONTOLOGY_SYSTEM_PROMPT = """\
You are an expert computational histopathologist and ontology engineer. Your task is to analyze \
academic histology/microscopy text and extract a structured ontology of distinct visual structures, \
cell types, lumens, and tissue compartments.

RULES:
1. For EVERY structure, provide a "prompt" field: a CONCISE English visual phrase (3-8 words max) \
   optimized for the SAM 3 open-vocabulary segmentation model.
   - Use direct visual nouns: "dark round cell nucleus", "tubular lumen cavity", "elongated spindle cell", "pale chromatin nucleus".
   - Avoid long comparative sentences ("smaller than...", "often found in...").
   - Focus on visual cues visible under H&E or brightfield microscopy (shape, staining intensity, chromatin texture).
2. The "name" field should be the canonical name in the original language (Spanish).
3. The "name_en" field is the direct English translation.
4. The "parent" field references the key of the parent anatomical structure (or null).
5. Generate a short "key" (snake_case, ASCII only).
6. Return ONLY a valid JSON array — no markdown fences, no commentary.

EXAMPLE OUTPUT:
[
  {
    "key": "seminiferous_tubule",
    "name": "Túbulo seminífero",
    "name_en": "Seminiferous tubule",
    "prompt": "circular tubule cross-section with lumen",
    "parent": null
  },
  {
    "key": "spermatogonia",
    "name": "Espermatogonia",
    "name_en": "Spermatogonium",
    "prompt": "small dark round nucleus at basement membrane",
    "parent": "seminiferous_tubule"
  },
  {
    "key": "leydig_cell",
    "name": "Célula de Leydig",
    "name_en": "Leydig cell",
    "prompt": "polygonal cell cluster in interstitial space",
    "parent": null
  }
]
"""

ONTOLOGY_USER_PROMPT_TEMPLATE = """\
Analyze the following academic text (in Spanish) and extract ALL visually \
distinct biological structures, cell types, tissue regions, or relevant \
objects. For each one, generate a visual English prompt suitable for the \
SAM 3 open-vocabulary segmentation model.

TEXT:
{text}
"""


# ---------------------------------------------------------------------------
# PDF Content Extraction (pymupdf)
# ---------------------------------------------------------------------------

def extract_pdf_content(
    pdf_bytes: bytes,
    filename: str,
    min_image_size: int = 100,
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
            try:
                pix = fitz.Pixmap(doc, xref)

                # Skip very small images (icons, decorations)
                if pix.width < min_image_size or pix.height < min_image_size:
                    pix = None
                    continue

                # Convert CMYK to RGB if necessary
                if pix.n > 4:
                    pix = fitz.Pixmap(fitz.csRGB, pix)
                elif pix.n == 4:
                    pix = fitz.Pixmap(fitz.csRGB, pix)

                img_filename = f"{pdf_id}_p{page_num + 1}_img{img_index + 1}.png"
                img_path = images_dir / img_filename

                pix.save(str(img_path))

                # Try to find a caption near the image
                caption = _find_image_caption(page_text, page_num + 1, img_index)

                extracted_images.append({
                    "filename": img_filename,
                    "path": str(img_path),
                    "page": page_num + 1,
                    "width": pix.width,
                    "height": pix.height,
                    "caption": caption,
                })
                image_count += 1
                pix = None

            except Exception as e:
                logger.warning(f"Failed to extract image xref={xref} from page {page_num + 1}: {e}")
                continue

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
    api_key: str,
    model_name: str = "gemini-2.5-flash",
    max_text_chars: int = 80000,
) -> List[Dict[str, Any]]:
    """
    Send extracted PDF text to Gemini and get back a structured ontology.

    Args:
        extracted_text: Full text extracted from PDF.
        api_key: Gemini API key.
        model_name: Gemini model to use.
        max_text_chars: Max characters to send (to stay within context limits).

    Returns:
        List of ontology structure dicts.
    """
    try:
        from google import genai
    except ImportError:
        raise ImportError(
            "google-genai is required. Install with: pip install google-genai"
        )

    # Truncate if very long
    text_for_llm = extracted_text[:max_text_chars]
    if len(extracted_text) > max_text_chars:
        logger.info(
            f"Text truncated from {len(extracted_text)} to {max_text_chars} chars for LLM"
        )

    client = genai.Client(api_key=api_key)

    user_prompt = ONTOLOGY_USER_PROMPT_TEMPLATE.format(text=text_for_llm)

    response = client.models.generate_content(
        model=model_name,
        contents=user_prompt,
        config={
            "system_instruction": ONTOLOGY_SYSTEM_PROMPT,
            "temperature": 0.2,
            "response_mime_type": "application/json",
        },
    )

    raw_text = response.text.strip()

    # Parse JSON — handle potential markdown fences
    if raw_text.startswith("```"):
        raw_text = re.sub(r"^```(?:json)?\s*", "", raw_text)
        raw_text = re.sub(r"\s*```$", "", raw_text)

    structures = json.loads(raw_text)

    if not isinstance(structures, list):
        raise ValueError(f"Expected JSON array from LLM, got {type(structures).__name__}")

    # Assign colors if missing
    for i, struct in enumerate(structures):
        if "color" not in struct:
            struct["color"] = DEFAULT_COLORS[i % len(DEFAULT_COLORS)]
        # Ensure label field (use name)
        if "label" not in struct:
            struct["label"] = struct.get("name", struct.get("key", f"Clase {i + 1}"))

    logger.info(f"Generated ontology with {len(structures)} structures")
    return structures


# ---------------------------------------------------------------------------
# Ontology Storage
# ---------------------------------------------------------------------------

def build_ontology_document(
    pdf_id: str,
    filename: str,
    structures: List[Dict[str, Any]],
    extracted_images: List[Dict[str, Any]],
    domain_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a complete ontology document for storage."""
    if domain_name is None:
        # Derive from filename
        base = Path(filename).stem
        domain_name = re.sub(r"[^a-zA-Z0-9_]", "_", base).lower()

    return {
        "domain": domain_name,
        "source_pdf": filename,
        "pdf_id": pdf_id,
        "structures": structures,
        "extracted_images": extracted_images,
        "prompts": [
            {
                "key": s["key"],
                "prompt": s["prompt"],
                "label": s.get("label", s.get("name", s["key"])),
                "color": s.get("color", DEFAULT_COLORS[i % len(DEFAULT_COLORS)]),
            }
            for i, s in enumerate(structures)
        ],
    }


def save_ontology(ontology: Dict[str, Any]) -> str:
    """Save ontology JSON to disk. Returns the file path."""
    ONTOLOGIES_DIR.mkdir(parents=True, exist_ok=True)
    domain = ontology.get("domain", "unnamed")
    filepath = ONTOLOGIES_DIR / f"{domain}.json"
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(ontology, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved ontology to {filepath}")
    return str(filepath)


def load_ontology(name: str) -> Optional[Dict[str, Any]]:
    """Load an ontology by domain name."""
    filepath = ONTOLOGIES_DIR / f"{name}.json"
    if not filepath.exists():
        return None
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


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
    """Update the structures (and regenerate prompts) of a saved ontology."""
    ontology = load_ontology(name)
    if ontology is None:
        return None

    ontology["structures"] = structures
    # Regenerate prompts from updated structures
    ontology["prompts"] = [
        {
            "key": s["key"],
            "prompt": s["prompt"],
            "label": s.get("label", s.get("name", s["key"])),
            "color": s.get("color", DEFAULT_COLORS[i % len(DEFAULT_COLORS)]),
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
    """Get the absolute path to an extracted PDF image."""
    path = PDF_IMAGES_DIR / pdf_id / filename
    if path.exists():
        return str(path)
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

