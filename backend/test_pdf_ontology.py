"""
Integration & unit tests for pdf_ontology module and FastAPI backend endpoints.
"""

import os
import json
import unittest
from pathlib import Path

from pdf_ontology import (
    build_ontology_document,
    save_ontology,
    load_ontology,
    list_ontologies,
    update_ontology_structures,
    get_ontology_prompts,
    merge_ontology_structures,
    ONTOLOGIES_DIR,
)

TEST_DOMAIN = "test_histologia_testiculo"


class TestPdfOntology(unittest.TestCase):
    def tearDown(self):
        test_file = ONTOLOGIES_DIR / f"{TEST_DOMAIN}.json"
        if test_file.exists():
            test_file.unlink()

    def test_build_and_save_ontology(self):
        structures = [
            {
                "key": "espermatogonia_b",
                "name": "Espermatogonia tipo B",
                "name_en": "Type B spermatogonium",
                "prompt": "small dark round cell at basement membrane",
                "label": "Espermatogonia B",
                "color": "#e11d48",
            },
            {
                "key": "tubulo_seminifero",
                "name": "Túbulo seminífero",
                "name_en": "Seminiferous tubule",
                "prompt": "large circular cross-section with lumen",
                "label": "Túbulo Seminífero",
                "color": "#8b5cf6",
            }
        ]

        doc = build_ontology_document(
            pdf_id="test1234",
            filename="test_paper.pdf",
            structures=structures,
            extracted_images=[],
            domain_name=TEST_DOMAIN
        )

        filepath = save_ontology(doc)
        self.assertTrue(Path(filepath).exists())

        loaded = load_ontology(TEST_DOMAIN)
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded["domain"], TEST_DOMAIN)
        self.assertEqual(len(loaded["structures"]), 2)

        # Check prompts format for SAM3
        prompts = get_ontology_prompts(TEST_DOMAIN)
        self.assertIsNotNone(prompts)
        self.assertEqual(len(prompts), 2)
        self.assertEqual(prompts[0]["key"], "espermatogonia_b")
        self.assertEqual(prompts[0]["prompt"], "small dark round cell at basement membrane")

    def test_update_ontology_structures(self):
        structures = [
            {
                "key": "celula_sertoli",
                "name": "Célula de Sertoli",
                "prompt": "columnar cell spanning basement to lumen",
                "label": "Célula de Sertoli",
                "color": "#06b6d4"
            }
        ]
        doc = build_ontology_document(
            pdf_id="test5678",
            filename="test_paper_2.pdf",
            structures=structures,
            extracted_images=[],
            domain_name=TEST_DOMAIN
        )
        save_ontology(doc)

        updated_structures = [
            {
                "key": "celula_sertoli",
                "name": "Célula de Sertoli Modificada",
                "prompt": "tall columnar cell with pale nucleus",
                "label": "Sertoli Custom",
                "color": "#10b981"
            }
        ]

        res = update_ontology_structures(TEST_DOMAIN, updated_structures)
        self.assertIsNotNone(res)
        self.assertEqual(res["structures"][0]["prompt"], "tall columnar cell with pale nucleus")

        prompts = get_ontology_prompts(TEST_DOMAIN)
        self.assertEqual(prompts[0]["prompt"], "tall columnar cell with pale nucleus")
        self.assertEqual(prompts[0]["color"], "#10b981")

    def test_merge_ontology_structures(self):
        """Test incremental merge: existing + new → deduplicated union."""
        # Step 1: Create initial ontology with 2 structures
        initial_structures = [
            {
                "key": "espermatogonia_b",
                "name": "Espermatogonia tipo B",
                "prompt": "small dark round cell at basement membrane",
                "label": "Espermatogonia B",
                "color": "#e11d48",
            },
            {
                "key": "tubulo_seminifero",
                "name": "Túbulo seminífero",
                "prompt": "large circular cross-section with lumen",
                "label": "Túbulo Seminífero",
                "color": "#8b5cf6",
            },
        ]
        initial_images = [
            {"filename": "pdf1_p1_img1.png", "path": "/tmp/pdf1_p1_img1.png", "page": 1, "width": 100, "height": 100, "caption": None},
        ]
        doc = build_ontology_document(
            pdf_id="pdf_first",
            filename="first_paper.pdf",
            structures=initial_structures,
            extracted_images=initial_images,
            domain_name=TEST_DOMAIN,
        )
        save_ontology(doc)

        # Step 2: Merge new structures from a second PDF
        new_structures = [
            {
                "key": "tubulo_seminifero",  # duplicate key → should update, not duplicate
                "name": "Túbulo seminífero (actualizado)",
                "prompt": "circular tubule with spermatids near lumen",
                "label": "Túbulo (v2)",
            },
            {
                "key": "celula_leydig",  # brand new → should be added
                "name": "Célula de Leydig",
                "prompt": "polygonal cell cluster in interstitial space",
                "label": "Célula de Leydig",
            },
        ]
        new_images = [
            {"filename": "pdf2_p1_img1.png", "path": "/tmp/pdf2_p1_img1.png", "page": 1, "width": 200, "height": 200, "caption": None},
            {"filename": "pdf1_p1_img1.png", "path": "/tmp/pdf1_p1_img1.png", "page": 1, "width": 100, "height": 100, "caption": None},  # duplicate
        ]

        existing = load_ontology(TEST_DOMAIN)
        self.assertIsNotNone(existing)

        merged = merge_ontology_structures(
            existing_ontology=existing,
            new_structures=new_structures,
            new_pdf_id="pdf_second",
            new_filename="second_paper.pdf",
            new_images=new_images,
        )

        # Verify: 3 unique structures (espermatogonia_b, tubulo_seminifero updated, celula_leydig new)
        self.assertEqual(len(merged["structures"]), 3)
        keys = [s["key"] for s in merged["structures"]]
        self.assertIn("espermatogonia_b", keys)
        self.assertIn("tubulo_seminifero", keys)
        self.assertIn("celula_leydig", keys)

        # tubulo_seminifero should have the updated prompt
        tubulo = next(s for s in merged["structures"] if s["key"] == "tubulo_seminifero")
        self.assertEqual(tubulo["prompt"], "circular tubule with spermatids near lumen")

        # source_pdfs should list both PDFs
        self.assertEqual(len(merged["source_pdfs"]), 2)
        pdf_ids = [sp["pdf_id"] for sp in merged["source_pdfs"]]
        self.assertIn("pdf_first", pdf_ids)
        self.assertIn("pdf_second", pdf_ids)

        # Images should be deduplicated: 2 unique (pdf1_p1_img1 + pdf2_p1_img1)
        self.assertEqual(len(merged["extracted_images"]), 2)
        img_filenames = {im["filename"] for im in merged["extracted_images"]}
        self.assertEqual(img_filenames, {"pdf1_p1_img1.png", "pdf2_p1_img1.png"})

        # Prompts should have 3 entries
        self.assertEqual(len(merged["prompts"]), 3)

        # Save and reload to confirm persistence
        save_ontology(merged)
        reloaded = load_ontology(TEST_DOMAIN)
        self.assertEqual(len(reloaded["structures"]), 3)
        self.assertEqual(len(reloaded["source_pdfs"]), 2)


if __name__ == "__main__":
    unittest.main()

