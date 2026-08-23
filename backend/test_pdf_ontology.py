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


if __name__ == "__main__":
    unittest.main()
