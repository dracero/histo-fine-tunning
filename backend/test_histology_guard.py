"""
Unit tests for Histology Guard and CONCH/UNI Domain Restriction.
"""

import unittest
from PIL import Image

from pdf_ontology import is_histology_ontology
from pathology_models import (
    classify_detections_with_conch,
    discriminate_and_cluster_with_pathology_models,
    extract_detection_embeddings_uni,
)


class TestHistologyGuard(unittest.TestCase):

    def test_is_histology_ontology_detection(self):
        # 1. Histology domain test
        histo_ontology = {
            "domain": "testiculo_histologia",
            "source_pdf": "Atlas_de_Histologia.pdf",
            "structures": [
                {
                    "key": "seminiferous_tubule",
                    "name": "Túbulo seminífero",
                    "prompt": "circular tubule cross-section with lumen",
                },
                {
                    "key": "leydig_cell",
                    "name": "Célula de Leydig",
                    "prompt": "polygonal cell cluster in interstitial space",
                },
            ],
        }
        self.assertTrue(is_histology_ontology(histo_ontology))

        # 2. General non-histology domain test (e.g. vehicles)
        general_ontology = {
            "domain": "vehiculos_urbanos",
            "source_pdf": "Manual_Autos.pdf",
            "structures": [
                {
                    "key": "wheel",
                    "name": "Rueda de automóvil",
                    "prompt": "black rubber car wheel tire",
                },
                {
                    "key": "windshield",
                    "name": "Parabrisas",
                    "prompt": "front glass windshield of sedan car",
                },
            ],
        }
        self.assertFalse(is_histology_ontology(general_ontology))

        # 3. Explicit is_histology flag overrides
        override_false = {
            "domain": "histology_text_but_flagged_false",
            "is_histology": False,
            "structures": [{"key": "cell", "prompt": "cell"}],
        }
        self.assertFalse(is_histology_ontology(override_false))

    def test_conch_skipped_for_non_histology(self):
        dummy_image = Image.new("RGB", (100, 100), color="white")
        detections = [
            {
                "id": "det1",
                "bbox": [10, 10, 50, 50],
                "polygon": [[10, 10], [50, 10], [50, 50], [10, 50]],
                "score": 0.9,
            }
        ]
        classes = [{"key": "car", "label": "Car", "prompt": "car wheel", "color": "#ff0000"}]

        # Call classify_detections_with_conch with is_histology=False
        res = classify_detections_with_conch(
            image=dummy_image,
            detections=detections,
            candidate_classes=classes,
            is_histology=False,
        )

        self.assertEqual(len(res), 1)
        self.assertTrue(res[0].get("conch_skipped"))
        self.assertIn("restricted", res[0].get("conch_reason", "").lower())

    def test_discriminate_and_cluster_bypassed_for_non_histology(self):
        dummy_image = Image.new("RGB", (100, 100), color="white")
        detections = [
            {
                "id": "det1",
                "bbox": [10, 10, 50, 50],
                "initial_class_key": "car_wheel",
                "initial_label": "Rueda",
            }
        ]

        res = discriminate_and_cluster_with_pathology_models(
            image=dummy_image,
            detections=detections,
            is_histology=False,
        )

        self.assertEqual(len(res), 1)
        self.assertEqual(res[0]["class_key"], "car_wheel")
        self.assertTrue(res[0].get("conch_skipped"))

    def test_uni_embeddings_skipped_for_non_histology(self):
        dummy_image = Image.new("RGB", (100, 100), color="white")
        detections = [{"id": "det1", "bbox": [10, 10, 50, 50]}]

        embeddings = extract_detection_embeddings_uni(
            image=dummy_image,
            detections=detections,
            is_histology=False,
        )

        self.assertEqual(embeddings, [])


if __name__ == "__main__":
    unittest.main()
