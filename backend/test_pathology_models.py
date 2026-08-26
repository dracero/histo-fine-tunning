"""
Unit & Integration Tests for Pathology Foundation Models (CONCH & UNI)
and PDF Images CRUD.
"""

import io
import json
import unittest
from pathlib import Path
from PIL import Image
import torch

from pathology_models import (
    ConchModelWrapper,
    UniModelWrapper,
    VirchowModelWrapper,
    classify_detections_with_conch,
    extract_detection_embeddings_uni,
    extract_detection_embeddings_virchow,
    get_pathology_models_status,
)
from pdf_ontology import (
    add_pdf_image,
    update_pdf_image_metadata,
    delete_pdf_image,
    get_pdf_metadata,
    PDF_IMAGES_DIR,
)


class TestPathologyModels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create a dummy RGB image with distinct colored circles
        cls.test_img = Image.new("RGB", (300, 300), color="#fce7f3")

    def test_pathology_models_status(self):
        status = get_pathology_models_status()
        self.assertIn("conch", status)
        self.assertIn("uni", status)
        self.assertIn("virchow", status)
        self.assertEqual(status["conch"]["embedding_dim"], 512)
        self.assertEqual(status["uni"]["embedding_dim"], 1024)
        self.assertEqual(status["virchow"]["embedding_dim"], 1280)

    def test_conch_zero_shot_classification(self):
        detections = [
            {
                "id": "det_1",
                "bbox": [20, 20, 50, 50],
                "polygon": [[20, 20], [70, 20], [70, 70], [20, 70]],
                "confidence": 0.85,
            },
            {
                "id": "det_2",
                "bbox": [100, 100, 60, 60],
                "polygon": [[100, 100], [160, 100], [160, 160], [100, 160]],
                "confidence": 0.90,
            }
        ]

        candidate_classes = [
            {
                "key": "spermatogonia",
                "prompt": "small dark round cell at basement membrane",
                "label": "Espermatogonia",
                "color": "#e11d48",
            },
            {
                "key": "sertoli_cell",
                "prompt": "tall columnar supportive cell",
                "label": "Célula de Sertoli",
                "color": "#06b6d4",
            }
        ]

        classified = classify_detections_with_conch(
            image=self.test_img,
            detections=detections,
            candidate_classes=candidate_classes,
        )

        self.assertEqual(len(classified), 2)
        self.assertIn("class_key", classified[0])
        self.assertIn(classified[0]["class_key"], ["spermatogonia", "sertoli_cell"])
        self.assertIn("conch_confidence", classified[0])
        self.assertIn("conch_scores", classified[0])
        self.assertIn("spermatogonia", classified[0]["conch_scores"])

    def test_uni_feature_extraction(self):
        detections = [
            {
                "id": "det_1",
                "bbox": [20, 20, 50, 50],
            }
        ]

        embeddings = extract_detection_embeddings_uni(
            image=self.test_img,
            detections=detections,
        )

        self.assertEqual(len(embeddings), 1)
        self.assertEqual(len(embeddings[0]), 1024)

    def test_virchow_feature_extraction_fallback(self):
        detections = [
            {
                "id": "det_1",
                "bbox": [20, 20, 50, 50],
            }
        ]

        # Testing fallback when is_histology=False
        embeddings_false = extract_detection_embeddings_virchow(
            image=self.test_img,
            detections=detections,
            is_histology=False,
        )
        self.assertEqual(embeddings_false, [])

    def test_pdf_images_crud(self):
        test_pdf_id = "test_crud_pdf"
        test_dir = PDF_IMAGES_DIR / test_pdf_id
        if test_dir.exists():
            import shutil
            shutil.rmtree(test_dir)

        # 1. Create / Add image
        img_byte_arr = io.BytesIO()
        self.test_img.save(img_byte_arr, format="PNG")
        raw_bytes = img_byte_arr.getvalue()

        img_info = add_pdf_image(
            pdf_id=test_pdf_id,
            image_bytes=raw_bytes,
            original_filename="sample_test.png",
            caption="Figura 1: Corte histológico de prueba"
        )
        self.assertTrue(Path(img_info["path"]).exists())
        self.assertEqual(img_info["caption"], "Figura 1: Corte histológico de prueba")

        # 2. Read / Check metadata
        meta = get_pdf_metadata(test_pdf_id)
        self.assertIsNotNone(meta)
        self.assertEqual(meta["total_images"], 1)

        # 3. Update caption
        updated = update_pdf_image_metadata(
            pdf_id=test_pdf_id,
            filename=img_info["filename"],
            caption="Caption actualizado para el test"
        )
        self.assertIsNotNone(updated)
        self.assertEqual(updated["caption"], "Caption actualizado para el test")

        # 4. Delete image
        deleted = delete_pdf_image(test_pdf_id, img_info["filename"])
        self.assertTrue(deleted)
        meta_after = get_pdf_metadata(test_pdf_id)
        self.assertEqual(meta_after["total_images"], 0)

        # Cleanup
        if test_dir.exists():
            import shutil
            shutil.rmtree(test_dir)


if __name__ == "__main__":
    unittest.main()
