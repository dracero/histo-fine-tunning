"""
Unit tests for Cellpose & Cellpose-SAM Histological Segmentation and GPU VRAM Management.
"""

import unittest
import numpy as np
from PIL import Image

from backend.cellpose_segmenter import (
    is_cellpose_available,
    get_cellpose_status,
    load_cellpose_model,
    run_cellpose_segmentation,
    offload_cellpose_to_cpu,
    AVAILABLE_CELLPOSE_MODELS,
)
from backend.main import prepare_engine_vram


class TestCellposeSegmentation(unittest.TestCase):

    def test_is_cellpose_available(self) -> None:
        self.assertTrue(is_cellpose_available(), "Cellpose must be installed and importable in environment.")

    def test_cellpose_status(self) -> None:
        status = get_cellpose_status()
        self.assertIsInstance(status, dict)
        self.assertTrue(status.get("available"))
        self.assertIn("device", status)
        self.assertIn("models", status)
        self.assertIn("cpsam", status["models"])
        self.assertIn("cpdino-vitb", status["models"])

    def test_load_and_run_cellpose_segmentation(self) -> None:
        # Create a synthetic image with a solid circle (cell simulation)
        img_np = np.zeros((128, 128, 3), dtype=np.uint8)
        # Draw a synthetic circle
        import cv2
        cv2.circle(img_np, (64, 64), 20, (220, 200, 240), -1)
        cv2.circle(img_np, (64, 64), 8, (70, 40, 120), -1)

        res = run_cellpose_segmentation(
            image_input=img_np,
            model_type="cpsam",
            prompt_label="spermatocyte",
            min_area=10,
        )

        self.assertIsInstance(res, dict)
        self.assertTrue(res.get("success"))
        self.assertEqual(res.get("engine"), "cellpose")
        self.assertIn("detections", res)
        self.assertIn("groups", res)
        self.assertIn("inference_time_seconds", res)

    def test_vram_swapping(self) -> None:
        # Test switching to cellpose
        prepare_engine_vram("cellpose")
        # Test switching back to sam3
        prepare_engine_vram("sam3")
        # Offload cellpose
        offload_cellpose_to_cpu()


if __name__ == "__main__":
    unittest.main()
