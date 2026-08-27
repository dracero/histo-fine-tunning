"""
Unit and Integration Tests for Histology Multi-Agent Graph.
"""

import numpy as np
import pytest
from PIL import Image

from backend.histology_graph import (
    HistologyGraphState,
    HistologyMultiAgentPipeline,
    build_histology_graph,
    ontology_reader_node,
    segmentation_cropper_node,
    final_classifier_node,
)


@pytest.fixture
def sample_image() -> Image.Image:
    # 512x512 synthetic histological-like image (pink background with violet blobs)
    arr = np.full((512, 512, 3), [240, 220, 230], dtype=np.uint8)
    # Draw a simulated nucleus (dark purple/violet)
    arr[100:140, 100:140] = [80, 20, 100]
    arr[300:330, 300:330] = [70, 30, 90]
    return Image.fromarray(arr)


@pytest.fixture
def sample_ontology():
    return [
        {
            "key": "spermatogonia",
            "name": "Espermatogonia",
            "prompt": "round dark basal germ cell nucleus with dense chromatin",
            "color": "#e11d48",
            "is_cellular": True,
        },
        {
            "key": "sertoli_cell",
            "name": "Célula de Sertoli",
            "prompt": "pale vesicular oval nucleus with prominent central nucleolus",
            "color": "#10b981",
            "is_cellular": True,
        },
    ]


@pytest.fixture
def sample_detections():
    return [
        {
            "id": "cell_001",
            "bbox": [100, 100, 40, 40],
            "polygon": [[100, 100], [140, 100], [140, 140], [100, 140]],
            "confidence": 0.85,
        },
        {
            "id": "cell_002",
            "bbox": [300, 300, 30, 30],
            "polygon": [[300, 300], [330, 300], [330, 330], [300, 330]],
            "confidence": 0.90,
        },
    ]


def test_build_graph():
    graph = build_histology_graph()
    assert graph is not None


def test_ontology_reader_node(sample_ontology):
    state: HistologyGraphState = {
        "raw_ontology": sample_ontology,
    }
    result = ontology_reader_node(state)
    assert "candidate_classes" in result
    assert len(result["candidate_classes"]) >= 2
    keys = [c["key"] for c in result["candidate_classes"]]
    assert "spermatogonia" in keys
    assert "sertoli_cell" in keys


def test_segmentation_cropper_node(sample_image, sample_detections):
    state: HistologyGraphState = {
        "image_pil": sample_image,
        "detections": sample_detections,
        "detected_figure_labels": [
            {
                "text": "S",
                "meaning": "Célula de Sertoli",
                "center": (110.0, 110.0),
            }
        ],
    }
    result = segmentation_cropper_node(state)
    assert "detections" in result
    dets = result["detections"]
    assert len(dets) == 2

    # cell_001 is at (100,100), close to (110,110)
    c1 = dets[0]
    assert c1["area"] > 0
    assert "circularity" in c1
    assert "mean_intensity" in c1
    assert "spatial_label_hint" in c1
    assert c1["spatial_label_hint"]["text"] == "S"
    assert c1["spatial_label_hint"]["meaning"] == "Célula de Sertoli"


def test_final_classifier_node_with_spatial_consensus(sample_ontology, sample_detections):
    # Detections enriched with spatial hint and candidate classes
    sample_detections[0]["spatial_label_hint"] = {
        "text": "sertoli_cell",
        "meaning": "Célula de Sertoli",
        "distance_px": 20.0,
    }
    sample_detections[1]["class_key"] = "spermatogonia"
    sample_detections[1]["class_label"] = "Espermatogonia"
    sample_detections[1]["confidence"] = 0.95
    sample_detections[1]["conch_confidence"] = 0.95
    sample_detections[1]["class_scores"] = {"spermatogonia": 0.95, "sertoli_cell": 0.05}

    state: HistologyGraphState = {
        "candidate_classes": sample_ontology,
        "conch_scored_detections": sample_detections,
    }

    result = final_classifier_node(state)
    assert "final_detections" in result
    assert len(result["final_detections"]) == 2
    c1 = result["final_detections"][0]
    assert c1["class_key"] == "sertoli_cell"
    assert c1["decision_source"] == "spatial_figure_label_consensus"


def test_pipeline_end_to_end(sample_image, sample_ontology, sample_detections):
    pipeline = HistologyMultiAgentPipeline()
    output = pipeline.run(
        image=sample_image,
        detections=sample_detections,
        raw_ontology=sample_ontology,
    )
    assert "detections" in output
    assert len(output["detections"]) == 2
    assert "classification_summary" in output
    assert output["classification_summary"]["total_nuclei_classified"] == 2


def test_fastapi_agentic_endpoints(sample_image, sample_ontology, sample_detections):
    import io
    import json
    from starlette.testclient import TestClient
    from backend.main import app

    client = TestClient(app)

    # 1. Test status endpoint
    res_status = client.get("/api/pipeline/agentic-status")
    assert res_status.status_code == 200
    data_status = res_status.json()
    assert data_status["status"] == "ready"
    assert "ontology_reader" in data_status["nodes"]

    # 2. Test classify endpoint
    buf = io.BytesIO()
    sample_image.save(buf, format="JPEG")
    buf.seek(0)

    res_classify = client.post(
        "/api/pipeline/agentic-classify",
        files={"image": ("test.jpg", buf.getvalue(), "image/jpeg")},
        data={
            "detections": json.dumps(sample_detections),
            "raw_ontology": json.dumps(sample_ontology),
        },
    )
    assert res_classify.status_code == 200
    data_classify = res_classify.json()
    assert data_classify["success"] is True
    assert len(data_classify["detections"]) == 2
    assert "classification_summary" in data_classify

