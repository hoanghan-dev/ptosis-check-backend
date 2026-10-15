"""
Unit tests cho Phase 1: Kiểm thử API, In-Memory Zero-Retention và Schema contracts.
Tuân thủ G4: Tuyệt đối không dùng từ ngữ thuộc danh mục cấm.
"""

import io
import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.core.config import settings

client = TestClient(app)


from pathlib import Path

_CURRENT_DIR = Path(__file__).resolve().parent

CANDIDATE_PATHS = [
    _CURRENT_DIR / "fixtures" / "sample_face.jpg",
    Path("/app/tests/fixtures/sample_face.jpg"),
    Path("tests/fixtures/sample_face.jpg"),
    Path("HIRSCHBERG_METHOD/harschberg_data_detect/normal_harschberg/img_023e6caea47f18ee__normal_truc.jpg"),
    Path("../HIRSCHBERG_METHOD/harschberg_data_detect/normal_harschberg/img_023e6caea47f18ee__normal_truc.jpg"),
    Path("/app/HIRSCHBERG_METHOD/harschberg_data_detect/normal_harschberg/img_023e6caea47f18ee__normal_truc.jpg"),
    Path("HIRSCHBERG_METHOD/hirschberg_review_all_v1/normal/img_023e6caea47f18ee__normal_truc.jpg"),
    Path("../HIRSCHBERG_METHOD/hirschberg_review_all_v1/normal/img_023e6caea47f18ee__normal_truc.jpg"),
    Path("/app/HIRSCHBERG_METHOD/hirschberg_review_all_v1/normal/img_023e6caea47f18ee__normal_truc.jpg"),
]


def create_synthetic_jpeg_bytes() -> bytes:
    """Tạo hoặc nạp ảnh JPEG hợp lệ hoàn toàn trong RAM."""
    for p in CANDIDATE_PATHS:
        if p.exists():
            img = cv2.imread(str(p))
            if img is not None:
                success, buffer = cv2.imencode(".jpg", img)
                assert success
                return buffer.tobytes()
    synthetic_img = np.zeros((300, 300, 3), dtype=np.uint8)
    synthetic_img[:] = (200, 200, 200)
    cv2.circle(synthetic_img, (150, 150), 50, (50, 50, 50), -1)
    success, buffer = cv2.imencode(".jpg", synthetic_img)
    assert success
    return buffer.tobytes()


def test_get_thresholds():
    """Kiểm thử GET /api/v1/config/thresholds trả về bảng ngưỡng lâm sàng đầy đủ."""
    response = client.get("/api/v1/config/thresholds")
    assert response.status_code == 200
    data = response.json()
    assert data["absolute_severe_mrd1_mm"] == settings.ABSOLUTE_SEVERE_MRD1_MM
    assert data["tier_a_min_mrd1_mm"] == settings.TIER_A_MIN_MRD1_MM
    assert data["assumed_corneal_diameter_mm"] == 11.71
    assert data["nominal_tolerance_mm"] == 0.5


def test_analyze_valid_jpeg():
    """TC-1.1: Upload ảnh JPEG hợp lệ -> HTTP 200, JSON MeasurementPayload mock."""
    jpeg_bytes = create_synthetic_jpeg_bytes()
    files = {"file": ("test_face.jpg", io.BytesIO(jpeg_bytes), "image/jpeg")}
    response = client.post("/api/v1/screening/analyze", files=files)

    assert response.status_code == 200
    payload = response.json()
    assert "session_id" in payload
    assert payload["image_quality_audit"]["passed"] is True
    assert "measurements" in payload
    assert "right_eye_od" in payload["measurements"]
    assert "left_eye_os" in payload["measurements"]
    assert isinstance(payload["measurements"]["right_eye_od"]["mrd1_mm"], (int, float))
    assert payload["measurements"]["right_eye_od"]["mrd1_mm"] > 0
    assert "triage_result" in payload
    assert payload["triage_result"]["tier"] == "TIER_B"
    assert "safety_disclaimer_vi" in payload["triage_result"]


def test_analyze_corrupt_data():
    """Kiểm thử gửi dữ liệu hỏng / không thể decode thành ảnh -> HTTP 400."""
    corrupt_bytes = b"This is not an image file format at all"
    files = {"file": ("corrupt.jpg", io.BytesIO(corrupt_bytes), "image/jpeg")}
    response = client.post("/api/v1/screening/analyze", files=files)

    assert response.status_code == 400
    data = response.json()
    assert "detail" in data
    assert "Không thể giải mã" in data["detail"]


def test_health_check():
    """Kiểm thử endpoint /health."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}
