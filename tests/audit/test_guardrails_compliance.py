"""
Bộ kiểm toán tổng thể tuân thủ 5 nguyên tắc cốt lõi (Guardrails G1 - G5 Compliance Audit).
Xác thực hệ thống PtosisCheck-1.1 tuân thủ nghiêm ngặt:
- Guardrail G1: Tách biệt Client - Server (Backend chịu trách nhiệm 100% logic đo đạc).
- Guardrail G2: Xử lý 100% trong bộ nhớ RAM, tuyệt đối không ghi file ảnh xuống đĩa (In-Memory Zero-Retention).
- Guardrail G3: Ưu tiên đo đạc song song 2 mắt, phát hiện bất đối xứng (Asymmetry-First & Bilateral Fail-Safe).
- Guardrail G4: Tuyệt đối không can thiệp y tế chuyên sâu, không dùng từ cấm (Non-Diagnostic Safety Ban).
- Guardrail G5: Bảo toàn giao ước API với Frontend React Canvas (Strict API Contract).
"""

import io
import re
from fastapi.testclient import TestClient
import numpy as np
import pytest

from app.core.config import settings
from app.main import app
from app.schemas.measurement import MeasurementPayload
from tests.test_helpers import get_sample_image_path

client = TestClient(app)
SAMPLE_IMAGE_PATH = get_sample_image_path()


@pytest.fixture
def sample_bytes() -> bytes:
    assert SAMPLE_IMAGE_PATH.exists()
    with open(SAMPLE_IMAGE_PATH, "rb") as f:
        return f.read()


# -----------------------------------------------------------------------------
# G1: Client - Server Separation
# -----------------------------------------------------------------------------
def test_guardrail_g1_client_server_separation(sample_bytes: bytes):
    """
    Xác minh Backend tính toán 100% logic số đo (MRD1, PFH, dung sai, delta_MRD1, phân tầng).
    Frontend chỉ nhận kết quả đã hoàn thiện để hiển thị, không tính toán lại hình học.
    """
    response = client.post(
        "/api/v1/screening/analyze",
        files={"file": ("g1_test.jpg", io.BytesIO(sample_bytes), "image/jpeg")},
    )
    assert response.status_code == 200
    payload = response.json()

    # Kiểm tra các trường tính toán hoàn chỉnh do backend xuất xưởng
    meas = payload["measurements"]
    assert "mrd1_mm" in meas["right_eye_od"]
    assert "mrd1_mm" in meas["left_eye_os"]
    assert "delta_mrd1_mm" in payload["bilateral_comparison"]
    assert "tier" in payload["triage_result"]


# -----------------------------------------------------------------------------
# G2: In-Memory Zero-Retention
# -----------------------------------------------------------------------------
def test_guardrail_g2_zero_retention_compliance(sample_bytes: bytes):
    """
    Xác minh trong suốt quá trình phân tích, không có tệp tin nào được ghi ra đĩa.
    """
    import os
    from pathlib import Path

    app_dir = Path(__file__).resolve().parent.parent.parent / "app"
    image_exts = {".jpg", ".jpeg", ".png", ".webp", ".tmp"}

    before = set()
    for root, _, files in os.walk(app_dir):
        for f in files:
            if Path(f).suffix.lower() in image_exts:
                before.add(Path(root) / f)

    response = client.post(
        "/api/v1/screening/analyze",
        files={"file": ("g2_test.jpg", io.BytesIO(sample_bytes), "image/jpeg")},
    )
    assert response.status_code == 200

    after = set()
    for root, _, files in os.walk(app_dir):
        for f in files:
            if Path(f).suffix.lower() in image_exts:
                after.add(Path(root) / f)

    assert len(after - before) == 0, "Vi phạm G2: Đã ghi tệp ảnh vào thư mục app!"


# -----------------------------------------------------------------------------
# G3: Asymmetry-First & Bilateral Fail-Safe
# -----------------------------------------------------------------------------
def test_guardrail_g3_asymmetry_and_bilateral_safety(sample_bytes: bytes):
    """
    Xác minh hệ thống luôn đánh giá đồng thời cả 2 mắt OD và OS,
    tính toán delta_mrd1_mm và phân loại đối xứng sinh lý chính xác.
    """
    response = client.post(
        "/api/v1/screening/analyze",
        files={"file": ("g3_test.jpg", io.BytesIO(sample_bytes), "image/jpeg")},
    )
    assert response.status_code == 200
    payload = response.json()

    comp = payload["bilateral_comparison"]
    od_mrd1 = payload["measurements"]["right_eye_od"]["mrd1_mm"]
    os_mrd1 = payload["measurements"]["left_eye_os"]["mrd1_mm"]

    expected_delta = round(abs(os_mrd1 - od_mrd1), 2)
    assert comp["delta_mrd1_mm"] == expected_delta
    assert comp["asymmetry_classification"] in [
        "PHYSIOLOGICAL_SYMMETRY",
        "BORDERLINE_ASYMMETRY",
        "SIGNIFICANT_ASYMMETRY",
    ]


# -----------------------------------------------------------------------------
# G4: Non-Diagnostic Safety Ban
# -----------------------------------------------------------------------------
def test_guardrail_g4_api_response_clean_from_prohibited_words(sample_bytes: bytes):
    """
    Xác minh toàn bộ nội dung JSON phản hồi từ endpoint /analyze
    không chứa bất kỳ từ cấm y tế chuyên sâu nào.
    """
    banned_words = [
        "chẩn đoán", "bệnh nhân", "bệnh", "nhược cơ", "liệt",
        "horner", "bell", "myasthenia", "diagnosis", "patient", "diagnose",
    ]

    response = client.post(
        "/api/v1/screening/analyze",
        files={"file": ("g4_test.jpg", io.BytesIO(sample_bytes), "image/jpeg")},
    )
    assert response.status_code == 200
    json_text = response.text.lower()

    for bw in banned_words:
        pattern = r"(?i)\b" + re.escape(bw) + r"\b"
        matches = re.findall(pattern, json_text)
        assert len(matches) == 0, f"Vi phạm Guardrail G4: Phát hiện từ cấm '{bw}' trong payload JSON API!"


# -----------------------------------------------------------------------------
# G5: Strict API Contract
# -----------------------------------------------------------------------------
def test_guardrail_g5_api_contract_overlay_compatibility(sample_bytes: bytes):
    """
    Xác minh payload khớp hoàn hảo với Pydantic MeasurementPayload và Canvas drawer của Frontend.
    """
    response = client.post(
        "/api/v1/screening/analyze",
        files={"file": ("g5_test.jpg", io.BytesIO(sample_bytes), "image/jpeg")},
    )
    assert response.status_code == 200
    data = response.json()

    # Xác thực qua Pydantic schema
    validated = MeasurementPayload.model_validate(data)
    assert validated.session_id is not None

    # Xác thực định dạng tọa độ Canvas
    coords = data["annotated_overlay_coordinates"]
    for side in ["right_eye", "left_eye"]:
        eye = coords[side]
        assert len(eye["corneal_reflex"]) == 2
        assert len(eye["pupil_center"]) == 2
        assert len(eye["iris_limbus_contour"]) == 2
        assert len(eye["upper_eyelid_contour"]) >= 10
        if eye["lower_eyelid_contour"] is not None:
            assert len(eye["lower_eyelid_contour"]) >= 10
