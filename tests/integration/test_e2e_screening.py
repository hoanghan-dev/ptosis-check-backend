"""
Kiểm thử tích hợp toàn trình (E2E Integration Test) cho Hệ thống Sàng lọc PtosisCheck.
Kiểm tra luồng xử lý hoàn chỉnh từ lúc nhận ảnh Multipart -> Decode trong RAM ->
Cổng chất lượng M2 -> Mốc mống mắt M3 -> Phản quang M4 -> Bờ sụn mi M5 ->
Đo lường MRD1 M6 -> Phân tầng nguy cơ M7 -> Trả payload MeasurementPayload chuẩn.
Khẳng định thời gian phản hồi toàn trình <= 3.0 giây trên máy local (NFR-001).
Tuân thủ G2 (RAM Zero-Retention) và G4 (Non-Diagnostic Safety).
"""

import time
import pytest
from fastapi.testclient import TestClient

from app.main import app
from tests.test_helpers import get_sample_image_path

client = TestClient(app)
SAMPLE_IMAGE_PATH = get_sample_image_path()


def test_e2e_screening_pipeline_latency_and_contract():
    """Kiểm thử toàn trình: Gửi ảnh chuẩn, đo thời gian xử lý <= 3.0s và kiểm tra tính toàn vẹn của payload."""
    if not SAMPLE_IMAGE_PATH.exists():
        pytest.skip("Không tìm thấy tệp ảnh mẫu thực tế để thực hiện kiểm thử E2E.")

    with open(SAMPLE_IMAGE_PATH, "rb") as f:
        image_bytes = f.read()

    # Bắt đầu đo thời gian phản hồi toàn trình
    start_time = time.perf_counter()

    response = client.post(
        "/api/v1/screening/analyze",
        files={"file": ("test_benchmark.jpg", image_bytes, "image/jpeg")},
    )

    elapsed_time = time.perf_counter() - start_time

    # 1. Khẳng định mã phản hồi HTTP 200
    assert response.status_code == 200, f"Endpoint trả về mã lỗi: {response.status_code} - {response.text}"

    # 2. Khẳng định thời gian phản hồi toàn trình <= 3.0 giây (NFR-001)
    assert elapsed_time <= 3.0, f"Thời gian xử lý ({elapsed_time:.2f}s) vượt quá giới hạn 3.0s!"

    data = response.json()

    # 3. Kiểm định các thành phần của cấu trúc MeasurementPayload (G5 / Mục 15)
    assert "session_id" in data and len(data["session_id"]) > 0
    assert "timestamp" in data

    # Image Quality Audit
    audit = data["image_quality_audit"]
    assert audit["passed"] is True
    assert "pose" in audit
    assert "yaw" in audit["pose"] and "pitch" in audit["pose"] and "roll" in audit["pose"]
    assert audit["sharpness_score"] > 0
    assert audit["mean_illumination"] > 0

    # Scale Reference
    scale = data["scale_reference"]
    assert scale["assumed_corneal_diameter_mm"] == 11.71
    assert scale["hvid_pixels"]["right_eye"] > 0
    assert scale["hvid_pixels"]["left_eye"] > 0
    assert scale["scale_factor_mm_per_px"]["right_eye"] > 0
    assert scale["scale_factor_mm_per_px"]["left_eye"] > 0

    # Measurements Container
    meas = data["measurements"]
    od = meas["right_eye_od"]
    os_eye = meas["left_eye_os"]

    assert od["mrd1_mm"] > 0
    assert os_eye["mrd1_mm"] > 0
    assert " - " in od["mrd1_estimated_range"]
    assert " - " in os_eye["mrd1_estimated_range"]
    assert od["status"] in ["VALID", "LOW_CONFIDENCE"]
    assert os_eye["status"] in ["VALID", "LOW_CONFIDENCE"]
    assert od["confidence_score"] >= 0.50
    assert os_eye["confidence_score"] >= 0.50
    assert od["pfh_mm"] is not None and od["pfh_mm"] > 0
    assert os_eye["pfh_mm"] is not None and os_eye["pfh_mm"] > 0

    # Bilateral Comparison
    comp = data["bilateral_comparison"]
    assert comp["delta_mrd1_mm"] >= 0
    assert comp["asymmetry_classification"] in [
        "PHYSIOLOGICAL_SYMMETRY",
        "BORDERLINE_ASYMMETRY",
        "SIGNIFICANT_ASYMMETRY",
    ]
    assert comp["delta_brow_px"] is not None

    # Triage Result (M7)
    triage = data["triage_result"]
    assert triage["tier"] in ["TIER_A", "TIER_B", "TIER_C"]
    assert len(triage["trigger_reasons"]) > 0
    assert len(triage["description_vi"]) > 0
    assert len(triage["recommendation_vi"]) > 0
    assert "chỉ mang tính sàng lọc" in triage["safety_disclaimer_vi"]
    assert "không thay thế thăm khám" in triage["safety_disclaimer_vi"]

    # Annotated Overlay Coordinates (M8)
    coords = data["annotated_overlay_coordinates"]
    assert "right_eye" in coords and "left_eye" in coords
    assert len(coords["right_eye"]["corneal_reflex"]) == 2
    assert len(coords["left_eye"]["corneal_reflex"]) == 2
    assert len(coords["right_eye"]["iris_limbus_contour"]) == 2
    assert len(coords["right_eye"]["upper_eyelid_contour"]) > 5
    assert len(coords["left_eye"]["upper_eyelid_contour"]) > 5


def test_e2e_screening_dual_eye_crop_pipeline():
    """Kiểm thử toàn trình cho ảnh crop 2 mắt (Dual-Eye Crop) qua Periocular Pipeline:
    Xác nhận HTTP 200, độ trễ <= 3.0s, định tuyến đúng DUAL_EYE_CROP và cờ PROCESSED_VIA_PERIOCULAR_PIPELINE.
    """
    fixtures_dir = SAMPLE_IMAGE_PATH.parent
    crop_fixture = fixtures_dir / "crop_eye_normal_sample.jpg"
    if not crop_fixture.exists():
        pytest.skip("Không tìm thấy tệp ảnh crop_eye_normal_sample.jpg.")

    with open(crop_fixture, "rb") as f:
        image_bytes = f.read()

    start_time = time.perf_counter()
    response = client.post(
        "/api/v1/screening/analyze",
        files={"file": ("crop_eye_normal_sample.jpg", image_bytes, "image/jpeg")},
    )
    elapsed_time = time.perf_counter() - start_time

    assert response.status_code == 200, f"Crop pipeline thất bại: {response.status_code} - {response.text}"
    assert elapsed_time <= 3.0, f"Thời gian xử lý crop ({elapsed_time:.2f}s) vượt quá 3.0s!"

    data = response.json()
    audit = data["image_quality_audit"]
    assert audit["passed"] is True
    assert audit["audit_details"]["input_type"] == "DUAL_EYE_CROP"

    od_meas = data["measurements"]["right_eye_od"]
    os_meas = data["measurements"]["left_eye_os"]
    assert "PROCESSED_VIA_PERIOCULAR_PIPELINE" in od_meas["warning_flags"]
    assert "PROCESSED_VIA_PERIOCULAR_PIPELINE" in os_meas["warning_flags"]
    assert data["triage_result"]["tier"] in ["TIER_A", "TIER_B", "TIER_C"]

