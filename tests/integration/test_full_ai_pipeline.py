"""
Kiểm thử tích hợp toàn diện quy trình AI (Full AI Pipeline Integration Test - Phase 6).
Kiểm tra luồng xử lý khép kín các mô hình học sâu tái sử dụng từ V2:
Eye ROI đa chế độ -> Iris Geometry & HVID -> Phân đoạn EyelidUNet -> Định vị Phản quang Purkinje/Đồng tử -> Đo lường MRD1 -> Phân tầng Nguy cơ.
Tuân thủ G2: 100% trong RAM, không ghi đĩa.
Tuân thủ G4: Tuyệt đối không dùng từ ngữ thuộc danh mục cấm.
Tuân thủ G5: Giữ nguyên cấu trúc MeasurementPayload tương thích giao diện Canvas Frontend.
"""

import io
from pathlib import Path
import time
import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app
from tests.test_helpers import get_sample_image_path

client = TestClient(app)
SAMPLE_IMAGE_PATH = get_sample_image_path()


@pytest.fixture
def sample_image_bytes() -> bytes:
    """Tải dữ liệu byte ảnh mẫu."""
    assert SAMPLE_IMAGE_PATH.exists(), f"Không tìm thấy ảnh mẫu tại: {SAMPLE_IMAGE_PATH}"
    with open(SAMPLE_IMAGE_PATH, "rb") as f:
        return f.read()


def test_full_ai_pipeline_fullface_mode(sample_image_bytes: bytes):
    """
    Kiểm tra toàn trình ở chế độ ảnh toàn mặt ('fullface'):
    - Endpoint phản hồi mã HTTP 200.
    - Thời gian thực thi toàn trình <= 3.0 giây trên CPU.
    - Cấu trúc MeasurementPayload đầy đủ mọi trường số đo và thông số mống mắt.
    - Tọa độ lớp phủ Canvas (AnnotatedOverlayContainer) sẵn sàng cho Frontend vẽ.
    """
    start_time = time.perf_counter()

    response = client.post(
        "/api/v1/screening/analyze",
        data={"image_type": "fullface"},
        files={"file": ("test_fullface.jpg", io.BytesIO(sample_image_bytes), "image/jpeg")},
    )

    elapsed_time = time.perf_counter() - start_time
    assert response.status_code == 200, f"Mã lỗi HTTP: {response.status_code} - {response.text}"
    assert elapsed_time <= 3.0, f"Thời gian xử lý ({elapsed_time:.2f}s) vượt quá NFR-001 (<= 3.0s)"

    data = response.json()

    # 1. Định danh phiên và thời gian
    assert "session_id" in data and len(data["session_id"]) > 0
    assert "timestamp" in data

    # 2. Cổng chất lượng hình ảnh
    audit = data["image_quality_audit"]
    assert audit["passed"] is True
    assert "pose" in audit
    assert audit["sharpness_score"] > 0
    assert audit["mean_illumination"] > 0

    # 3. Thước đo mống mắt chuẩn HVID
    scale = data["scale_reference"]
    assert scale["assumed_corneal_diameter_mm"] == 11.71
    assert scale["hvid_pixels"]["right_eye"] > 20.0
    assert scale["hvid_pixels"]["left_eye"] > 20.0
    assert scale["scale_factor_mm_per_px"]["right_eye"] > 0.0
    assert scale["scale_factor_mm_per_px"]["left_eye"] > 0.0

    # 4. Số đo hình thái hai mắt OD và OS
    meas = data["measurements"]
    od = meas["right_eye_od"]
    os_eye = meas["left_eye_os"]

    assert od["mrd1_mm"] is not None
    assert os_eye["mrd1_mm"] is not None
    assert " - " in od["mrd1_estimated_range"]
    assert " - " in os_eye["mrd1_estimated_range"]
    assert od["status"] in ["VALID", "LOW_CONFIDENCE"]
    assert os_eye["status"] in ["VALID", "LOW_CONFIDENCE"]
    assert od["confidence_score"] >= 0.50
    assert os_eye["confidence_score"] >= 0.50
    assert od["pfh_mm"] is not None and od["pfh_mm"] > 0.0
    assert os_eye["pfh_mm"] is not None and os_eye["pfh_mm"] > 0.0
    assert od["reference_point_used"] in ["CORNEAL_LIGHT_REFLEX", "PUPIL_CENTER_FALLBACK"]
    assert os_eye["reference_point_used"] in ["CORNEAL_LIGHT_REFLEX", "PUPIL_CENTER_FALLBACK"]

    # 5. So sánh đối xứng hai bên
    comp = data["bilateral_comparison"]
    assert comp["delta_mrd1_mm"] >= 0.0
    assert comp["asymmetry_classification"] in [
        "PHYSIOLOGICAL_SYMMETRY",
        "BORDERLINE_ASYMMETRY",
        "SIGNIFICANT_ASYMMETRY",
    ]
    assert comp["delta_brow_px"] is not None

    # 6. Tọa độ lớp phủ trực quan Canvas Frontend
    coords = data["annotated_overlay_coordinates"]
    assert "right_eye" in coords and "left_eye" in coords

    # Phản quang / điểm tham chiếu
    assert len(coords["right_eye"]["corneal_reflex"]) == 2
    assert len(coords["left_eye"]["corneal_reflex"]) == 2

    # Tâm mống mắt / đồng tử
    assert len(coords["right_eye"]["pupil_center"]) == 2
    assert len(coords["left_eye"]["pupil_center"]) == 2

    # Đường viền mống mắt (2 điểm cực ngang để Frontend tính bán kính và vẽ hình tròn)
    assert len(coords["right_eye"]["iris_limbus_contour"]) == 2
    assert len(coords["left_eye"]["iris_limbus_contour"]) == 2

    # Đường viền bờ mi trên và mi dưới (mịn, liên tục qua nội suy spline)
    assert len(coords["right_eye"]["upper_eyelid_contour"]) >= 10
    assert len(coords["left_eye"]["upper_eyelid_contour"]) >= 10
    assert len(coords["right_eye"]["lower_eyelid_contour"]) >= 10
    assert len(coords["left_eye"]["lower_eyelid_contour"]) >= 10

    # 7. Kết quả phân tầng nguy cơ sàng lọc
    triage = data["triage_result"]
    assert triage["tier"] in ["TIER_A", "TIER_B", "TIER_C"]
    assert len(triage["trigger_reasons"]) > 0
    assert len(triage["description_vi"]) > 0
    assert len(triage["recommendation_vi"]) > 0
    assert "chỉ mang tính sàng lọc" in triage["safety_disclaimer_vi"]
    assert "không thay thế thăm khám" in triage["safety_disclaimer_vi"]


def test_full_ai_pipeline_eye_crop_mode(sample_image_bytes: bytes):
    """
    Kiểm tra toàn trình ở chế độ ảnh cận cảnh hai mắt ('eye_crop'):
    - Hoạt động ổn định qua thuật toán Haar Cascade kết hợp eye_crop_seed.
    - Không bị kích hoạt lỗi từ chối khuôn mặt (NO_FACE_DETECTED).
    - Trả về payload hoàn chỉnh với số đo chính xác.
    """
    # Tạo ảnh crop cận cảnh vùng 2 mắt từ ảnh mẫu
    nparr = np.frombuffer(sample_image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    h, w = img.shape[:2]

    # Cắt phần nửa trên chứa hai mắt
    eye_strip = img[int(h * 0.2):int(h * 0.6), int(w * 0.1):int(w * 0.9)]
    _, encoded_crop = cv2.imencode(".jpg", eye_strip)
    crop_bytes = encoded_crop.tobytes()

    response = client.post(
        "/api/v1/screening/analyze",
        data={"image_type": "eye_crop"},
        files={"file": ("test_crop.jpg", io.BytesIO(crop_bytes), "image/jpeg")},
    )

    assert response.status_code == 200, f"Mã lỗi HTTP: {response.status_code} - {response.text}"
    data = response.json()

    assert data["image_quality_audit"]["passed"] is True
    assert data["measurements"]["right_eye_od"]["mrd1_mm"] is not None
    assert data["measurements"]["left_eye_os"]["mrd1_mm"] is not None
    assert data["scale_reference"]["hvid_pixels"]["right_eye"] > 10.0


def test_full_ai_pipeline_zero_disk_write(sample_image_bytes: bytes):
    """
    Kiểm tra Guardrail G2 (In-Memory Zero-Retention):
    Khẳng định không có bất kỳ tệp ảnh tạm nào được ghi xuống đĩa trong toàn bộ quy trình AI.
    """
    backend_path = Path("app") if Path("app").exists() else Path("backend/app")
    files_before = set(backend_path.rglob("*.jpg")) | set(backend_path.rglob("*.png")) | set(backend_path.rglob("*.tmp"))

    response = client.post(
        "/api/v1/screening/analyze",
        files={"file": ("test_g2.jpg", io.BytesIO(sample_image_bytes), "image/jpeg")},
    )
    assert response.status_code == 200

    files_after = set(backend_path.rglob("*.jpg")) | set(backend_path.rglob("*.png")) | set(backend_path.rglob("*.tmp"))
    assert files_before == files_after, "Phát hiện tệp tin mới phát sinh trên ổ đĩa vi phạm Guardrail G2!"


def test_full_ai_pipeline_empty_image_rejection():
    """Kiểm tra từ chối hợp thức khi tải lên tệp tin rỗng."""
    response = client.post(
        "/api/v1/screening/analyze",
        files={"file": ("empty.jpg", io.BytesIO(b""), "image/jpeg")},
    )
    assert response.status_code == 400
    assert "trống hoặc không hợp lệ" in response.json()["detail"]
