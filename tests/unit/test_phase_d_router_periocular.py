"""
Bộ kiểm thử đơn vị cho Phase D: Dual-Pipeline Input Router & Periocular Pipeline.
Xác minh:
1. Bộ điều tuyến Input Router phân loại chính xác Toàn mặt (FULL_FACE), Cận cảnh hai mắt (DUAL_EYE_CROP), hoặc Không hỗ trợ (UNSUPPORTED).
2. Bộ tiếp hợp Periocular Adapter trích xuất 478 mốc giải phẫu qua Synthetic Canvas Injection và ánh xạ ngược chính xác.
3. Cổng Sàng lọc Chất lượng Periocular Quality Gate áp dụng kiểm định tư thế 2D (|Roll| <= 7.0 deg) và độ phân giải crop.
4. Endpoint /api/v1/screening/analyze cứu thành công các fixtures cận cảnh hai mắt (HTTP 200).
Tuân thủ G2: Xử lý 100% trong RAM, không ghi đĩa.
Tuân thủ G4: Tuyệt đối không dùng từ ngữ thuộc danh mục cấm.
"""

import io
from pathlib import Path
import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services.input_router_service import InputType, classify_input_image
from app.services.periocular_adapter_service import process_periocular_crop
from app.services.quality_gate_service import evaluate_periocular_quality
from app.schemas.measurement import MeasurementPayload

client = TestClient(app)
FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"


def _read_fixture(filename: str) -> np.ndarray:
    path = FIXTURES_DIR / filename
    assert path.exists(), f"Fixture {filename} không tồn tại tại {path}"
    img = cv2.imread(str(path))
    assert img is not None, f"Không thể giải mã {filename}"
    return img


# ==============================================================================
# 1. KIỂM THỬ BỘ ĐIỀU TUYẾN INPUT ROUTER (CASSETTED ROUTER PER RISK 1)
# ==============================================================================

def test_router_full_face_classification():
    """Kiểm tra Router phân loại ảnh toàn mặt chuẩn thành FULL_FACE (Direct Probe)."""
    img = _read_fixture("sample_face.jpg")
    input_type, meta = classify_input_image(img)
    assert input_type == InputType.FULL_FACE
    assert meta["method"] == "FACEMESH_DIRECT"
    assert meta["landmarks_count"] >= 478


def test_router_dual_eye_crop_classification():
    """Kiểm tra Router phân loại ảnh crop hai mắt thành DUAL_EYE_CROP (Canvas Injection)."""
    img = _read_fixture("crop_eye_normal_sample.jpg")
    input_type, meta = classify_input_image(img)
    assert input_type == InputType.DUAL_EYE_CROP
    assert meta["method"] == "CANVAS_INJECTION"
    assert "iris_od" in meta
    assert "iris_os" in meta
    # Mắt phải (OD) có hoành độ x nhỏ hơn mắt trái (OS)
    assert meta["iris_od"][0] < meta["iris_os"][0]


def test_router_unsupported_noise_image():
    """Kiểm tra Router từ chối ảnh nhiễu ngẫu nhiên hoặc ảnh không chứa mắt."""
    noise_img = np.random.randint(50, 180, (400, 600, 3), dtype=np.uint8)
    input_type, meta = classify_input_image(noise_img)
    assert input_type == InputType.UNSUPPORTED
    assert meta["reason"] == "NO_FACE_OR_EYES_DETECTED"


def test_router_empty_image():
    """Kiểm tra Router xử lý an toàn mảng rỗng hoặc None."""
    empty_img = np.zeros((0, 0, 3), dtype=np.uint8)
    input_type, meta = classify_input_image(empty_img)
    assert input_type == InputType.UNSUPPORTED


# ==============================================================================
# 2. KIỂM THỬ PERIOCULAR ADAPTER & ÁNH XẠ MỐC GIẢI PHẪU (M3-CROP)
# ==============================================================================

def test_periocular_adapter_landmarks_remapping():
    """Kiểm tra Periocular Adapter trích xuất mốc và ánh xạ ngược chính xác về ảnh crop."""
    img = _read_fixture("crop_eye_normal_sample.jpg")
    h, w = img.shape[:2]

    landmarks = process_periocular_crop(img)
    assert len(landmarks.landmarks_px) >= 478
    assert "PROCESSED_VIA_PERIOCULAR_PIPELINE" in landmarks.warning_flags

    # Kiểm tra tọa độ tâm mống mắt OD và OS nằm trong không gian ảnh crop
    x_od, y_od = landmarks.iris_center_od
    x_os, y_os = landmarks.iris_center_os

    assert 0 <= x_od < w
    assert 0 <= y_od < h
    assert 0 <= x_os < w
    assert 0 <= y_os < h
    assert x_od < x_os  # OD bên trái người quan sát

    # HVID phải dương và hợp lý (> 50 px với ảnh crop lớn)
    assert landmarks.hvid_od_px > 50.0
    assert landmarks.hvid_os_px > 50.0
    assert 0.015 <= landmarks.scale_factor_od <= 0.60
    assert 0.015 <= landmarks.scale_factor_os <= 0.60


# ==============================================================================
# 3. KIỂM THỬ PERIOCULAR QUALITY GATE (M2-CROP / RISK 2 & RISK 3)
# ==============================================================================

def test_periocular_quality_gate_passes_normal_crop():
    """Kiểm tra ảnh crop bình thường vượt qua Cổng Chất lượng Periocular."""
    img = _read_fixture("crop_eye_normal_sample.jpg")
    landmarks = process_periocular_crop(img)
    result = evaluate_periocular_quality(img, landmarks)

    assert result.passed
    assert result.rejection_reason is None
    assert result.pose is not None
    assert abs(result.pose.roll) <= 7.0
    assert result.sharpness_score >= 20.0
    assert result.audit_details.get("pipeline") == "PERIOCULAR_PIPELINE"


def test_periocular_quality_gate_rejects_closed_eyes():
    """Kiểm tra Cổng Chất lượng Periocular từ chối ảnh crop nhắm mi (EYES_CLOSED)."""
    img = _read_fixture("crop_face_eyes_closed_sample.jpg")
    landmarks = process_periocular_crop(img)
    result = evaluate_periocular_quality(img, landmarks)

    assert not result.passed
    assert result.rejection_reason == "EYES_CLOSED"
    assert "nhắm hoặc nheo" in result.rejection_message_vi


def test_periocular_quality_gate_rejects_motion_blur():
    """Kiểm tra Cổng Chất lượng Periocular từ chối ảnh crop mờ nhòe (BLUR_DETECTED)."""
    img = _read_fixture("crop_eye_motion_blurred_sample.jpg")
    landmarks = process_periocular_crop(img)
    result = evaluate_periocular_quality(img, landmarks)

    assert not result.passed
    # Bị từ chối bởi BLUR_DETECTED hoặc HEAD_ROLL_EXCEEDED
    assert result.rejection_reason in ["BLUR_DETECTED", "HEAD_ROLL_EXCEEDED"]


def test_periocular_quality_gate_rejects_low_resolution_crop():
    """Kiểm tra Cổng Chất lượng Periocular từ chối ảnh crop có chiều cao < 180 px."""
    img = _read_fixture("patient_31_image.jpg")
    h, w = img.shape[:2]
    assert h < 180  # Height 175 px
    landmarks = process_periocular_crop(img)
    result = evaluate_periocular_quality(img, landmarks)

    assert not result.passed
    assert result.rejection_reason == "LOW_RESOLUTION"
    assert "quá nhỏ" in result.rejection_message_vi


# ==============================================================================
# 4. KIỂM THỬ ENDPOINT TOÀN TRÌNH (/api/v1/screening/analyze)
# ==============================================================================

def test_endpoint_analyzes_crop_eye_normal_sample():
    """Kiểm tra endpoint /analyze phân tích thành công ảnh crop hai mắt (HTTP 200)."""
    path = FIXTURES_DIR / "crop_eye_normal_sample.jpg"
    with open(path, "rb") as f:
        file_bytes = f.read()

    response = client.post(
        "/api/v1/screening/analyze",
        files={"file": ("crop_eye_normal.jpg", io.BytesIO(file_bytes), "image/jpeg")},
    )
    assert response.status_code == 200
    data = response.json()
    payload = MeasurementPayload.model_validate(data)

    assert payload.image_quality_audit.passed
    assert payload.image_quality_audit.audit_details.get("input_type") == "DUAL_EYE_CROP"
    assert "PROCESSED_VIA_PERIOCULAR_PIPELINE" in payload.measurements.right_eye_od.warning_flags
    assert "PROCESSED_VIA_PERIOCULAR_PIPELINE" in payload.measurements.left_eye_os.warning_flags
    assert payload.measurements.right_eye_od.mrd1_mm is not None
    assert payload.measurements.left_eye_os.mrd1_mm is not None


def test_endpoint_rescues_tight_eye_crop_sample():
    """
    Kiểm tra cứu thành công tight_eye_crop_sample.jpg (chiều cao 198 px, AR=8.57):
    Trước đây bị từ chối 422 do NO_FACE_DETECTED / LOW_RESOLUTION (Height < 200).
    Nay vượt qua thành công với HTTP 200.
    """
    path = FIXTURES_DIR / "tight_eye_crop_sample.jpg"
    with open(path, "rb") as f:
        file_bytes = f.read()

    response = client.post(
        "/api/v1/screening/analyze",
        files={"file": ("tight_eye_crop.jpg", io.BytesIO(file_bytes), "image/jpeg")},
    )
    assert response.status_code == 200
    data = response.json()
    payload = MeasurementPayload.model_validate(data)

    assert payload.image_quality_audit.passed
    assert payload.image_quality_audit.audit_details.get("input_type") == "DUAL_EYE_CROP"
    assert "PROCESSED_VIA_PERIOCULAR_PIPELINE" in payload.measurements.right_eye_od.warning_flags


def test_endpoint_rescues_patient_26_image():
    """
    Kiểm tra cứu thành công patient_26_image.jpg qua cấu hình Compact-Wide Canvas:
    Trước đây bị từ chối 422 do NO_FACE_DETECTED.
    Nay vượt qua thành công với HTTP 200.
    """
    path = FIXTURES_DIR / "patient_26_image.jpg"
    with open(path, "rb") as f:
        file_bytes = f.read()

    response = client.post(
        "/api/v1/screening/analyze",
        files={"file": ("patient_26.jpg", io.BytesIO(file_bytes), "image/jpeg")},
    )
    assert response.status_code == 200
    data = response.json()
    payload = MeasurementPayload.model_validate(data)

    assert payload.image_quality_audit.passed
    assert payload.image_quality_audit.audit_details.get("input_type") == "DUAL_EYE_CROP"


def test_endpoint_corneal_reflex_occluded_crop_rescued():
    """
    Kiểm tra corneal_reflex_fully_occluded_sample.png vượt qua nhận diện khuôn mặt
    thay vì bị từ chối cứng 422 NO_FACE_DETECTED.
    """
    path = FIXTURES_DIR / "corneal_reflex_fully_occluded_sample.png"
    with open(path, "rb") as f:
        file_bytes = f.read()

    response = client.post(
        "/api/v1/screening/analyze",
        files={"file": ("occluded.png", io.BytesIO(file_bytes), "image/png")},
    )
    # Ảnh được phân tích thành công (HTTP 200) thay vì bị từ chối 422 NO_FACE_DETECTED
    assert response.status_code == 200
    data = response.json()
    payload = MeasurementPayload.model_validate(data)
    assert payload.image_quality_audit.passed
    assert payload.image_quality_audit.audit_details.get("input_type") == "DUAL_EYE_CROP"
