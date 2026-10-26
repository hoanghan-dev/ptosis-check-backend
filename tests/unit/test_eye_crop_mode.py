"""
Kiểm thử chuyên sâu cho tính năng Sàng lọc Chế độ Ảnh Crop Mắt (Module M2 & M3 - Eye Crop Mode).
Xác thực:
1. Chấp thuận ảnh crop 2 mắt hợp lệ qua Cổng chất lượng chuyên biệt (evaluate_quality với image_type='eye_crop').
2. Từ chối ảnh crop mắt bị nghiêng (HEAD_ROLL_EXCEEDED) dựa trên trục 2 tâm mống mắt.
3. Từ chối ảnh crop quá tối (TOO_DARK), quá mờ (BLUR_DETECTED), hoặc không nhận diện được 2 mắt (BILATERAL_EYES_NOT_DETECTED).
4. Xác thực chế độ ảnh toàn mặt (fullface) giữ nguyên 100% tính năng kiểm định khắt khe và không bị ảnh hưởng.
5. Kiểm thử tích hợp qua endpoint /api/v1/screening/analyze với image_type='eye_crop'.
Tuân thủ tuyệt đối G2 (In-Memory Zero-Retention) và G4 (Non-Diagnostic Safety Ban).
"""

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.core.config import settings
from app.main import app
from app.schemas.quality import QualityGateResult
from app.services.landmark_service import detect_landmarks
from app.services.quality_gate_service import evaluate_quality
from tests.test_helpers import get_sample_image_path

client = TestClient(app)
SAMPLE_IMAGE_PATH = get_sample_image_path()


def _get_realistic_eye_crop() -> np.ndarray:
    """Tạo vùng crop 2 mắt thực tế từ ảnh mẫu chuẩn."""
    img = cv2.imread(str(SAMPLE_IMAGE_PATH))
    h, w = img.shape[:2]
    # Tọa độ vùng 2 mắt: từ trán (dưới chân tóc) đến ngang sống mũi/chóp mũi
    # Cắt tương ứng khoảng 30% đến 58% chiều cao, 18% đến 82% chiều rộng
    y1, y2 = int(h * 0.30), int(h * 0.58)
    x1, x2 = int(w * 0.18), int(w * 0.82)
    return img[y1:y2, x1:x2].copy()


def test_eye_crop_quality_gate_valid():
    """Kiểm tra ảnh crop 2 mắt chuẩn đạt Cổng Chất lượng ở chế độ eye_crop."""
    crop = _get_realistic_eye_crop()
    res = evaluate_quality(crop, image_type="eye_crop")
    assert res.passed is True, f"Ảnh crop chuẩn phải đạt Quality Gate nhưng bị từ chối: {res.rejection_reason}"
    assert res.pose is not None
    assert abs(res.pose.roll) <= settings.MAX_ROLL_DEG
    assert res.audit_details.get("image_type") == "eye_crop"


def test_fullface_mode_strictly_rejects_eye_crop():
    """Xác thực chế độ 'fullface' giữ nguyên tính năng và kiên quyết từ chối ảnh chỉ có vùng mắt."""
    crop = _get_realistic_eye_crop()
    # Khi gọi chế độ mặc định fullface, vì thiếu cằm/miệng nên phải bị từ chối
    res_fullface = evaluate_quality(crop, image_type="fullface")
    assert res_fullface.passed is False
    assert res_fullface.rejection_reason in ["NO_FACE_DETECTED", "HEAD_POSE_ERROR"]


def test_eye_crop_rejects_tilted_roll():
    """Kiểm tra từ chối ảnh crop mắt bị nghiêng quá ngưỡng an toàn (Roll > 5 độ)."""
    crop = _get_realistic_eye_crop()
    ch, cw = crop.shape[:2]
    # Xoay ảnh 7.5 độ quanh tâm (vừa vượt ngưỡng 5 độ an toàn và mắt vẫn định vị được)
    rot_matrix = cv2.getRotationMatrix2D((cw / 2, ch / 2), 7.5, 1.0)
    tilted = cv2.warpAffine(crop, rot_matrix, (cw, ch), borderMode=cv2.BORDER_REFLECT)

    res = evaluate_quality(tilted, image_type="eye_crop")
    assert res.passed is False
    assert res.rejection_reason == "HEAD_ROLL_EXCEEDED"
    assert "nghiêng" in res.rejection_message_vi.lower()


def test_eye_crop_rejects_dark_image():
    """Kiểm tra từ chối ảnh crop mắt bị thiếu sáng (quá tối)."""
    crop = _get_realistic_eye_crop()
    dark = (crop * 0.1).astype(np.uint8)
    res = evaluate_quality(dark, image_type="eye_crop")
    assert res.passed is False
    assert res.rejection_reason == "TOO_DARK"


def test_eye_crop_rejects_blurry_image():
    """Kiểm tra từ chối ảnh crop mắt bị nhòe mờ do rung chuyển."""
    crop = _get_realistic_eye_crop()
    blurred = cv2.GaussianBlur(crop, (9, 9), 2.5)
    res = evaluate_quality(blurred, image_type="eye_crop")
    assert res.passed is False
    assert res.rejection_reason == "BLUR_DETECTED"


def test_eye_crop_rejects_image_without_eyes():
    """Kiểm tra từ chối ảnh cắt không chứa cặp mắt (ví dụ ảnh da trơn hoặc nền)."""
    blank = np.full((300, 500, 3), 128, dtype=np.uint8)
    res = evaluate_quality(blank, image_type="eye_crop")
    assert res.passed is False
    assert res.rejection_reason == "BILATERAL_EYES_NOT_DETECTED"


def test_eye_crop_endpoint_integration():
    """Kiểm thử tích hợp endpoint POST /api/v1/screening/analyze với tham số image_type='eye_crop'."""
    crop = _get_realistic_eye_crop()
    _, enc = cv2.imencode(".jpg", crop)
    file_bytes = enc.tobytes()

    response = client.post(
        "/api/v1/screening/analyze",
        files={"file": ("eye_crop_test.jpg", file_bytes, "image/jpeg")},
        data={"image_type": "eye_crop"},
    )

    assert response.status_code == 200, f"Endpoint trả về mã lỗi: {response.status_code}, nội dung: {response.text}"
    data = response.json()
    assert "measurements" in data
    assert "bilateral_comparison" in data
    assert "triage_result" in data
    assert data["measurements"]["right_eye_od"]["mrd1_mm"] is not None
    assert data["measurements"]["left_eye_os"]["mrd1_mm"] is not None
    assert data["triage_result"]["tier"] in ["TIER_A", "TIER_B", "TIER_C"]


def _get_tight_eye_crop() -> np.ndarray:
    """Tạo vùng crop 2 mắt cắt sát (chỉ có 2 mắt, không có chóp mũi/trán/cằm)."""
    img = cv2.imread(str(SAMPLE_IMAGE_PATH))
    h, w = img.shape[:2]
    # Cắt vừa khít dải mắt từ bờ chân mày đến bờ mi dưới
    y1, y2 = int(h * 0.38), int(h * 0.51)
    x1, x2 = int(w * 0.20), int(w * 0.80)
    return img[y1:y2, x1:x2].copy()


def test_tight_eye_crop_passes_via_deep_learning_fallback():
    """Kiểm thử ảnh crop mắt cắt rất sát (FaceMesh không bắt được mặt) vẫn đạt Quality Gate qua Deep Learning Fallback."""
    tight_crop = _get_tight_eye_crop()
    res = evaluate_quality(tight_crop, image_type="eye_crop")
    assert res.passed is True, f"Ảnh crop sát phải đạt chuẩn qua Fallback nhưng bị từ chối: {res.rejection_reason}"
    assert res.pose is not None
    assert abs(res.pose.roll) <= settings.MAX_ROLL_DEG


def test_tight_eye_crop_endpoint_integration():
    """Kiểm thử tích hợp sàng lọc hoàn chỉnh cho ảnh crop cắt sát qua endpoint."""
    tight_crop = _get_tight_eye_crop()
    _, enc = cv2.imencode(".jpg", tight_crop)
    file_bytes = enc.tobytes()

    response = client.post(
        "/api/v1/screening/analyze",
        files={"file": ("tight_crop_test.jpg", file_bytes, "image/jpeg")},
        data={"image_type": "eye_crop"},
    )

    assert response.status_code == 200, f"Endpoint trả về mã lỗi: {response.status_code}, nội dung: {response.text}"
    data = response.json()
    assert "measurements" in data
    assert "bilateral_comparison" in data
    assert data["measurements"]["right_eye_od"]["mrd1_mm"] > 0
    assert data["measurements"]["left_eye_os"]["mrd1_mm"] > 0
    assert "triage_result" in data
