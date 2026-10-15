"""
Kiểm thử đơn vị và tích hợp cho Cổng Sàng lọc Chất lượng Chuyên sâu (Module M2).
Bao phủ toàn bộ các kịch bản kiểm thử TC-2.1 đến TC-2.5 theo plan_implement.md.
Tuyệt đối tuân thủ G2 (In-Memory Zero-Retention) và G4 (Safety Ban List).
"""

import io
from pathlib import Path
from unittest.mock import MagicMock, patch
import cv2
from fastapi.testclient import TestClient
import numpy as np
import pytest

from app.core.config import settings
from app.main import app
from app.schemas.quality import QualityGateResult
from app.services.quality_gate_service import (
    LANDMARK_LEFT_EYE_BOTTOM,
    LANDMARK_LEFT_EYE_INNER,
    LANDMARK_LEFT_EYE_OUTER,
    LANDMARK_LEFT_EYE_TOP,
    LANDMARK_RIGHT_EYE_BOTTOM,
    LANDMARK_RIGHT_EYE_INNER,
    LANDMARK_RIGHT_EYE_OUTER,
    LANDMARK_RIGHT_EYE_TOP,
    evaluate_quality,
)

client = TestClient(app)

from tests.test_helpers import get_sample_image_path
SAMPLE_IMAGE_PATH = get_sample_image_path()


def get_valid_test_image() -> np.ndarray:
    """Nạp hoặc tạo ảnh khuôn mặt hợp lệ trong bộ nhớ RAM."""
    if SAMPLE_IMAGE_PATH.exists():
        img = cv2.imread(str(SAMPLE_IMAGE_PATH))
        if img is not None:
            return img
    pytest.skip("Không tìm thấy tệp ảnh mẫu thực tế để thực hiện kiểm thử.")


def encode_image_to_bytes(image_bgr: np.ndarray, ext: str = ".jpg") -> bytes:
    """Mã hóa ảnh BGR thành chuỗi bytes trong RAM (không lưu đĩa)."""
    success, buffer = cv2.imencode(ext, image_bgr)
    assert success, "Mã hóa ảnh sang buffer nhị phân thất bại"
    return buffer.tobytes()


# ==============================================================================
# TC-2.1: Kiểm tra tư thế đầu vượt ngưỡng (Yaw > 10, Pitch > 10, Roll > 5)
# ==============================================================================

def test_tc2_1_head_roll_exceeded():
    """Kiểm tra từ chối khi góc nghiêng đầu roll > 5 độ."""
    img = get_valid_test_image()
    h, w = img.shape[:2]
    # Xoay ảnh 15 độ quanh tâm
    rot_mat = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), 15.0, 1.0)
    rotated_img = cv2.warpAffine(img, rot_mat, (w, h))

    # 1. Kiểm tra trực tiếp qua service
    result = evaluate_quality(rotated_img)
    assert not result.passed
    assert result.rejection_reason in ("HEAD_ROLL_EXCEEDED", "HEAD_YAW_EXCEEDED")
    assert result.rejection_message_vi is not None

    # 2. Kiểm tra qua endpoint HTTP API
    img_bytes = encode_image_to_bytes(rotated_img)
    response = client.post(
        "/api/v1/screening/analyze",
        files={"file": ("test_roll.jpg", io.BytesIO(img_bytes), "image/jpeg")},
    )
    assert response.status_code == 422
    data = response.json()
    detail = data.get("detail", data)
    assert detail["error_code"] == "QUALITY_GATE_REJECTED"
    assert detail["rejection_reason"] in ("HEAD_ROLL_EXCEEDED", "HEAD_YAW_EXCEEDED")


def test_tc2_1_head_yaw_exceeded_mock():
    """Kiểm tra từ chối khi góc quay đầu yaw > 10 độ."""
    img = get_valid_test_image()

    with patch("app.services.quality_gate_service.estimate_head_pose_3d") as mock_pose:
        # Giả lập góc yaw = 14.5 độ (> MAX_YAW_DEG 10.0)
        mock_pose.return_value = (2.0, 14.5, 1.0)

        result = evaluate_quality(img)
        assert not result.passed
        assert result.rejection_reason == "HEAD_YAW_EXCEEDED"
        assert "quay sang bên" in result.rejection_message_vi
        assert result.audit_details["yaw_deg"] == 14.5

    img_bytes = encode_image_to_bytes(img)
    with patch("app.services.quality_gate_service.estimate_head_pose_3d") as mock_pose:
        mock_pose.return_value = (2.0, 14.5, 1.0)
        response = client.post(
            "/api/v1/screening/analyze",
            files={"file": ("test_yaw.jpg", io.BytesIO(img_bytes), "image/jpeg")},
        )
        assert response.status_code == 422
        data = response.json()
        detail = data.get("detail", data)
        assert detail["rejection_reason"] == "HEAD_YAW_EXCEEDED"


def test_tc2_1_head_pitch_exceeded_mock():
    """Kiểm tra từ chối khi góc ngửa/cúi đầu pitch > 10 độ."""
    img = get_valid_test_image()

    with patch("app.services.quality_gate_service.estimate_head_pose_3d") as mock_pose:
        # Giả lập góc pitch = -13.2 độ (> MAX_PITCH_DEG 10.0)
        mock_pose.return_value = (-13.2, 1.5, 0.8)

        result = evaluate_quality(img)
        assert not result.passed
        assert result.rejection_reason == "HEAD_PITCH_EXCEEDED"
        assert "ngửa hoặc cúi" in result.rejection_message_vi


# ==============================================================================
# TC-2.2: Kiểm tra ảnh rung mờ (Laplacian variance < 50)
# ==============================================================================

def test_tc2_2_blur_detected():
    """Kiểm tra từ chối ảnh bị rung nhòe với phương sai Laplacian thấp."""
    img = get_valid_test_image()
    # Làm nhòe mạnh bằng GaussianBlur
    blurred_img = cv2.GaussianBlur(img, (61, 61), 20.0)

    # 1. Kiểm tra trực tiếp qua service
    result = evaluate_quality(blurred_img)
    assert not result.passed
    assert result.rejection_reason == "BLUR_DETECTED"
    assert "mờ nhòe" in result.rejection_message_vi
    assert result.sharpness_score < settings.LAPLACIAN_SHARP_THRESHOLD

    # 2. Kiểm tra qua endpoint HTTP API
    img_bytes = encode_image_to_bytes(blurred_img)
    response = client.post(
        "/api/v1/screening/analyze",
        files={"file": ("test_blur.jpg", io.BytesIO(img_bytes), "image/jpeg")},
    )
    assert response.status_code == 422
    data = response.json()
    detail = data.get("detail", data)
    assert detail["rejection_reason"] == "BLUR_DETECTED"
    assert "mờ nhòe" in detail["rejection_message_vi"]


# ==============================================================================
# TC-2.3: Kiểm tra ảnh quá tối (mean < 40) hoặc quá sáng (mean > 220)
# ==============================================================================

def test_tc2_3_too_dark():
    """Kiểm tra từ chối ảnh quá tối với độ sáng xám trung bình < 40."""
    img = get_valid_test_image()
    # Giảm độ sáng xuống mức thấp (trung bình ~ 20-30)
    dark_img = (img.astype(np.float32) * 0.15).astype(np.uint8)

    # 1. Kiểm tra trực tiếp qua service
    result = evaluate_quality(dark_img)
    assert not result.passed
    assert result.rejection_reason == "TOO_DARK"
    assert "quá tối" in result.rejection_message_vi

    # 2. Kiểm tra qua endpoint HTTP API
    img_bytes = encode_image_to_bytes(dark_img)
    response = client.post(
        "/api/v1/screening/analyze",
        files={"file": ("test_dark.jpg", io.BytesIO(img_bytes), "image/jpeg")},
    )
    assert response.status_code == 422
    data = response.json()
    detail = data.get("detail", data)
    assert detail["rejection_reason"] == "TOO_DARK"


def test_tc2_3_too_bright():
    """Kiểm tra từ chối ảnh cháy sáng quá mức với độ sáng xám trung bình > 220."""
    # Tạo ảnh trắng sáng toàn phần
    bright_img = np.full((400, 400, 3), 235, dtype=np.uint8)

    result = evaluate_quality(bright_img)
    assert not result.passed
    assert result.rejection_reason == "TOO_BRIGHT"
    assert "quá sáng" in result.rejection_message_vi

    img_bytes = encode_image_to_bytes(bright_img)
    response = client.post(
        "/api/v1/screening/analyze",
        files={"file": ("test_bright.jpg", io.BytesIO(img_bytes), "image/jpeg")},
    )
    assert response.status_code == 422
    data = response.json()
    detail = data.get("detail", data)
    assert detail["rejection_reason"] == "TOO_BRIGHT"


# ==============================================================================
# TC-2.4: Kiểm tra ảnh chuẩn đạt chất lượng (PASS -> HTTP 200)
# ==============================================================================

def test_tc2_4_valid_image_pass():
    """Kiểm tra ảnh chính diện, rõ nét, đủ ánh sáng vượt qua toàn bộ cổng chất lượng."""
    img = get_valid_test_image()

    # 1. Kiểm tra trực tiếp qua service
    result = evaluate_quality(img)
    assert result.passed
    assert result.rejection_reason is None
    assert result.pose is not None
    assert abs(result.pose.yaw) <= settings.MAX_YAW_DEG
    assert abs(result.pose.pitch) <= settings.MAX_PITCH_DEG
    assert abs(result.pose.roll) <= settings.MAX_ROLL_DEG
    assert result.sharpness_score >= settings.LAPLACIAN_SHARP_THRESHOLD
    assert settings.MIN_BRIGHTNESS <= result.mean_illumination <= settings.MAX_BRIGHTNESS

    # 2. Kiểm tra qua endpoint HTTP API
    img_bytes = encode_image_to_bytes(img)
    response = client.post(
        "/api/v1/screening/analyze",
        files={"file": ("test_valid.jpg", io.BytesIO(img_bytes), "image/jpeg")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["session_id"] is not None
    assert data["image_quality_audit"]["passed"] is True
    assert data["image_quality_audit"]["rejection_reason"] is None


# ==============================================================================
# TC-2.5: Kiểm tra ảnh nhắm mắt / chớp mắt (Aperture ratio < 0.15)
# ==============================================================================

def test_tc2_5_eyes_closed():
    """Kiểm tra từ chối khi đối tượng đang nhắm hoặc nheo mắt."""
    img = get_valid_test_image()
    h, w = img.shape[:2]

    # Giả lập mốc mi trên và mi dưới sát nhau (mắt nhắm: aperture ratio ~ 0.02)
    with patch("app.services.quality_gate_service._FACE_MESH.process") as mock_process, \
         patch("app.services.quality_gate_service.estimate_head_pose_3d") as mock_pose:

        mock_pose.return_value = (0.0, 0.0, 0.0)

        # Tạo mốc giả lập 478 điểm
        mock_landmarks = [MagicMock(x=0.5, y=0.5) for _ in range(478)]
        # Mắt phải: bề rộng = 100px (0.2 -> 0.4), chiều cao = 2px (0.3 -> 0.302) -> ratio = 0.02
        mock_landmarks[LANDMARK_RIGHT_EYE_OUTER] = MagicMock(x=200.0 / w, y=300.0 / h)
        mock_landmarks[LANDMARK_RIGHT_EYE_INNER] = MagicMock(x=300.0 / w, y=300.0 / h)
        mock_landmarks[LANDMARK_RIGHT_EYE_TOP] = MagicMock(x=250.0 / w, y=299.0 / h)
        mock_landmarks[LANDMARK_RIGHT_EYE_BOTTOM] = MagicMock(x=250.0 / w, y=301.0 / h)

        # Mắt trái: tương tự
        mock_landmarks[LANDMARK_LEFT_EYE_OUTER] = MagicMock(x=600.0 / w, y=300.0 / h)
        mock_landmarks[LANDMARK_LEFT_EYE_INNER] = MagicMock(x=500.0 / w, y=300.0 / h)
        mock_landmarks[LANDMARK_LEFT_EYE_TOP] = MagicMock(x=550.0 / w, y=299.0 / h)
        mock_landmarks[LANDMARK_LEFT_EYE_BOTTOM] = MagicMock(x=550.0 / w, y=301.0 / h)

        mock_process.return_value = MagicMock(
            multi_face_landmarks=[MagicMock(landmark=mock_landmarks)]
        )

        result = evaluate_quality(img)
        assert not result.passed
        assert result.rejection_reason == "EYES_CLOSED"
        assert "nhắm hoặc nheo" in result.rejection_message_vi

        # Kiểm tra qua API
        img_bytes = encode_image_to_bytes(img)
        response = client.post(
            "/api/v1/screening/analyze",
            files={"file": ("test_closed.jpg", io.BytesIO(img_bytes), "image/jpeg")},
        )
        assert response.status_code == 422
        data = response.json()
        detail = data.get("detail", data)
        assert detail["rejection_reason"] == "EYES_CLOSED"


# ==============================================================================
# Kiểm thử bổ sung: Không phát hiện mặt, độ phân giải thấp, không ghi đĩa
# ==============================================================================

def test_no_face_detected():
    """Kiểm tra từ chối khi không có khuôn mặt trong ảnh."""
    blank_img = np.full((300, 300, 3), 128, dtype=np.uint8)
    result = evaluate_quality(blank_img)
    assert not result.passed
    assert result.rejection_reason == "NO_FACE_DETECTED"
    assert "Không phát hiện khuôn mặt" in result.rejection_message_vi


def test_zero_disk_write_during_screening():
    """Kiểm tra G2: Toàn bộ quá trình đánh giá chất lượng diễn ra trong RAM, không tạo file đĩa."""
    img = get_valid_test_image()
    img_bytes = encode_image_to_bytes(img)

    # Đếm số lượng tệp trong thư mục app trước và sau khi gọi API (tránh quét qua node_modules/.venv)
    backend_path = Path("app") if Path("app").exists() else Path("/app/app")
    files_before = set(backend_path.rglob("*.jpg")) | set(backend_path.rglob("*.png"))

    response = client.post(
        "/api/v1/screening/analyze",
        files={"file": ("test_memory.jpg", io.BytesIO(img_bytes), "image/jpeg")},
    )
    assert response.status_code == 200

    files_after = set(backend_path.rglob("*.jpg")) | set(backend_path.rglob("*.png"))
    assert files_before == files_after, "Phát hiện tệp ảnh mới được ghi ra đĩa trong quá trình xử lý!"
