"""
Kiểm thử chuyên sâu chốt chặn an toàn kiểm soát mép khung hình (Ocular Region Touches Image Edge).
Tuân thủ G2 (In-Memory Zero-Retention) và G4 (Non-Diagnostic Safety).
"""

import io
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.core.config import settings
from app.main import app
from app.services.quality_gate_service import (
    LANDMARK_IRIS_CENTER_OD,
    LANDMARK_IRIS_CENTER_OS,
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


def get_base_test_image(width: int = 640, height: int = 480) -> np.ndarray:
    """Tạo ảnh xám nhân tạo có độ tương phản và độ sáng chuẩn trong bộ nhớ RAM."""
    np.random.seed(42)
    img = np.random.randint(100, 150, (height, width, 3), dtype=np.uint8)
    return img


@pytest.mark.parametrize(
    "landmark_idx, test_x, test_y, boundary_name",
    [
        (LANDMARK_RIGHT_EYE_OUTER, 1.0, 240.0, "mép trái x <= 2"),
        (LANDMARK_LEFT_EYE_OUTER, 639.0, 240.0, "mép phải x >= 638"),
        (LANDMARK_RIGHT_EYE_TOP, 320.0, 1.0, "mép trên y <= 2"),
        (LANDMARK_LEFT_EYE_BOTTOM, 320.0, 479.0, "mép dưới y >= 478"),
        (LANDMARK_IRIS_CENTER_OD, 2.0, 240.0, "ngay tại biên chạm margin x = 2"),
        (LANDMARK_IRIS_CENTER_OS, 320.0, 478.0, "ngay tại biên chạm margin y = 478"),
    ],
)
def test_ocular_landmarks_touching_any_edge_rejected(
    landmark_idx: int,
    test_x: float,
    test_y: float,
    boundary_name: str,
):
    """Kiểm tra mọi mốc mắt khi chạm hoặc nằm sát mép (<= 2 px) đều bị từ chối OCULAR_REGION_TOUCHES_IMAGE_EDGE."""
    img = get_base_test_image(640, 480)
    w, h = 640.0, 480.0

    with patch("app.services.quality_gate_service._FACE_MESH.process") as mock_process, \
         patch("app.services.quality_gate_service.estimate_head_pose_3d") as mock_pose:

        mock_pose.return_value = (0.0, 0.0, 0.0)
        mock_lms = [MagicMock(x=0.5, y=0.5) for _ in range(478)]

        # Đặt mốc cụ thể vào tọa độ chạm viền
        mock_lms[landmark_idx] = MagicMock(x=test_x / w, y=test_y / h)
        mock_process.return_value = MagicMock(
            multi_face_landmarks=[MagicMock(landmark=mock_lms)]
        )

        result = evaluate_quality(img)

        assert not result.passed, f"Lỗi: Không phát hiện vi phạm chạm biên cho {boundary_name}"
        assert result.rejection_reason == "OCULAR_REGION_TOUCHES_IMAGE_EDGE"
        assert "sát mép ảnh" in result.rejection_message_vi
        assert result.audit_details.get("edge_margin_px") == settings.OCULAR_EDGE_MARGIN_PX


def test_ocular_landmarks_safe_clearance_passed():
    """Kiểm tra khi các mốc mắt cách mép ảnh an toàn (> 2 px), chốt chặn biên không từ chối."""
    img = get_base_test_image(640, 480)
    w, h = 640.0, 480.0

    with patch("app.services.quality_gate_service._FACE_MESH.process") as mock_process, \
         patch("app.services.quality_gate_service.estimate_head_pose_3d") as mock_pose:

        mock_pose.return_value = (0.0, 0.0, 0.0)
        # Tất cả các mốc mắt đặt ở khoảng cách an toàn (cách mép >= 10 px)
        mock_lms = [MagicMock(x=0.5, y=0.5) for _ in range(478)]
        mock_lms[LANDMARK_RIGHT_EYE_OUTER] = MagicMock(x=10.0 / w, y=240.0 / h)
        mock_lms[LANDMARK_LEFT_EYE_OUTER] = MagicMock(x=(w - 10.0) / w, y=240.0 / h)

        mock_process.return_value = MagicMock(
            multi_face_landmarks=[MagicMock(landmark=mock_lms)]
        )

        result = evaluate_quality(img)

        # Lý do từ chối không được là chạm biên
        if not result.passed:
            assert result.rejection_reason != "OCULAR_REGION_TOUCHES_IMAGE_EDGE"
