"""
Kiểm thử đo lường độ chính xác sai số định vị tâm đồng tử sub-pixel (Pupil Detection Accuracy).
Tuân thủ G2 (In-Memory Zero-Retention) và G4 (Non-Diagnostic Safety).
"""

import math
import cv2
import numpy as np
import pytest

from app.core.config import settings
from app.services.pupil_service import (
    PupilResult,
    detect_pupil_in_iris_roi,
)


def create_synthetic_eye(
    width: int = 160,
    height: int = 160,
    iris_center: tuple = (80.0, 80.0),
    iris_radius: float = 40.0,
    pupil_offset: tuple = (0.0, 0.0),
    pupil_radius: float = 16.0,
    sclera_luma: int = 210,
    iris_luma: int = 90,
    pupil_luma: int = 20,
) -> np.ndarray:
    """Tạo ảnh nhân tạo 3 kênh BGR trong bộ nhớ RAM."""
    img = np.full((height, width, 3), sclera_luma, dtype=np.uint8)
    icx, icy = int(iris_center[0]), int(iris_center[1])

    # Vẽ mống mắt
    cv2.circle(img, (icx, icy), int(iris_radius), (iris_luma, iris_luma, iris_luma), -1, lineType=cv2.LINE_AA)

    # Vẽ đồng tử
    pcx = int(round(iris_center[0] + pupil_offset[0]))
    pcy = int(round(iris_center[1] + pupil_offset[1]))
    cv2.circle(img, (pcx, pcy), int(pupil_radius), (pupil_luma, pupil_luma, pupil_luma), -1, lineType=cv2.LINE_AA)

    return img


@pytest.mark.parametrize(
    "offset_x, offset_y, iris_luma, pupil_luma",
    [
        (0.0, 0.0, 90, 20),      # Tâm chuẩn, độ tương phản cao
        (1.0, -1.0, 80, 25),     # Lệch nhẹ góc trên-phải, mống mắt sẫm
        (-1.5, 1.5, 110, 30),    # Lệch nhẹ góc dưới-trái, mống mắt sáng
        (0.5, 1.5, 95, 22),      # Lệch nhẹ hướng dưới
        (-1.0, -1.0, 85, 18),    # Lệch nhẹ hướng trên-trái
    ],
)
def test_pupil_detection_accuracy_subpixel_error(
    offset_x: float,
    offset_y: float,
    iris_luma: int,
    pupil_luma: int,
):
    """Xác nhận độ lệch Euclidean giữa tâm phát hiện và ground-truth luôn <= 1.0 px."""
    iris_center = (80.0, 80.0)
    iris_radius = 40.0
    pupil_radius = 16.0

    gt_x = round(iris_center[0] + offset_x)
    gt_y = round(iris_center[1] + offset_y)

    img = create_synthetic_eye(
        iris_center=iris_center,
        iris_radius=iris_radius,
        pupil_offset=(offset_x, offset_y),
        pupil_radius=pupil_radius,
        iris_luma=iris_luma,
        pupil_luma=pupil_luma,
    )

    result = detect_pupil_in_iris_roi(img, iris_center, iris_radius)

    assert result.status == "DETECTED"
    assert result.pupil_point is not None

    det_x, det_y = result.pupil_point
    euclidean_error = math.hypot(det_x - gt_x, det_y - gt_y)

    assert euclidean_error <= 1.0, (
        f"Sai số định vị đồng tử ({euclidean_error:.3f} px) vượt quá ngưỡng 1.0 px "
        f"cho offset=({offset_x}, {offset_y}), ground_truth=({gt_x}, {gt_y}), detected=({det_x:.2f}, {det_y:.2f})"
    )
    assert result.circularity is not None and result.circularity >= 0.35
