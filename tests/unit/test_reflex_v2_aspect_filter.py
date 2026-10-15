"""
Kiểm thử chuyên sâu bộ lọc tỷ lệ trục vệt sáng giác mạc V2 (Aspect Ratio Filter).
Tuân thủ G2 (In-Memory Zero-Retention) và G4 (Non-Diagnostic Safety).
"""

import cv2
import numpy as np
import pytest

from app.core.config import settings
from app.services.reflex_service import (
    ReflexDetectionStatus,
    ReflexResult,
    detect_corneal_reflex,
)


def create_synthetic_iris_roi(
    width: int = 120,
    height: int = 120,
    iris_center: tuple = (60.0, 60.0),
    iris_radius: float = 30.0,
    iris_luma: int = 50,
    sclera_luma: int = 160,
) -> np.ndarray:
    """Tạo ảnh xám tổng hợp mô phỏng mống mắt và tròng trắng trong bộ nhớ RAM."""
    img = np.full((height, width), sclera_luma, dtype=np.uint8)
    cx, cy = int(iris_center[0]), int(iris_center[1])
    cv2.circle(img, (cx, cy), int(iris_radius), iris_luma, -1)
    return img


def test_reject_elongated_spot_aspect_3_5():
    """Kiểm tra loại bỏ đốm sáng dài có Aspect Ratio = 3.5, diện tích 12-14 px -> LARGE_OR_ELONGATED_GLARE."""
    iris_center = (60.0, 60.0)
    iris_radius = 30.0
    img = create_synthetic_iris_roi(iris_center=iris_center, iris_radius=iris_radius)

    # Đặt vệt sáng dài: chiều rộng 7 px, chiều cao 2 px (aspect_ratio = 7 / 2 = 3.5 > 2.8)
    # Diện tích = 14 px (nằm trong dải min_area=4 và max_area=60)
    img[59:61, 57:64] = 250

    result = detect_corneal_reflex(img, iris_center, iris_radius)

    assert result.has_fallback is True
    assert result.reference_point_used == "PUPIL_CENTER_FALLBACK"
    assert result.status == ReflexDetectionStatus.LARGE_OR_ELONGATED_GLARE
    assert "REFLEX_GLARE_ARTIFACT" in result.warning_flags
    assert "REFLEX_FALLBACK_TO_PUPIL" in result.warning_flags


def test_accept_circular_spot_aspect_1_2():
    """Kiểm tra chấp nhận đốm sáng tròn có Aspect Ratio ~ 1.2, diện tích ~ 10-12 px -> SUCCESS."""
    iris_center = (60.0, 60.0)
    iris_radius = 30.0
    img = create_synthetic_iris_roi(iris_center=iris_center, iris_radius=iris_radius)

    # Vẽ đốm tròn nhỏ bán kính 2 px tại tâm mống mắt (diện tích ~ 12 px)
    cv2.circle(img, (60, 60), 2, 255, -1)

    result = detect_corneal_reflex(img, iris_center, iris_radius)

    assert result.has_fallback is False
    assert result.reference_point_used == "CORNEAL_LIGHT_REFLEX"
    assert result.status == ReflexDetectionStatus.SUCCESS
    assert result.confidence_score == 1.0
    assert result.glint_aspect_ratio is not None
    assert result.glint_aspect_ratio <= 1.5
    # Tọa độ điểm phản quang phát hiện phải sát tâm (60, 60) với sai số < 1 px
    assert abs(result.reflex_point[0] - 60.0) < 1.0
    assert abs(result.reflex_point[1] - 60.0) < 1.0
