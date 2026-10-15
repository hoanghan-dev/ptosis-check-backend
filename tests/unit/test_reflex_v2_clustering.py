"""
Kiểm thử chuyên sâu chốt chặn phân cụm đốm sáng giác mạc V2 (Cluster Guard).
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
    width: int = 140,
    height: int = 140,
    iris_center: tuple = (70.0, 70.0),
    iris_radius: float = 35.0,
    iris_luma: int = 50,
    sclera_luma: int = 160,
) -> np.ndarray:
    """Tạo ảnh xám tổng hợp mô phỏng mống mắt trong bộ nhớ RAM."""
    img = np.full((height, width), sclera_luma, dtype=np.uint8)
    cx, cy = int(iris_center[0]), int(iris_center[1])
    cv2.circle(img, (cx, cy), int(iris_radius), iris_luma, -1)
    return img


def test_detect_cluster_candidates_within_distance():
    """Kiểm tra 2 đốm sáng cạnh tranh cách nhau 4.0 px (<= cluster limit 8.0 px) -> CLUSTERED_REFLEX_CANDIDATES."""
    iris_center = (70.0, 70.0)
    iris_radius = 35.0
    img = create_synthetic_iris_roi(iris_center=iris_center, iris_radius=iris_radius)

    # Đặt 2 đốm nhỏ cách nhau 4.0 px tại (68, 70) và (72, 70)
    img[70, 68] = 250
    img[70, 72] = 250

    result = detect_corneal_reflex(img, iris_center, iris_radius)

    assert result.has_fallback is True
    assert result.reference_point_used == "PUPIL_CENTER_FALLBACK"
    assert result.status == ReflexDetectionStatus.CLUSTERED_REFLEX_CANDIDATES
    assert "CLUSTERED_REFLEX_CANDIDATES" in result.warning_flags
    assert "REFLEX_FALLBACK_TO_PUPIL" in result.warning_flags


def test_candidates_separated_beyond_cluster_distance():
    """Kiểm tra 2 đốm sáng cách nhau 16.0 px (> 8.0 px) -> AMBIGUOUS_MULTI_REFLECTION (không phải CLUSTER)."""
    iris_center = (70.0, 70.0)
    iris_radius = 35.0
    img = create_synthetic_iris_roi(iris_center=iris_center, iris_radius=iris_radius)

    # Đặt 2 đốm sáng có kích thước tương đương cách nhau 16.0 px tại (62, 70) và (78, 70)
    # Khoảng cách d = 16 px > cluster_dist_limit = max(8.0, 0.2 * 35 = 7.0) = 8.0 px
    cv2.circle(img, (62, 70), 1, 248, -1)
    cv2.circle(img, (78, 70), 1, 248, -1)

    result = detect_corneal_reflex(img, iris_center, iris_radius)

    assert result.has_fallback is True
    assert result.reference_point_used == "PUPIL_CENTER_FALLBACK"
    # Phải rơi vào trạng thái AMBIGUOUS_MULTI_REFLECTION thay vì CLUSTERED_REFLEX_CANDIDATES
    assert result.status == ReflexDetectionStatus.AMBIGUOUS_MULTI_REFLECTION
    assert "AMBIGUOUS_MULTI_REFLECTION" in result.warning_flags
    assert "CLUSTERED_REFLEX_CANDIDATES" not in result.warning_flags
