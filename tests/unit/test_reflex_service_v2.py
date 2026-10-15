"""
Kiểm thử đơn vị chuyên sâu cho Dịch vụ Phản quang Giác mạc V2 (Module M4 - Phase 2).
Kiểm tra bộ lọc vệt lóa dài (Aspect Ratio <= 2.8), chốt chặn cụm đốm sáng (Cluster Guard <= 8 px),
hàm chấm điểm ứng viên đa biến (Multi-Objective Scoring), và đa phản xạ xa nhau.
Tuân thủ tuyệt đối G2 (In-Memory Zero-Retention) và G4 (Non-Diagnostic Safety).
"""

import math
import cv2
import numpy as np
import pytest

from app.core.config import settings
from app.services.reflex_service import (
    ReflexDetectionStatus,
    ReflexResult,
    detect_corneal_reflex,
)


def create_base_synthetic_eye(
    width: int = 140,
    height: int = 140,
    iris_center: tuple = (70.0, 70.0),
    iris_radius: float = 35.0,
    iris_intensity: int = 60,
    background_intensity: int = 160,
) -> np.ndarray:
    """Tạo ảnh xám nhân tạo trong RAM mô phỏng vùng mắt và mống mắt."""
    img = np.full((height, width), background_intensity, dtype=np.uint8)
    cx, cy = int(iris_center[0]), int(iris_center[1])
    cv2.circle(img, (cx, cy), int(iris_radius), iris_intensity, -1)
    return img


# ==============================================================================
# 1. Kiểm thử phát hiện vệt lóa dài (Elongated Glare Filter)
# ==============================================================================

def test_v2_elongated_glare_rejected():
    """Kiểm tra vệt sáng kéo dài (Aspect Ratio = 12 / 2 = 6.0 > 2.8) bị loại trừ và trả về LARGE_OR_ELONGATED_GLARE."""
    iris_center = (70.0, 70.0)
    iris_radius = 35.0
    img = create_base_synthetic_eye(iris_center=iris_center, iris_radius=iris_radius)

    # Vẽ vệt sáng dạng đường kẻ ngang dài 12 px, rộng 2 px quanh tâm (70, 70)
    # aspect_ratio = 12 / 2 = 6.0 > 2.8, area = 24 px >= 4
    img[69:71, 64:76] = 250

    result = detect_corneal_reflex(img, iris_center, iris_radius)

    assert result.has_fallback
    assert result.reference_point_used == "PUPIL_CENTER_FALLBACK"
    assert result.status == ReflexDetectionStatus.LARGE_OR_ELONGATED_GLARE
    assert "REFLEX_GLARE_ARTIFACT" in result.warning_flags
    assert "REFLEX_FALLBACK_TO_PUPIL" in result.warning_flags


def test_v2_excessive_area_glare_rejected():
    """Kiểm tra đốm sáng diện tích vượt quá ngưỡng tối đa bị loại trừ và trả về LARGE_OR_ELONGATED_GLARE."""
    iris_center = (70.0, 70.0)
    iris_radius = 35.0
    img = create_base_synthetic_eye(iris_center=iris_center, iris_radius=iris_radius)

    # Vẽ hình tròn lớn bán kính 6 px (diện tích ~ pi * 36 ≈ 113 px > REFLEX_MAX_AREA_PX = 60)
    cv2.circle(img, (70, 70), 6, 250, -1)

    result = detect_corneal_reflex(img, iris_center, iris_radius)

    assert result.has_fallback
    assert result.status == ReflexDetectionStatus.LARGE_OR_ELONGATED_GLARE
    assert "REFLEX_GLARE_ARTIFACT" in result.warning_flags


# ==============================================================================
# 2. Kiểm thử chốt chặn phân cụm ứng viên cạnh tranh (Cluster Guard)
# ==============================================================================

def test_v2_clustered_candidates_guard():
    """Kiểm tra 2 đốm sáng cạnh tranh sát nhau (cách nhau 5 px <= 8 px) bị bắt bởi Cluster Guard."""
    iris_center = (70.0, 70.0)
    iris_radius = 35.0
    img = create_base_synthetic_eye(iris_center=iris_center, iris_radius=iris_radius)

    # Đặt 2 đốm nhỏ cách nhau 5 px: đốm 1 tại (68, 70), đốm 2 tại (73, 70)
    # Khoảng cách d = 5 px <= cluster_dist_limit (8.0 px)
    img[70, 68] = 250
    img[70, 73] = 250

    result = detect_corneal_reflex(img, iris_center, iris_radius)

    assert result.has_fallback
    assert result.reference_point_used == "PUPIL_CENTER_FALLBACK"
    assert result.status == ReflexDetectionStatus.CLUSTERED_REFLEX_CANDIDATES
    assert "CLUSTERED_REFLEX_CANDIDATES" in result.warning_flags
    assert "REFLEX_FALLBACK_TO_PUPIL" in result.warning_flags


# ==============================================================================
# 3. Kiểm thử phản quang chuẩn (Single Sharp Reflex)
# ==============================================================================

def test_v2_standard_single_glint_success():
    """Kiểm tra đốm phản quang chuẩn đơn lẻ thu được trạng thái SUCCESS và glint_aspect_ratio hợp lệ."""
    iris_center = (70.0, 70.0)
    iris_radius = 35.0
    img = create_base_synthetic_eye(iris_center=iris_center, iris_radius=iris_radius)

    # Đặt đốm sáng tròn 3x3 pixel tại (71, 69)
    cv2.circle(img, (71, 69), 1, 252, -1)

    result = detect_corneal_reflex(img, iris_center, iris_radius)

    assert not result.has_fallback
    assert result.reference_point_used == "CORNEAL_LIGHT_REFLEX"
    assert result.status == ReflexDetectionStatus.SUCCESS
    assert result.confidence_score == settings.PURKINJE_NORMAL_CONFIDENCE
    assert result.glint_aspect_ratio is not None
    assert 1.0 <= result.glint_aspect_ratio <= 1.5
    assert result.warning_flags == []


# ==============================================================================
# 4. Kiểm thử đa phản quang xa nhau (Ambiguous Multi-Reflection)
# ==============================================================================

def test_v2_separated_ambiguous_multi_reflection():
    """Kiểm tra 2 đốm sáng có kích thước tương đương nhưng cách xa nhau (> 8 px) rơi vào AMBIGUOUS_MULTI_REFLECTION."""
    iris_center = (70.0, 70.0)
    iris_radius = 35.0
    img = create_base_synthetic_eye(iris_center=iris_center, iris_radius=iris_radius)

    # Đặt 2 đốm sáng tại (62, 70) và (78, 70)
    # Khoảng cách d = 16.0 px > 8.0 px (vượt ngưỡng cụm)
    # Cả hai nằm trong vùng tâm 60% (bán kính tâm = 35 * 0.6 = 21 px; khoảng cách từ tâm là 8 px < 21 px)
    cv2.circle(img, (62, 70), 1, 248, -1)
    cv2.circle(img, (78, 70), 1, 248, -1)

    result = detect_corneal_reflex(img, iris_center, iris_radius)

    assert result.has_fallback
    assert result.reference_point_used == "PUPIL_CENTER_FALLBACK"
    assert result.status == ReflexDetectionStatus.AMBIGUOUS_MULTI_REFLECTION
    assert "AMBIGUOUS_MULTI_REFLECTION" in result.warning_flags


# ==============================================================================
# 5. Kiểm thử hàm chấm điểm ứng viên đa biến (Multi-Objective Scoring)
# ==============================================================================

def test_v2_multi_objective_scoring_prefers_closer_round_candidate():
    """Kiểm tra thuật toán ưu tiên ứng viên gần tâm và tròn hơn thay vì ứng viên hơi méo hoặc xa tâm."""
    iris_center = (70.0, 70.0)
    iris_radius = 35.0
    img = create_base_synthetic_eye(iris_center=iris_center, iris_radius=iris_radius)

    # Ứng viên A: tại (71, 70) (cách tâm 1 px), hình tròn hoàn hảo 1 pixel, độ sáng 250
    img[70, 71] = 250

    # Ứng viên B: tại (70, 84) (cách tâm 14 px), hình chữ nhật 2x4, độ sáng 250
    # Khoảng cách A-B = 14 px > 8 px (không thành cụm)
    img[83:85, 70:74] = 250

    result = detect_corneal_reflex(img, iris_center, iris_radius)

    # Thuật toán phải chọn ứng viên A (tại x ~ 71, y ~ 70) vì có score nhỏ hơn nhiều
    assert not result.has_fallback
    assert result.status == ReflexDetectionStatus.SUCCESS
    assert abs(result.reflex_point[0] - 71.0) < 0.5
    assert abs(result.reflex_point[1] - 70.0) < 0.5
