"""
Kiểm thử đơn vị cho Dịch vụ Định vị Đồng tử Độc lập (Module M4.5 - Phase 3).
Kiểm tra khả năng bắt tâm đồng tử sub-pixel, lọc hình thái học, chốt chặn độ tương phản ROI,
và kiểm tra hiệu năng CPU (< 3 ms) trong RAM.
Tuân thủ tuyệt đối G2 (In-Memory Zero-Retention) và G4 (Non-Diagnostic Safety).
"""

import math
import time
import cv2
import numpy as np
import pytest

from app.core.config import settings
from app.services.pupil_service import (
    PupilResult,
    detect_pupil_in_iris_roi,
)


def create_synthetic_eye_with_pupil(
    width: int = 160,
    height: int = 160,
    iris_center: tuple = (80.0, 80.0),
    iris_radius: float = 40.0,
    pupil_offset: tuple = (0.0, 0.0),
    pupil_radius: float = 14.0,
    sclera_luma: int = 200,
    iris_luma: int = 85,
    pupil_luma: int = 25,
) -> np.ndarray:
    """Tạo ảnh nhân tạo 3 kênh BGR trong RAM mô phỏng củng mạc, mống mắt và đồng tử."""
    img = np.full((height, width, 3), sclera_luma, dtype=np.uint8)
    icx, icy = int(iris_center[0]), int(iris_center[1])

    # Mống mắt
    cv2.circle(img, (icx, icy), int(iris_radius), (iris_luma, iris_luma, iris_luma), -1, lineType=cv2.LINE_AA)

    # Đồng tử
    pcx = int(round(iris_center[0] + pupil_offset[0]))
    pcy = int(round(iris_center[1] + pupil_offset[1]))
    cv2.circle(img, (pcx, pcy), int(pupil_radius), (pupil_luma, pupil_luma, pupil_luma), -1, lineType=cv2.LINE_AA)

    return img


# ==============================================================================
# 1. Kiểm thử định vị đồng tử chuẩn (Standard Centered Pupil)
# ==============================================================================

def test_detect_pupil_standard_centered():
    """Kiểm tra phát hiện đồng tử tròn rõ nằm gần tâm mống mắt với sai số < 1.0 px."""
    iris_center = (80.0, 80.0)
    iris_radius = 40.0
    pupil_radius = 14.0

    img = create_synthetic_eye_with_pupil(
        iris_center=iris_center,
        iris_radius=iris_radius,
        pupil_offset=(1.0, -1.0),
        pupil_radius=pupil_radius,
    )

    result = detect_pupil_in_iris_roi(img, iris_center, iris_radius)

    assert isinstance(result, PupilResult)
    assert result.status == "DETECTED"
    assert result.pupil_point is not None
    assert result.confidence_score == settings.PUPIL_FALLBACK_CONFIDENCE
    assert result.circularity is not None and result.circularity >= settings.PUPIL_MIN_CIRCULARITY

    # Tọa độ thực tế đặt tại (81.0, 79.0)
    expected_x, expected_y = 81.0, 79.0
    error = math.hypot(result.pupil_point[0] - expected_x, result.pupil_point[1] - expected_y)
    assert error < 1.0, f"Sai số vị trí đồng tử ({error:.2f} px) vượt quá 1.0 px."

    # Đường kính thực tế đặt tại 2 * 14.0 = 28.0 px
    assert result.pupil_diameter_px is not None
    assert abs(result.pupil_diameter_px - 28.0) < 3.0


# ==============================================================================
# 2. Kiểm thử từ chối khi lệch tâm quá mức (> 25% đường kính mống mắt)
# ==============================================================================

def test_detect_pupil_excessive_offset_rejected():
    """Kiểm tra từ chối đồng tử khi độ lệch tâm vượt quá 25% đường kính mống mắt."""
    iris_center = (80.0, 80.0)
    iris_radius = 40.0  # đường kính = 80.0 px; max offset = 0.25 * 80 = 20.0 px
    pupil_radius = 12.0

    # Đặt đồng tử lệch 25 px (> 20.0 px)
    img = create_synthetic_eye_with_pupil(
        iris_center=iris_center,
        iris_radius=iris_radius,
        pupil_offset=(25.0, 0.0),
        pupil_radius=pupil_radius,
    )

    result = detect_pupil_in_iris_roi(img, iris_center, iris_radius)

    assert result.status == "PUPIL_NOT_FOUND"
    assert result.pupil_point is None
    assert result.confidence_score == 0.0


# ==============================================================================
# 3. Kiểm thử độ bền với ảnh đồng màu hoặc tương phản quá thấp
# ==============================================================================

def test_detect_pupil_low_contrast_roi():
    """Kiểm tra vùng mống mắt đồng màu không có tương phản rõ rệt (P85 - P15 < 12.0) -> PUPIL_NOT_FOUND."""
    iris_center = (80.0, 80.0)
    iris_radius = 40.0
    # Mống mắt đồng màu xám 70, không có đồng tử
    img = np.full((160, 160, 3), 180, dtype=np.uint8)
    cv2.circle(img, (80, 80), int(iris_radius), (70, 70, 70), -1)

    result = detect_pupil_in_iris_roi(img, iris_center, iris_radius)

    assert result.status == "PUPIL_NOT_FOUND"
    assert result.pupil_point is None


def test_detect_pupil_glint_masking():
    """Kiểm tra mặt nạ khử lóa (bright_mask >= 180) loại bỏ đốm flash nằm đè lên đồng tử."""
    iris_center = (80.0, 80.0)
    iris_radius = 40.0
    img = create_synthetic_eye_with_pupil(
        iris_center=iris_center,
        iris_radius=iris_radius,
        pupil_offset=(0.0, 0.0),
        pupil_radius=14.0,
    )

    # Thêm đốm lóa trắng 255 đè lên đồng tử
    cv2.circle(img, (80, 80), 3, (255, 255, 255), -1)

    result = detect_corneal_pupil = detect_pupil_in_iris_roi(img, iris_center, iris_radius)

    # Dù có đốm sáng lóa, thuật toán vẫn nhận diện được đồng tử
    assert result.status == "DETECTED"
    assert result.pupil_point is not None
    assert math.hypot(result.pupil_point[0] - 80.0, result.pupil_point[1] - 80.0) < 1.5


# ==============================================================================
# 4. Kiểm thử xử lý ngoại lệ đầu vào
# ==============================================================================

def test_detect_pupil_invalid_inputs():
    """Kiểm tra các trường hợp bán kính mống mắt quá nhỏ hoặc dữ liệu ảnh rỗng."""
    img = np.full((100, 100, 3), 120, dtype=np.uint8)

    # Bán kính <= 2.0 px
    result = detect_pupil_in_iris_roi(img, (50.0, 50.0), 1.5)
    assert result.status == "LOW_QUALITY_INPUT"
    assert result.pupil_point is None

    # Ảnh rỗng
    with pytest.raises(ValueError):
        detect_pupil_in_iris_roi(np.array([]), (50.0, 50.0), 30.0)


# ==============================================================================
# 5. Kiểm thử hiệu năng thực thi CPU (< 3 ms)
# ==============================================================================

def test_detect_pupil_cpu_execution_speed():
    """Kiểm tra thời gian thực thi trung bình trên CPU đạt chuẩn dưới 3 ms mỗi mắt."""
    iris_center = (80.0, 80.0)
    iris_radius = 40.0
    img = create_synthetic_eye_with_pupil(iris_center=iris_center, iris_radius=iris_radius)

    # Khởi động (warmup)
    for _ in range(5):
        detect_pupil_in_iris_roi(img, iris_center, iris_radius)

    # Đo 50 lần
    iterations = 50
    start = time.perf_counter()
    for _ in range(iterations):
        detect_pupil_in_iris_roi(img, iris_center, iris_radius)
    elapsed = time.perf_counter() - start

    avg_ms = (elapsed / iterations) * 1000.0
    assert avg_ms < 3.0, f"Thời gian thực thi trung bình ({avg_ms:.2f} ms) vượt ngưỡng 3.0 ms."
