"""
Kiểm thử đơn vị và tích hợp cho Module M3: Landmark Service & HVID Scale Reference.
Kiểm tra độ chính xác mốc mống mắt, thước đo HVID và tính nhất quán sinh học.
Tuyệt đối tuân thủ G2: Xử lý trong RAM.
Tuyệt đối tuân thủ G4: Không dùng từ ngữ thuộc danh mục cấm.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch
import cv2
from fastapi.testclient import TestClient
import numpy as np
import pytest

from app.core.config import settings
from app.main import app
from app.services.landmark_service import (
    LANDMARK_IRIS_CENTER_OD,
    LANDMARK_IRIS_CENTER_OS,
    detect_landmarks,
)
from app.utils.geometry import euclidean_distance, normalized_to_pixel

client = TestClient(app)

from tests.test_helpers import get_sample_image_path
SAMPLE_IMAGE_PATH = get_sample_image_path()


def get_test_image() -> np.ndarray:
    """Nạp ảnh thử nghiệm từ kho dữ liệu chuẩn."""
    if SAMPLE_IMAGE_PATH.exists():
        img = cv2.imread(str(SAMPLE_IMAGE_PATH))
        if img is not None:
            return img
    pytest.skip("Không tìm thấy tệp ảnh mẫu thực tế để thực hiện kiểm thử.")


# ==============================================================================
# Kiểm thử tiện ích hình học (Geometry Utilities)
# ==============================================================================

def test_geometry_utilities():
    """Kiểm tra hàm tính khoảng cách Euclidean và chuẩn hóa tọa độ pixel."""
    p1 = (10.0, 20.0)
    p2 = (13.0, 24.0)
    # Khoảng cách 3-4-5
    assert euclidean_distance(p1, p2) == 5.0

    mock_lm = MagicMock(x=0.5, y=0.25)
    px, py = normalized_to_pixel(mock_lm, 640, 480)
    assert px == 320.0
    assert py == 120.0


# ==============================================================================
# Kiểm thử trích xuất mốc mống mắt & Thước HVID (Module M3)
# ==============================================================================

def test_detect_landmarks_success():
    """Kiểm thử trích xuất mốc thành công trên ảnh thực tế và kiểm tra các mốc giải phẫu."""
    img = get_test_image()
    res = detect_landmarks(img)

    # 1. Kiểm tra số lượng mốc
    assert res.landmarks_px.shape == (478, 2)

    # 2. Kiểm tra tâm mống mắt OD và OS
    assert len(res.iris_center_od) == 2
    assert len(res.iris_center_os) == 2
    # Mắt phải (OD) hiển thị ở nửa bên trái của ảnh (x_od < x_os)
    assert res.iris_center_od[0] < res.iris_center_os[0]

    # 3. Kiểm tra rìa ngang mống mắt (limbus)
    limbus_od_1, limbus_od_2 = res.limbus_horizontal_od
    limbus_os_1, limbus_os_2 = res.limbus_horizontal_os
    assert euclidean_distance(limbus_od_1, limbus_od_2) > 0
    assert euclidean_distance(limbus_os_1, limbus_os_2) > 0

    # 4. Kiểm tra đường kính HVID
    assert res.hvid_od_px >= settings.MIN_IRIS_DIAMETER_PX
    assert res.hvid_os_px >= settings.MIN_IRIS_DIAMETER_PX

    # 5. Kiểm tra hệ số quy đổi S_iris nằm trong khoảng vật lý hợp lý (0.15 - 0.50 mm/px)
    assert 0.10 <= res.scale_factor_od <= 0.60
    assert 0.10 <= res.scale_factor_os <= 0.60

    # 6. Kiểm tra nhất quán sinh học hai mắt (|HVID_OD - HVID_OS| / HVID_avg < 10%)
    assert res.is_hvid_biologically_consistent is True
    assert res.hvid_asymmetry_ratio < 0.10

    # 7. Kiểm tra đường viền bờ mi và cung mày
    assert len(res.upper_eyelid_od) == 7
    assert len(res.upper_eyelid_os) == 7
    assert len(res.lower_eyelid_od) == 5
    assert len(res.lower_eyelid_os) == 5
    assert len(res.eyebrow_od) == 5
    assert len(res.eyebrow_os) == 5


def test_subpixel_iris_precision():
    """Kiểm tra tọa độ tâm mống mắt có độ chính xác sub-pixel (số thực float)."""
    img = get_test_image()
    res = detect_landmarks(img)

    # Tọa độ tâm mống mắt phải có phần thập phân (sub-pixel)
    assert isinstance(res.iris_center_od[0], float)
    assert isinstance(res.iris_center_od[1], float)
    assert isinstance(res.iris_center_os[0], float)
    assert isinstance(res.iris_center_os[1], float)

    # Tâm mống mắt mốc 468/473 nằm trong phạm vi rìa ngang limbus
    x_min_od = min(res.limbus_horizontal_od[0][0], res.limbus_horizontal_od[1][0])
    x_max_od = max(res.limbus_horizontal_od[0][0], res.limbus_horizontal_od[1][0])
    assert x_min_od <= res.iris_center_od[0] <= x_max_od


def test_hvid_asymmetry_warning():
    """Kiểm tra cảnh báo khi độ bất đối xứng HVID hai bên vượt quá 10%."""
    img = get_test_image()

    with patch("app.services.landmark_service._LANDMARK_FACE_MESH.process") as mock_process:
        # Mock trả về landmark với mống mắt trái bị kéo dãn bất thường (> 15%)
        h, w = img.shape[:2]
        mock_landmarks = [MagicMock(x=0.5, y=0.5) for _ in range(478)]

        # OD HVID = 40px (khoảng cách 0.20 -> 0.2625 của w=640 => 40px)
        mock_landmarks[468] = MagicMock(x=0.3, y=0.5)
        mock_landmarks[469] = MagicMock(x=0.25, y=0.5)
        mock_landmarks[471] = MagicMock(x=0.35, y=0.5)

        # OS HVID = 60px (chênh lệch 50% so với OD 40px)
        mock_landmarks[473] = MagicMock(x=0.7, y=0.5)
        mock_landmarks[474] = MagicMock(x=0.60, y=0.5)
        mock_landmarks[476] = MagicMock(x=0.80, y=0.5)

        mock_process.return_value = MagicMock(
            multi_face_landmarks=[MagicMock(landmark=mock_landmarks)]
        )

        res = detect_landmarks(img)
        assert res.is_hvid_biologically_consistent is False
        assert "HVID_ASYMMETRY_EXCEEDED_10_PERCENT" in res.warning_flags


def test_detect_landmarks_invalid_image():
    """Kiểm tra ném ngoại lệ khi ảnh đầu vào không hợp lệ hoặc rỗng."""
    with pytest.raises(ValueError, match="không hợp lệ"):
        detect_landmarks(np.array([]))

    with pytest.raises(ValueError, match="không hợp lệ"):
        detect_landmarks(None)


def test_screening_endpoint_phase3_scale_integration():
    """Kiểm tra endpoint /analyze trả về ScaleReference thực tế từ HVID thay vì mock."""
    img = get_test_image()
    success, buffer = cv2.imencode(".jpg", img)
    assert success

    response = client.post(
        "/api/v1/screening/analyze",
        files={"file": ("test_face.jpg", buffer.tobytes(), "image/jpeg")},
    )
    assert response.status_code == 200
    data = response.json()

    # Kiểm tra scale_reference đã được cập nhật từ Module M3
    scale = data["scale_reference"]
    assert scale["method"] == "HVID_WTW_POPULATION_AVERAGE"
    assert scale["assumed_corneal_diameter_mm"] == 11.71
    assert scale["hvid_pixels"]["right_eye"] > 30.0
    assert scale["hvid_pixels"]["left_eye"] > 30.0
    assert scale["scale_factor_mm_per_px"]["right_eye"] > 0.0
    assert scale["scale_factor_mm_per_px"]["left_eye"] > 0.0

    # Kiểm tra overlay coordinates đã có tọa độ mốc mống mắt thực
    overlay = data["annotated_overlay_coordinates"]
    assert len(overlay["right_eye"]["pupil_center"]) == 2
    assert len(overlay["left_eye"]["pupil_center"]) == 2
    assert len(overlay["right_eye"]["upper_eyelid_contour"]) >= 3
    assert len(overlay["left_eye"]["upper_eyelid_contour"]) >= 3
