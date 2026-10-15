"""
Kiểm thử đơn vị và tích hợp cho Module M4: Corneal Reflex Service & Fallback Mechanism.
Kiểm tra khả năng bắt đốm phản xạ Purkinje I, tính toán trọng tâm sub-pixel,
lọc đa phản xạ, và cơ chế tự động dự phòng sang tâm đồng tử (FR-015).
Tuyệt đối tuân thủ G2: Xử lý trong RAM.
Tuyệt đối tuân thủ G4: Không dùng từ ngữ thuộc danh mục cấm.
"""

from io import BytesIO
from pathlib import Path
import cv2
from fastapi.testclient import TestClient
import numpy as np
import pytest

from app.core.config import settings
from app.main import app
from app.services.landmark_service import detect_landmarks
from app.services.reflex_service import detect_corneal_reflex, ReflexResult
from app.utils.geometry import euclidean_distance

client = TestClient(app)

from tests.test_helpers import get_sample_image_path
SAMPLE_IMAGE_PATH = get_sample_image_path()


def create_synthetic_iris_roi(
    width: int = 120,
    height: int = 120,
    iris_center: tuple = (60.0, 60.0),
    iris_radius: float = 30.0,
    iris_intensity: int = 60,
    background_intensity: int = 160,
) -> np.ndarray:
    """Tạo ảnh xám nhân tạo mô phỏng vùng mống mắt."""
    img = np.full((height, width), background_intensity, dtype=np.uint8)
    cx, cy = int(iris_center[0]), int(iris_center[1])
    cv2.circle(img, (cx, cy), int(iris_radius), iris_intensity, -1)
    return img


# ==============================================================================
# 1. Kiểm thử phát hiện đốm phản xạ đơn sắc nét (Sharp Single Flash Glint)
# ==============================================================================

def test_detect_corneal_reflex_single_glint():
    """Kiểm tra bắt chính xác đốm Purkinje duy nhất trong vùng 60% mống mắt."""
    iris_center = (60.0, 60.0)
    iris_radius = 30.0
    img = create_synthetic_iris_roi(iris_center=iris_center, iris_radius=iris_radius)

    # Đặt đốm sáng flash 3x3 pixel cường độ 250 tại (63, 58) (nằm trong 60% tâm: dist = sqrt(3^2 + 2^2) = 3.6 < 18)
    glint_x, glint_y = 63, 58
    cv2.circle(img, (glint_x, glint_y), 1, 250, -1)

    result = detect_corneal_reflex(img, iris_center, iris_radius)

    assert isinstance(result, ReflexResult)
    assert not result.has_fallback
    assert result.reference_point_used == "CORNEAL_LIGHT_REFLEX"
    assert result.confidence_score == 1.0
    assert result.glint_intensity >= 240.0
    assert result.warning_flags == []
    # Tọa độ trọng tâm sub-pixel phải rất gần điểm đặt glint
    dist = euclidean_distance(result.reflex_point, (float(glint_x), float(glint_y)))
    assert dist < 1.0


def test_detect_corneal_reflex_subpixel_precision():
    """Kiểm tra độ chính xác sub-pixel của trọng tâm cường độ sáng."""
    iris_center = (60.0, 60.0)
    iris_radius = 30.0
    img = create_synthetic_iris_roi(iris_center=iris_center, iris_radius=iris_radius)

    # Tạo đốm sáng 2 pixel cạnh nhau với cường độ khác nhau để kiểm tra sub-pixel centroid
    # Pixel A tại (60, 60) luma=250, Pixel B tại (61, 60) luma=235
    img[60, 60] = 250
    img[60, 61] = 235

    expected_x = (60 * 250 + 61 * 235) / (250 + 235)
    expected_y = 60.0

    result = detect_corneal_reflex(img, iris_center, iris_radius)

    assert not result.has_fallback
    assert abs(result.reflex_point[0] - expected_x) < 0.1
    assert abs(result.reflex_point[1] - expected_y) < 0.1


# ==============================================================================
# 2. Kiểm thử cơ chế Fallback sang tâm đồng tử (FR-015)
# ==============================================================================

def test_detect_corneal_reflex_no_glint_fallback():
    """Kiểm tra ảnh không có flash (mống mắt tối, không có điểm sáng >= 230)."""
    iris_center = (60.0, 60.0)
    iris_radius = 30.0
    img = create_synthetic_iris_roi(iris_center=iris_center, iris_radius=iris_radius)

    # Toàn bộ mống mắt có độ sáng <= 100, không có flash
    result = detect_corneal_reflex(img, iris_center, iris_radius)

    assert result.has_fallback
    assert result.reference_point_used == "PUPIL_CENTER"
    assert result.confidence_score == 0.65
    assert result.reflex_point == iris_center
    assert "REFLEX_FALLBACK_TO_PUPIL" in result.warning_flags


def test_detect_corneal_reflex_glare_outside_central_60():
    """Kiểm tra đốm sáng nằm ngoài vùng trung tâm 60% đĩa mống bị loại trừ và fallback."""
    iris_center = (60.0, 60.0)
    iris_radius = 30.0
    img = create_synthetic_iris_roi(iris_center=iris_center, iris_radius=iris_radius)

    # Đặt điểm sáng tại khoảng cách 25 px từ tâm (tức 25/30 = 83% bán kính, vượt ngưỡng 60%)
    img[60, 85] = 255
    img[60, 86] = 255

    result = detect_corneal_reflex(img, iris_center, iris_radius)

    # Đốm sáng ngoài 60% bị lọc -> Không còn đốm sáng hợp lệ trong tâm -> Fallback
    assert result.has_fallback
    assert result.reference_point_used == "PUPIL_CENTER"
    assert result.confidence_score == 0.65
    assert result.reflex_point == iris_center
    assert "REFLEX_FALLBACK_TO_PUPIL" in result.warning_flags


def test_detect_corneal_reflex_ambiguous_multi_reflection():
    """Kiểm tra tình huống có 2 đốm sáng cạnh tranh trong vùng tâm -> Gắn cờ AMBIGUOUS và Fallback."""
    iris_center = (60.0, 60.0)
    iris_radius = 30.0
    img = create_synthetic_iris_roi(iris_center=iris_center, iris_radius=iris_radius)

    # Đặt 2 đốm sáng độc lập có kích thước và độ sáng tương đương trong vùng tâm
    # Đốm 1 tại (56, 60)
    cv2.circle(img, (56, 60), 1, 245, -1)
    # Đốm 2 tại (64, 60)
    cv2.circle(img, (64, 60), 1, 245, -1)

    result = detect_corneal_reflex(img, iris_center, iris_radius)

    assert result.has_fallback
    assert result.reference_point_used == "PUPIL_CENTER"
    assert result.confidence_score == 0.65
    assert "AMBIGUOUS_MULTI_REFLECTION" in result.warning_flags
    assert "REFLEX_FALLBACK_TO_PUPIL" in result.warning_flags


def test_detect_corneal_reflex_invalid_inputs():
    """Kiểm tra xử lý đầu vào dị thường hoặc bán kính quá nhỏ."""
    img = np.zeros((100, 100), dtype=np.uint8)

    # Bán kính mống mắt <= 2 px
    result = detect_corneal_reflex(img, (50.0, 50.0), 1.0)
    assert result.has_fallback
    assert "INVALID_IRIS_RADIUS" in result.warning_flags

    # Ảnh rỗng
    with pytest.raises(ValueError):
        detect_corneal_reflex(np.array([]), (50.0, 50.0), 20.0)


# ==============================================================================
# 3. Kiểm thử tích hợp trên ảnh thực tế & API Endpoint
# ==============================================================================

def test_detect_corneal_reflex_real_image():
    """Kiểm thử trích xuất phản quang trên ảnh người thật từ kho dữ liệu."""
    if not SAMPLE_IMAGE_PATH.exists():
        pytest.skip("Không tìm thấy tệp ảnh mẫu thực tế.")

    img_bgr = cv2.imread(str(SAMPLE_IMAGE_PATH))
    assert img_bgr is not None

    landmarks = detect_landmarks(img_bgr)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    r_od = landmarks.hvid_od_px / 2.0
    r_os = landmarks.hvid_os_px / 2.0

    res_od = detect_corneal_reflex(img_gray, landmarks.iris_center_od, r_od)
    res_os = detect_corneal_reflex(img_gray, landmarks.iris_center_os, r_os)

    # Cả hai mắt phải trả về kết quả hợp lệ mà không phát sinh ngoại lệ
    assert res_od.reference_point_used in ["CORNEAL_LIGHT_REFLEX", "PUPIL_CENTER"]
    assert res_os.reference_point_used in ["CORNEAL_LIGHT_REFLEX", "PUPIL_CENTER"]
    assert res_od.confidence_score in [0.65, 1.0]
    assert res_os.confidence_score in [0.65, 1.0]

    # Điểm tham chiếu phải nằm trong phạm vi đĩa mống mắt
    dist_od = euclidean_distance(res_od.reflex_point, landmarks.iris_center_od)
    dist_os = euclidean_distance(res_os.reflex_point, landmarks.iris_center_os)
    assert dist_od <= r_od
    assert dist_os <= r_os


def test_screening_endpoint_reflex_integration():
    """Kiểm thử tích hợp đầu cuối API /analyze truyền tải đúng thông tin phản quang."""
    if not SAMPLE_IMAGE_PATH.exists():
        pytest.skip("Không tìm thấy tệp ảnh mẫu thực tế.")

    with open(SAMPLE_IMAGE_PATH, "rb") as f:
        file_bytes = f.read()

    response = client.post(
        "/api/v1/screening/analyze",
        files={"file": ("test_reflex.jpg", file_bytes, "image/jpeg")},
    )

    assert response.status_code == 200
    data = response.json()

    # Kiểm tra cấu trúc đo lường và điểm tham chiếu
    od_meas = data["measurements"]["right_eye_od"]
    os_meas = data["measurements"]["left_eye_os"]
    assert od_meas["reference_point_used"] in ["CORNEAL_LIGHT_REFLEX", "PUPIL_CENTER"]
    assert os_meas["reference_point_used"] in ["CORNEAL_LIGHT_REFLEX", "PUPIL_CENTER"]
    assert od_meas["reference_coordinates"] is not None
    assert os_meas["reference_coordinates"] is not None

    # Kiểm tra overlay
    overlay = data["annotated_overlay_coordinates"]
    assert len(overlay["right_eye"]["corneal_reflex"]) == 2
    assert len(overlay["left_eye"]["corneal_reflex"]) == 2
    assert len(overlay["right_eye"]["pupil_center"]) == 2
    assert len(overlay["left_eye"]["pupil_center"]) == 2
