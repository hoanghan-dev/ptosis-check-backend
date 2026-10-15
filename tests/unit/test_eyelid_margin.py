"""
Kiểm thử đơn vị và tích hợp cho Module M5: Upper Eyelid Margin Extractor & B-Spline Interpolator.
Kiểm tra khả năng khớp đường cong B-spline bậc ba, giao điểm cột x_ref,
quét gradient dọc sụn mi, loại bỏ nhiễu lông mi và tích hợp API endpoint.
Tuyệt đối tuân thủ G2: Xử lý trong RAM.
Tuyệt đối tuân thủ G4: Không dùng từ ngữ thuộc danh mục cấm.
"""

from pathlib import Path
import cv2
from fastapi.testclient import TestClient
import numpy as np
import pytest

from app.core.config import settings
from app.main import app
from app.services.eyelid_margin_service import extract_upper_margin, MarginResult
from app.services.landmark_service import detect_landmarks
from app.services.reflex_service import detect_corneal_reflex
from app.utils.spline_interpolator import EyelidSplineInterpolator

client = TestClient(app)

from tests.test_helpers import get_sample_image_path
SAMPLE_IMAGE_PATH = get_sample_image_path()


# ==============================================================================
# 1. Kiểm thử tiện ích nội suy B-spline bậc ba (EyelidSplineInterpolator)
# ==============================================================================

def test_spline_interpolator_cubic_accuracy():
    """Kiểm tra độ chính xác của B-spline bậc ba trên đường cong giải phẫu hình vòm."""
    # Mô phỏng đường cong bờ mi vòm cung: y = -0.01 * (x - 50)^2 + 40
    # Tại x = 50: đỉnh cung mi y = 40
    # Các mốc tại x = 20, 35, 50, 65, 80
    xs = [20.0, 35.0, 50.0, 65.0, 80.0]
    points = [(x, -0.01 * (x - 50.0) ** 2 + 40.0) for x in xs]

    interpolator = EyelidSplineInterpolator(points)

    # Đánh giá tại đỉnh x = 50.0
    y_apex = interpolator.evaluate_at_x(50.0)
    assert abs(y_apex - 40.0) < 0.1

    # Đánh giá tại điểm giữa x = 42.5
    expected_mid = -0.01 * (42.5 - 50.0) ** 2 + 40.0
    y_mid = interpolator.evaluate_at_x(42.5)
    assert abs(y_mid - expected_mid) < 0.5

    # Kiểm tra tạo đường cong mịn
    dense = interpolator.generate_dense_curve(num_points=25)
    assert len(dense) == 25
    assert dense[0][0] == 20.0
    assert dense[-1][0] == 80.0


def test_spline_interpolator_duplicate_x_handling():
    """Kiểm tra xử lý điểm mốc có hoành độ x trùng lặp mà không gây lỗi ma trận kỳ dị."""
    points = [(20.0, 48.0), (35.0, 44.0), (35.0, 44.2), (50.0, 40.0), (70.0, 46.0)]
    interpolator = EyelidSplineInterpolator(points)
    y_eval = interpolator.evaluate_at_x(35.0)
    assert abs(y_eval - 44.1) < 0.2


# ==============================================================================
# 2. Kiểm thử trích xuất bờ sụn mi (Module M5: extract_upper_margin)
# ==============================================================================

def test_extract_upper_margin_clean_edge():
    """Kiểm tra bắt đúng ranh giới bờ sụn mi có bước nhảy tương phản da -> khe mi."""
    width, height = 120, 120
    img = np.zeros((height, width), dtype=np.uint8)

    # Vùng da mi trên sáng màu (luma=180) từ y=0 đến y=42
    # Vùng khe mi / giác mạc tối hơn (luma=60) từ y=43 đến y=120
    img[0:43, :] = 180
    img[43:, :] = 60

    # Tập mốc bờ mi lân cận y=43
    landmarks_upper = [(20.0, 47.0), (40.0, 44.0), (60.0, 43.0), (80.0, 44.0), (100.0, 47.0)]
    reflex_point = (60.0, 75.0)  # Phản quang tại x_ref = 60, y_ref = 75

    result = extract_upper_margin(landmarks_upper, reflex_point, img)

    assert isinstance(result, MarginResult)
    assert result.margin_point[0] == 60.0
    # Bờ sụn mi phải nằm chính xác tại ranh giới tương phản y=43 +- 1 px
    assert abs(result.margin_point[1] - 43.0) <= 1.0
    assert result.is_gradient_refined
    assert result.gradient_magnitude > 20.0


def test_extract_upper_margin_eyelash_rejection():
    """
    Kiểm tra thuật toán bám vào bờ sụn mi thực tế tại y=40,
    không bị nhiễu kéo xuống bởi ngọn lông mi chúc xuống tại y=46.
    """
    width, height = 120, 120
    img = np.full((height, width), 160, dtype=np.uint8)  # Da mi trên luma=160

    # Ranh giới bờ sụn mi thực tế tại y=40 (bắt đầu khe mi luma=100)
    img[40:, :] = 100

    # Lông mi chúc xuống thành các vệt tối luma=40 kéo dài xuống y=46 tại cột x=60
    img[41:47, 59:62] = 40

    landmarks_upper = [(30.0, 43.0), (45.0, 41.0), (60.0, 40.0), (75.0, 41.0), (90.0, 43.0)]
    reflex_point = (60.0, 75.0)

    result = extract_upper_margin(landmarks_upper, reflex_point, img)

    # Thuật toán phải nhận diện bờ sụn mi tại lân cận y=40 (dựa trên y_init spline và gradient chuyển tiếp da)
    # chứ không bị kéo xuống ngọn lông mi y=46
    assert abs(result.margin_point[1] - 40.0) <= 2.0
    assert result.margin_point[1] < 45.0


def test_extract_upper_margin_no_gradient_fallback():
    """Kiểm tra ảnh phẳng không có gradient rõ nét -> Giữ nguyên vị trí mốc B-spline an toàn."""
    img = np.full((100, 100), 120, dtype=np.uint8)
    landmarks_upper = [(20.0, 45.0), (40.0, 41.0), (60.0, 40.0), (80.0, 42.0)]
    reflex_point = (60.0, 70.0)

    result = extract_upper_margin(landmarks_upper, reflex_point, img)

    assert not result.is_gradient_refined
    assert result.margin_point[1] == result.y_init_spline
    assert "EYELID_MARGIN_GRADIENT_NOT_FOUND_USING_SPLINE" in result.warning_flags


def test_extract_upper_margin_invalid_inputs():
    """Kiểm tra xử lý ngoại lệ khi danh sách mốc rỗng hoặc ảnh không hợp lệ."""
    with pytest.raises(ValueError):
        extract_upper_margin([], (50.0, 50.0), np.zeros((100, 100)))

    with pytest.raises(ValueError):
        extract_upper_margin([(20.0, 30.0), (40.0, 30.0)], None, np.zeros((100, 100)))

    with pytest.raises(ValueError):
        extract_upper_margin([(20.0, 30.0), (40.0, 30.0)], (50.0, 50.0), np.array([]))


# ==============================================================================
# 3. Kiểm thử trên ảnh thực tế và tích hợp Endpoint API
# ==============================================================================

def test_extract_upper_margin_real_image():
    """Kiểm thử trích xuất bờ mi trên ảnh khuôn mặt người thật."""
    if not SAMPLE_IMAGE_PATH.exists():
        pytest.skip("Không tìm thấy tệp ảnh mẫu thực tế.")

    img_bgr = cv2.imread(str(SAMPLE_IMAGE_PATH))
    assert img_bgr is not None

    landmarks = detect_landmarks(img_bgr)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    r_od = landmarks.hvid_od_px / 2.0
    r_os = landmarks.hvid_os_px / 2.0

    reflex_od = detect_corneal_reflex(img_gray, landmarks.iris_center_od, r_od)
    reflex_os = detect_corneal_reflex(img_gray, landmarks.iris_center_os, r_os)

    margin_od = extract_upper_margin(landmarks.upper_eyelid_od, reflex_od.reflex_point, img_gray)
    margin_os = extract_upper_margin(landmarks.upper_eyelid_os, reflex_os.reflex_point, img_gray)

    # 1. Bờ mi trên phải nằm phía trên điểm phản quang / tâm đồng tử (y_margin < y_reflex)
    assert margin_od.margin_point[1] < reflex_od.reflex_point[1]
    assert margin_os.margin_point[1] < reflex_os.reflex_point[1]

    # 2. Khoảng cách MRD1 pixel (y_reflex - y_margin) phải nằm trong phạm vi giải phẫu tự nhiên (5 - 40 px)
    mrd1_px_od = reflex_od.reflex_point[1] - margin_od.margin_point[1]
    mrd1_px_os = reflex_os.reflex_point[1] - margin_os.margin_point[1]
    assert 5.0 <= mrd1_px_od <= 40.0
    assert 5.0 <= mrd1_px_os <= 40.0

    # 3. Đường cong spline tạo ra phải có đủ các điểm mịn
    assert len(margin_od.spline_curve_points) == settings.EYELID_DENSE_CURVE_POINTS
    assert len(margin_os.spline_curve_points) == settings.EYELID_DENSE_CURVE_POINTS


def test_screening_endpoint_margin_integration():
    """Kiểm thử tích hợp đầu cuối API /analyze trả về tọa độ bờ mi thật và đường cong spline."""
    if not SAMPLE_IMAGE_PATH.exists():
        pytest.skip("Không tìm thấy tệp ảnh mẫu thực tế.")

    with open(SAMPLE_IMAGE_PATH, "rb") as f:
        file_bytes = f.read()

    response = client.post(
        "/api/v1/screening/analyze",
        files={"file": ("test_margin.jpg", file_bytes, "image/jpeg")},
    )

    assert response.status_code == 200
    data = response.json()

    od = data["measurements"]["right_eye_od"]
    os_eye = data["measurements"]["left_eye_os"]

    assert od["eyelid_margin_point"] is not None
    assert os_eye["eyelid_margin_point"] is not None
    assert od["eyelid_margin_point"]["x"] == od["reference_coordinates"]["x"]
    assert os_eye["eyelid_margin_point"]["x"] == os_eye["reference_coordinates"]["x"]

    # Kiểm tra đường viền spline trong overlay
    overlay = data["annotated_overlay_coordinates"]
    assert len(overlay["right_eye"]["upper_eyelid_contour"]) == settings.EYELID_DENSE_CURVE_POINTS
    assert len(overlay["left_eye"]["upper_eyelid_contour"]) == settings.EYELID_DENSE_CURVE_POINTS
