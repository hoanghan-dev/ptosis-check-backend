"""
Kiểm thử đơn vị cho Dịch vụ Phân đoạn Bờ Mi & Khe Mi (Module Eyelid Model Service - Phase 4).
Xác thực:
1. Kiến trúc mô hình EyelidUNet và nạp trọng số eyelid_unet.pt trên CPU.
2. Hàm phân tách contour thành bờ mi trên (upper) và bờ mi dưới (lower).
3. Trích xuất giao điểm bờ sụn mi trên y_margin, bờ mi dưới y_lower và PFH.
4. Xử lý an toàn trường hợp sa trễ mi nặng (y_margin > y_ref) không bị crash.
5. Xử lý an toàn khi ảnh rỗng hoặc không hợp lệ.
Tuân thủ tuyệt đối G2: Xử lý 100% trong RAM, không ghi đĩa.
Tuân thủ tuyệt đối G4: Không dùng từ ngữ thuộc danh mục cấm.
"""

import cv2
import numpy as np
import pytest
import torch

from app.core.config import settings
from app.services.eye_roi_service import extract_bilateral_eye_rois
from app.services.eyelid_model_service import (
    EyelidSegmentationResult,
    EyelidUNet,
    _split_contour_into_upper_and_lower,
    get_eyelid_model,
    segment_eyelid_margins,
)
from app.services.iris_model_service import detect_iris_geometry
from tests.test_helpers import get_sample_image_path

SAMPLE_IMAGE_PATH = get_sample_image_path()


@pytest.fixture
def sample_image() -> np.ndarray:
    """Ảnh chụp chuẩn từ bộ dữ liệu mẫu."""
    img = cv2.imread(str(SAMPLE_IMAGE_PATH))
    assert img is not None, f"Không thể tải ảnh mẫu tại: {SAMPLE_IMAGE_PATH}"
    return img


def test_eyelid_unet_model_loading_and_forward():
    """Kiểm tra nạp mô hình EyelidUNet và forward pass trên CPU."""
    model = get_eyelid_model()
    assert model is not None, "Không thể nạp mô hình eyelid_unet.pt từ thư mục weights"
    assert isinstance(model, EyelidUNet)

    dummy = torch.randn(1, 3, 192, 192, device=settings.TORCH_DEVICE)
    with torch.no_grad():
        out = model(dummy)

    assert out.shape == (1, 1, 192, 192), f"Kích thước đầu ra không khớp: {out.shape}"


def test_split_contour_into_upper_and_lower():
    """Kiểm tra giải thuật phân tách đa giác khe mi thành 2 đường viền bờ mi trên và mi dưới."""
    # Tạo đa giác dạng elip đại diện cho khe mi
    ellipse_pts = []
    for deg in range(0, 360, 20):
        rad = np.radians(deg)
        x = 100.0 + 40.0 * np.cos(rad)
        y = 50.0 + 15.0 * np.sin(rad)
        ellipse_pts.append([x, y])
    pts_arr = np.array(ellipse_pts, dtype=np.float32)

    upper, lower = _split_contour_into_upper_and_lower(pts_arr)
    assert len(upper) > 0
    assert len(lower) > 0

    mean_upper_y = np.mean([p[1] for p in upper])
    mean_lower_y = np.mean([p[1] for p in lower])

    # Bờ mi trên phải có tung độ nhỏ hơn (nằm cao hơn) bờ mi dưới
    assert mean_upper_y < mean_lower_y

    # Kiểm tra tính chất sắp xếp tăng dần theo trục X
    for i in range(len(upper) - 1):
        assert upper[i][0] <= upper[i + 1][0]
    for i in range(len(lower) - 1):
        assert lower[i][0] <= lower[i + 1][0]


def test_segment_eyelid_margins_od_and_os(sample_image: np.ndarray):
    """Kiểm tra phân đoạn bờ mi trên và dưới trên ảnh mắt thực tế."""
    roi_result = extract_bilateral_eye_rois(sample_image, image_type="fullface")
    assert roi_result.is_valid is True

    # Kiểm tra cho mắt phải (OD)
    od_iris = detect_iris_geometry(sample_image, roi_result.od.bbox, eye_side="OD")
    assert od_iris is not None
    x_ref_od = od_iris.iris_center[0]
    y_ref_od = od_iris.iris_center[1]

    od_eyelid = segment_eyelid_margins(
        sample_image,
        roi_result.od.bbox,
        x_ref=x_ref_od,
        y_ref=y_ref_od,
        eye_side="OD",
    )
    assert od_eyelid is not None
    assert od_eyelid.is_valid is True
    assert od_eyelid.eye_side == "OD"
    assert od_eyelid.margin_point[0] == round(x_ref_od, 2)
    assert od_eyelid.lower_margin_point[0] == round(x_ref_od, 2)

    # Bờ mi trên phải nằm trên bờ mi dưới
    assert od_eyelid.margin_point[1] < od_eyelid.lower_margin_point[1]
    assert od_eyelid.pfh_px > 15.0, f"Khe mi quá hẹp bất thường: {od_eyelid.pfh_px}"

    assert len(od_eyelid.upper_eyelid_contour) >= 3
    assert len(od_eyelid.lower_eyelid_contour) >= 3
    assert len(od_eyelid.palpebral_contour) >= 6

    # Kiểm tra cho mắt trái (OS)
    os_iris = detect_iris_geometry(sample_image, roi_result.os.bbox, eye_side="OS")
    assert os_iris is not None
    x_ref_os = os_iris.iris_center[0]
    y_ref_os = os_iris.iris_center[1]

    os_eyelid = segment_eyelid_margins(
        sample_image,
        roi_result.os.bbox,
        x_ref=x_ref_os,
        y_ref=y_ref_os,
        eye_side="OS",
    )
    assert os_eyelid is not None
    assert os_eyelid.is_valid is True
    assert os_eyelid.eye_side == "OS"
    assert os_eyelid.margin_point[1] < os_eyelid.lower_margin_point[1]
    assert os_eyelid.pfh_px > 15.0


def test_severe_droop_margin_below_reflex_handling(sample_image: np.ndarray):
    """
    Xác thực trường hợp sa trễ mi nặng (y_margin > y_ref):
    Hệ thống xử lý bình thường, không gây lỗi và đánh dấu cờ cảnh báo chính xác.
    """
    roi_result = extract_bilateral_eye_rois(sample_image, image_type="fullface")
    od_iris = detect_iris_geometry(sample_image, roi_result.od.bbox, eye_side="OD")
    x_ref = od_iris.iris_center[0]

    # Giả lập điểm tham chiếu phản quang nằm rất cao (y_ref nhỏ hơn y_margin thực tế)
    artificially_high_y_ref = 800.0

    res = segment_eyelid_margins(
        sample_image,
        roi_result.od.bbox,
        x_ref=x_ref,
        y_ref=artificially_high_y_ref,
        eye_side="OD",
    )
    assert res is not None
    assert res.is_valid is True
    assert res.margin_point[1] > artificially_high_y_ref
    assert "SEVERE_DROOP_MARGIN_BELOW_REFLEX" in res.warning_flags


def test_segment_eyelid_margins_empty_inputs():
    """Kiểm tra xử lý an toàn khi đầu vào rỗng hoặc sai kích thước."""
    assert segment_eyelid_margins(None, (0, 0, 10, 10), x_ref=50.0) is None
    empty_img = np.zeros((0, 0, 3), dtype=np.uint8)
    assert segment_eyelid_margins(empty_img, (0, 0, 10, 10), x_ref=50.0) is None
    blank = np.zeros((100, 100, 3), dtype=np.uint8)
    assert segment_eyelid_margins(blank, (0, 0, 0, 0), x_ref=50.0) is None
