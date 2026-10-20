"""
Kiểm thử đơn vị cho Dịch vụ Định vị Mống mắt & Xác lập Thước đo HVID (Module Iris Model Service - Phase 3).
Xác thực:
1. Kiến trúc mô hình học sâu IrisFour và nạp trọng số iris_four.pt trên CPU.
2. Hàm tiện ích mở rộng ngữ cảnh vuông square_height_context.
3. Trích xuất chính xác 4 điểm cực mống mắt và xác lập thước đo mm/px (detect_iris_geometry).
4. Cảnh báo góc chụp nghiêng méo mống mắt PERSPECTIVE_TILT_DETECTED khi Axis Ratio < 0.85.
5. Kiểm định chất lượng đối xứng mống mắt hai bên evaluate_bilateral_iris_qc (<= 15%).
6. Xử lý an toàn khi ảnh rỗng hoặc ngoài vùng biên.
Tuân thủ tuyệt đối G2: Xử lý 100% trong RAM, không ghi đĩa.
Tuân thủ tuyệt đối G4: Không dùng từ ngữ thuộc danh mục cấm.
"""

import cv2
import numpy as np
import pytest
import torch

from app.core.config import settings
from app.services.eye_roi_service import extract_bilateral_eye_rois
from app.services.iris_model_service import (
    BilateralIrisQCResult,
    IrisFour,
    IrisGeometryResult,
    detect_iris_geometry,
    evaluate_bilateral_iris_qc,
    get_iris_model,
    square_height_context,
)
from tests.test_helpers import get_sample_image_path

SAMPLE_IMAGE_PATH = get_sample_image_path()


@pytest.fixture
def sample_image() -> np.ndarray:
    """Ảnh chụp chuẩn từ bộ dữ liệu mẫu."""
    img = cv2.imread(str(SAMPLE_IMAGE_PATH))
    assert img is not None, f"Không thể tải ảnh mẫu tại: {SAMPLE_IMAGE_PATH}"
    return img


def test_iris_four_model_loading_and_forward():
    """Kiểm tra nạp mô hình IrisFour và forward pass trên CPU."""
    model = get_iris_model()
    assert model is not None, "Không thể nạp mô hình iris_four.pt từ thư mục weights"
    assert isinstance(model, IrisFour)

    dummy = torch.randn(1, 3, 192, 192, device=settings.TORCH_DEVICE)
    with torch.no_grad():
        out = model(dummy)

    assert out.shape == (1, 8), f"Đầu ra mô hình không đúng kích thước 8 tọa độ: {out.shape}"
    # Các giá trị phải nằm trong khoảng chuẩn hóa [0, 1]
    assert (out >= 0.0).all() and (out <= 1.0).all()


def test_square_height_context_expansion():
    """Kiểm tra mở rộng ngữ cảnh vuông theo chiều cao khi crop quá hẹp."""
    box = (100, 200, 300, 250)  # w = 200, h = 50
    expanded = square_height_context(box, width=1000, height=1000)
    assert expanded[0] == 100
    assert expanded[2] == 300
    new_h = expanded[3] - expanded[1]
    assert new_h == 200, f"Chiều cao mới phải mở rộng bằng chiều rộng 200: {new_h}"


def test_detect_iris_geometry_od_and_os(sample_image: np.ndarray):
    """Kiểm tra trích xuất 4 cực mống mắt và xác lập thước đo HVID trên ảnh mẫu chuẩn."""
    roi_result = extract_bilateral_eye_rois(sample_image, image_type="fullface")
    assert roi_result.is_valid is True

    # Đo mắt phải (OD)
    od_geom = detect_iris_geometry(sample_image, roi_result.od.bbox, eye_side="OD")
    assert od_geom is not None
    assert od_geom.eye_side == "OD"
    assert od_geom.hvid_px > 30.0, f"Đường kính HVID quá nhỏ: {od_geom.hvid_px}"
    assert od_geom.vvid_px > 30.0
    assert od_geom.scale_factor_mm_per_px > 0.0
    assert od_geom.iris_left[0] < od_geom.iris_right[0]
    assert od_geom.iris_top[1] < od_geom.iris_bottom[1]
    assert od_geom.axis_ratio >= 0.85
    assert od_geom.is_circular is True

    # Đo mắt trái (OS)
    os_geom = detect_iris_geometry(sample_image, roi_result.os.bbox, eye_side="OS")
    assert os_geom is not None
    assert os_geom.eye_side == "OS"
    assert os_geom.hvid_px > 30.0
    assert os_geom.iris_left[0] < os_geom.iris_right[0]
    assert os_geom.axis_ratio >= 0.85
    assert os_geom.is_circular is True


def test_bilateral_iris_qc_success(sample_image: np.ndarray):
    """Kiểm tra chốt chặn chất lượng đối xứng mống mắt hai mắt trên ảnh đạt chuẩn."""
    roi_result = extract_bilateral_eye_rois(sample_image, image_type="fullface")
    od_geom = detect_iris_geometry(sample_image, roi_result.od.bbox, eye_side="OD")
    os_geom = detect_iris_geometry(sample_image, roi_result.os.bbox, eye_side="OS")

    qc_res = evaluate_bilateral_iris_qc(od_geom, os_geom)
    assert qc_res.bilateral_qc_passed is True
    assert qc_res.hvid_difference_ratio <= 0.15, (
        f"Chênh lệch HVID 2 mắt vượt ngưỡng 15%: {qc_res.hvid_difference_ratio}"
    )


def test_perspective_tilt_warning_on_distorted_crop(sample_image: np.ndarray):
    """Kiểm tra cảnh báo PERSPECTIVE_TILT_DETECTED khi ảnh mống mắt bị nghiêng méo (Axis Ratio < 0.85)."""
    # Tạo bản giả lập kết quả có tỷ lệ trục bị méo
    distorted_geom = IrisGeometryResult(
        eye_side="OD",
        iris_left=(100.0, 150.0),
        iris_right=(180.0, 150.0),  # hvid = 80 px
        iris_top=(140.0, 120.0),
        iris_bottom=(140.0, 170.0),  # vvid = 50 px -> ratio = 50/80 = 0.625 < 0.85
        iris_center=(140.0, 145.0),
        hvid_px=80.0,
        vvid_px=50.0,
        axis_ratio=0.625,
        scale_factor_mm_per_px=round(11.71 / 80.0, 5),
        is_circular=False,
        warning_flags=["PERSPECTIVE_TILT_DETECTED"],
    )

    normal_geom = IrisGeometryResult(
        eye_side="OS",
        iris_left=(200.0, 150.0),
        iris_right=(280.0, 150.0),
        iris_top=(240.0, 110.0),
        iris_bottom=(240.0, 190.0),
        iris_center=(240.0, 150.0),
        hvid_px=80.0,
        vvid_px=80.0,
        axis_ratio=1.0,
        scale_factor_mm_per_px=round(11.71 / 80.0, 5),
        is_circular=True,
    )

    qc_res = evaluate_bilateral_iris_qc(distorted_geom, normal_geom)
    assert "OBLIQUE_ANGLE_FLAG" in qc_res.warning_flags


def test_bilateral_qc_fails_on_asymmetric_hvid():
    """Kiểm tra từ chối khi chênh lệch đường kính mống mắt 2 mắt vượt quá 15%."""
    od_geom = IrisGeometryResult(
        eye_side="OD",
        iris_left=(100.0, 150.0),
        iris_right=(190.0, 150.0),  # HVID = 90 px
        iris_top=(145.0, 105.0),
        iris_bottom=(145.0, 195.0),
        iris_center=(145.0, 150.0),
        hvid_px=90.0,
        vvid_px=90.0,
        axis_ratio=1.0,
        scale_factor_mm_per_px=round(11.71 / 90.0, 5),
        is_circular=True,
    )

    os_geom = IrisGeometryResult(
        eye_side="OS",
        iris_left=(300.0, 150.0),
        iris_right=(360.0, 150.0),  # HVID = 60 px -> Diff = |90 - 60| / 90 = 33.3% > 15%
        iris_top=(330.0, 120.0),
        iris_bottom=(330.0, 180.0),
        iris_center=(330.0, 150.0),
        hvid_px=60.0,
        vvid_px=60.0,
        axis_ratio=1.0,
        scale_factor_mm_per_px=round(11.71 / 60.0, 5),
        is_circular=True,
    )

    qc_res = evaluate_bilateral_iris_qc(od_geom, os_geom)
    assert qc_res.bilateral_qc_passed is False
    assert "HVID_ASYMMETRY_EXCEEDED" in qc_res.warning_flags
    assert qc_res.hvid_difference_ratio == 0.333


def test_detect_iris_geometry_empty_input():
    """Kiểm tra xử lý an toàn khi đầu vào rỗng."""
    assert detect_iris_geometry(None, (0, 0, 10, 10)) is None
    empty_img = np.zeros((0, 0, 3), dtype=np.uint8)
    assert detect_iris_geometry(empty_img, (0, 0, 10, 10)) is None
    blank = np.zeros((100, 100, 3), dtype=np.uint8)
    assert detect_iris_geometry(blank, (0, 0, 0, 0)) is None
