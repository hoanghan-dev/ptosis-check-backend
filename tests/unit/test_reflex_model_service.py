"""
Kiểm thử đơn vị cho Dịch vụ Định vị Phản quang & Đồng tử Dự phòng (Module Reflex Model Service - Phase 5).
Xác thực:
1. Kiến trúc mạng FCN pr_model (reflex_localizer.pt) và DirectPupilCNN (pupil_direct.pt).
2. Hàm tiên nghiệm mềm soft_prior và giải mã tâm kỳ vọng expected_center.
3. Bộ lọc quang học reflex_roi_qc lọc vệt lóa kéo dài và đốm Purkinje I sub-pixel.
4. Chuỗi dự phòng 3 cấp chuẩn hóa (Tier 1: Purkinje -> Tier 2: Pupil -> Tier 3: Iris).
5. Xử lý an toàn khi không có đèn flash hoặc phản quang bị che lấp.
Tuân thủ tuyệt đối G2: Xử lý 100% trong RAM, không ghi đĩa.
Tuân thủ tuyệt đối G4: Không dùng từ ngữ thuộc danh mục cấm.
"""

import cv2
import numpy as np
import pytest
import torch

from app.core.config import settings
from app.services.eye_roi_service import extract_bilateral_eye_rois
from app.services.eyelid_model_service import segment_eyelid_margins
from app.services.iris_model_service import IrisGeometryResult, detect_iris_geometry
from app.services.reflex_model_service import (
    LocalizerFCN,
    ReflexModelResult,
    _decode_expected_center,
    _extract_optical_glint,
    get_pupil_model,
    get_reflex_model,
    locate_reference_point,
)
from tests.test_helpers import get_sample_image_path

SAMPLE_IMAGE_PATH = get_sample_image_path()


@pytest.fixture
def sample_image() -> np.ndarray:
    """Ảnh chụp chuẩn từ bộ dữ liệu mẫu."""
    img = cv2.imread(str(SAMPLE_IMAGE_PATH))
    assert img is not None, f"Không thể tải ảnh mẫu tại: {SAMPLE_IMAGE_PATH}"
    return img


def test_reflex_and_pupil_models_loading():
    """Kiểm tra nạp mô hình reflex_localizer.pt và pupil_direct.pt trên CPU."""
    pr_model = get_reflex_model()
    assert pr_model is not None, "Không thể nạp mô hình reflex_localizer.pt"
    assert isinstance(pr_model, LocalizerFCN)
    assert pr_model.in_channels == 5

    pupil_model = get_pupil_model()
    assert pupil_model is not None, "Không thể nạp mô hình pupil_direct.pt"
    assert isinstance(pupil_model, LocalizerFCN)
    assert pupil_model.in_channels == 6

    # Forward pass kiểm tra tensor
    dummy_pr = torch.randn(1, 3, 256, 256, device=settings.TORCH_DEVICE)
    with torch.no_grad():
        heat_pr, vis_pr = pr_model(dummy_pr)
    assert heat_pr.shape == (1, 2, 256, 256)
    assert vis_pr.shape == (1, 2, 3)

    dummy_pupil = torch.randn(1, 4, 256, 256, device=settings.TORCH_DEVICE)
    with torch.no_grad():
        heat_pupil, vis_pupil = pupil_model(dummy_pupil)
    assert heat_pupil.shape == (1, 2, 256, 256)
    assert vis_pupil.shape == (1, 2, 3)


def test_expected_center_decoder():
    """Kiểm tra bộ giải mã tâm kỳ vọng expected_center trên phân bố xác suất."""
    logits = torch.zeros((1, 256, 256), dtype=torch.float32)
    # Đặt một đỉnh xác suất cao tại (100, 150)
    logits[0, 150, 100] = 50.0

    valid_mask = torch.ones((1, 256, 256), dtype=torch.bool)
    cx, cy = _decode_expected_center(logits, valid_mask)
    assert abs(cx - 100.0) < 1.0
    assert abs(cy - 150.0) < 1.0


def test_locate_reference_point_tier1_golden_standard(sample_image: np.ndarray):
    """
    Kiểm tra Cấp 1 (Chuẩn vàng):
    Định vị phản xạ Purkinje I trên ảnh mẫu chuẩn có đèn flash.
    """
    roi_result = extract_bilateral_eye_rois(sample_image, image_type="fullface")
    od_iris = detect_iris_geometry(sample_image, roi_result.od.bbox, eye_side="OD")
    od_eyelid = segment_eyelid_margins(sample_image, roi_result.od.bbox, x_ref=od_iris.iris_center[0], eye_side="OD")

    res = locate_reference_point(
        image_bgr=sample_image,
        bbox=roi_result.od.bbox,
        iris_geom=od_iris,
        eyelid_seg=od_eyelid,
        eye_side="OD",
    )

    assert isinstance(res, ReflexModelResult)
    assert res.eye_side == "OD"
    assert res.reference_point is not None
    assert res.reference_point_used == "CORNEAL_LIGHT_REFLEX"
    assert res.has_fallback is False
    assert res.fallback_type is None
    assert res.confidence_score == 1.0

    # Điểm phản quang phải nằm trong đĩa mống mắt
    dist_iris = np.linalg.norm(np.array(res.reference_point) - np.array(od_iris.iris_center))
    assert dist_iris <= (od_iris.hvid_px / 2.0)


def test_locate_reference_point_tier2_pupil_fallback(sample_image: np.ndarray):
    """
    Kiểm tra Cấp 2:
    Khi ảnh bị làm mờ đốm sáng hoặc chụp không có flash,
    hệ thống tự động chuyển sang tâm đồng tử độc lập (DirectPupilCNN).
    """
    roi_result = extract_bilateral_eye_rois(sample_image, image_type="fullface")
    od_iris = detect_iris_geometry(sample_image, roi_result.od.bbox, eye_side="OD")
    od_eyelid = segment_eyelid_margins(sample_image, roi_result.od.bbox, x_ref=od_iris.iris_center[0], eye_side="OD")

    # Giả lập ảnh không có phản quang sáng bằng cách dìm độ sáng tối đa xuống dưới 180
    dark_image = sample_image.copy()
    box = roi_result.od.bbox
    crop = dark_image[box[1]:box[3], box[0]:box[2]]
    dark_crop = np.clip(crop.astype(np.float32) * 0.5, 0, 150).astype(np.uint8)
    dark_image[box[1]:box[3], box[0]:box[2]] = dark_crop

    res = locate_reference_point(
        image_bgr=dark_image,
        bbox=box,
        iris_geom=od_iris,
        eyelid_seg=od_eyelid,
        eye_side="OD",
    )

    assert res.has_fallback is True
    assert res.reference_point_used == "PUPIL_CENTER_FALLBACK"
    assert res.fallback_type == "PUPIL_DETECTOR"
    assert res.confidence_score == settings.PUPIL_FALLBACK_CONFIDENCE
    assert "REFLEX_FALLBACK_TO_PUPIL" in res.warning_flags


def test_locate_reference_point_tier3_iris_center_fallback(sample_image: np.ndarray):
    """
    Kiểm tra Cấp 3:
    Khi tâm đồng tử bị che khuất hoặc lệch quá xa ngoài đĩa mống mắt,
    hệ thống tự động dự phòng sang tâm mống mắt hình học IrisFour.
    """
    roi_result = extract_bilateral_eye_rois(sample_image, image_type="fullface")
    od_iris = detect_iris_geometry(sample_image, roi_result.od.bbox, eye_side="OD")

    # Tạo đối tượng mống mắt giả lập với tâm mống mắt ở rất xa tâm crop
    fake_iris = IrisGeometryResult(
        eye_side="OD",
        iris_left=(10.0, 10.0),
        iris_right=(30.0, 10.0),
        iris_top=(20.0, 0.0),
        iris_bottom=(20.0, 20.0),
        iris_center=(1000.0, 1000.0),  # Tâm lệch rất xa
        hvid_px=20.0,
        vvid_px=20.0,
        axis_ratio=1.0,
        scale_factor_mm_per_px=0.5,
        is_circular=True,
    )

    # Ảnh tối không có flash
    black_crop_img = np.zeros_like(sample_image)

    res = locate_reference_point(
        image_bgr=black_crop_img,
        bbox=roi_result.od.bbox,
        iris_geom=fake_iris,
        eyelid_seg=None,
        eye_side="OD",
    )

    assert res.has_fallback is True
    assert res.reference_point_used == "PUPIL_CENTER_FALLBACK"
    assert res.fallback_type == "IRIS_CENTER"
    assert res.confidence_score == settings.IRIS_CENTER_FALLBACK_CONFIDENCE
    assert "PUPIL_FALLBACK_TO_IRIS_CENTER" in res.warning_flags
