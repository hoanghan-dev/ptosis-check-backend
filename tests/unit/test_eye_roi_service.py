"""
Kiểm thử đơn vị cho Dịch vụ Điều phối Vùng ROI Mắt Đa Chế Độ (Module Eye ROI Service - Phase 2).
Xác thực:
1. Kiến trúc mô hình học sâu EyeCropSeedLocalizer (eye_crop_seed.pt).
2. Giải thuật mesh_eyes bóc tách ROI từ MediaPipe FaceMesh kèm đệm an toàn.
3. Giải thuật paired_eye_rois qua Haar Cascade đối xứng ngang.
4. Tinh chỉnh hộp bao nhãn cầu bằng mô hình hạt giống refine_eye_bbox_with_seed.
5. Điều phối bóc tách toàn diện extract_bilateral_eye_rois cho cả 2 chế độ 'fullface' và 'eye_crop'.
Tuân thủ tuyệt đối G2: Xử lý 100% trong RAM, không ghi đĩa.
Tuân thủ tuyệt đối G4: Không dùng từ ngữ thuộc danh mục cấm.
"""

import cv2
import numpy as np
import pytest
import torch

from app.core.config import settings
from app.services.eye_roi_service import (
    EyeCropSeedLocalizer,
    EyeRoi,
    BilateralEyeRoiResult,
    extract_bilateral_eye_rois,
    get_eye_crop_seed_model,
    mesh_eyes,
    paired_eye_rois,
    refine_eye_bbox_with_seed,
)
from tests.test_helpers import get_sample_image_path

SAMPLE_IMAGE_PATH = get_sample_image_path()


@pytest.fixture
def sample_fullface_image() -> np.ndarray:
    """Ảnh chụp toàn mặt chuẩn từ bộ dữ liệu mẫu."""
    img = cv2.imread(str(SAMPLE_IMAGE_PATH))
    assert img is not None, f"Không thể tải ảnh mẫu tại: {SAMPLE_IMAGE_PATH}"
    return img


@pytest.fixture
def sample_eye_crop_image(sample_fullface_image: np.ndarray) -> np.ndarray:
    """Ảnh cắt cận cảnh vùng 2 mắt từ ảnh mẫu toàn mặt."""
    h, w = sample_fullface_image.shape[:2]
    y1, y2 = int(h * 0.30), int(h * 0.58)
    x1, x2 = int(w * 0.18), int(w * 0.82)
    return sample_fullface_image[y1:y2, x1:x2].copy()


def test_eye_crop_seed_model_architecture_and_forward():
    """Kiểm tra khởi tạo và lan truyền tiến của mô hình EyeCropSeedLocalizer."""
    model = get_eye_crop_seed_model()
    assert model is not None, "Không thể nạp mô hình eye_crop_seed.pt từ thư mục weights"
    assert isinstance(model, EyeCropSeedLocalizer)

    dummy_input = torch.zeros((1, 3, 192, 192), dtype=torch.float32, device=settings.TORCH_DEVICE)
    with torch.no_grad():
        heat, vis = model(dummy_input)

    assert heat.shape == (1, 2, 192, 192), f"Kích thước heatmap không khớp: {heat.shape}"
    assert vis.shape == (1, 2, 3), f"Kích thước visibility logits không khớp: {vis.shape}"


def test_mesh_eyes_on_fullface_landmarks(sample_fullface_image: np.ndarray):
    """Kiểm tra bóc tách hộp bao OD và OS qua mesh_eyes với đệm viền an toàn."""
    import mediapipe as mp
    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as fm:
        rgb = cv2.cvtColor(sample_fullface_image, cv2.COLOR_BGR2RGB)
        res = fm.process(rgb)
        assert res.multi_face_landmarks is not None

        h, w = sample_fullface_image.shape[:2]
        od_roi, os_roi = mesh_eyes(res.multi_face_landmarks[0].landmark, w, h)

    assert od_roi.eye_side == "OD"
    assert os_roi.eye_side == "OS"

    # Kiểm tra tính hợp lệ của tọa độ
    for roi in (od_roi, os_roi):
        x0, y0, x1, y1 = roi.bbox
        assert 0 <= x0 < x1 <= w, f"Hộp bao vượt quá khung ngang: {roi.bbox}"
        assert 0 <= y0 < y1 <= h, f"Hộp bao vượt quá khung dọc: {roi.bbox}"
        assert roi.bbox_normalized[0] < roi.bbox_normalized[2]
        assert roi.bbox_normalized[1] < roi.bbox_normalized[3]

    # OD (Mắt phải) nằm ở phía bên trái ảnh nên hoành độ x nhỏ hơn OS (Mắt trái)
    assert od_roi.bbox[0] < os_roi.bbox[0]


def test_paired_eye_rois_on_crop(sample_eye_crop_image: np.ndarray):
    """Kiểm tra phát hiện cặp mắt độc lập qua Haar Cascade trên ảnh cận cảnh hai mắt."""
    pairs = paired_eye_rois(sample_eye_crop_image)
    assert pairs is not None, "Haar Cascade không phát hiện được cặp mắt trên ảnh crop chuẩn"

    od_box, os_box = pairs
    ch, cw = sample_eye_crop_image.shape[:2]

    for b in (od_box, os_box):
        x0, y0, x1, y1 = b
        assert 0 <= x0 < x1 <= cw
        assert 0 <= y0 < y1 <= ch

    # od_box nằm bên trái ảnh, os_box nằm bên phải ảnh
    assert od_box[0] < os_box[0]


def test_refine_eye_bbox_with_seed(sample_eye_crop_image: np.ndarray):
    """Kiểm tra tinh chỉnh hộp bao bằng mô hình hạt giống eye_crop_seed.pt."""
    ch, cw = sample_eye_crop_image.shape[:2]
    rough_bbox = (50, 50, min(cw - 1, 250), min(ch - 1, 250))

    refined = refine_eye_bbox_with_seed(sample_eye_crop_image, rough_bbox, "OD")
    assert isinstance(refined, tuple)
    assert len(refined) == 4
    x0, y0, x1, y1 = refined
    assert 0 <= x0 < x1 <= cw
    assert 0 <= y0 < y1 <= ch


def test_extract_bilateral_eye_rois_fullface_mode(sample_fullface_image: np.ndarray):
    """Kiểm tra hàm điều phối chính trên ảnh toàn mặt (image_type='fullface')."""
    res = extract_bilateral_eye_rois(sample_fullface_image, image_type="fullface")
    assert res.is_valid is True
    assert res.image_type == "fullface"
    assert "facemesh" in res.detection_method.lower() or "cascade" in res.detection_method.lower()

    assert res.od.eye_side == "OD"
    assert res.os.eye_side == "OS"
    assert res.od.bbox[0] < res.os.bbox[0]


def test_extract_bilateral_eye_rois_eye_crop_mode(sample_eye_crop_image: np.ndarray):
    """Kiểm tra hàm điều phối chính trên ảnh cận cảnh hai mắt (image_type='eye_crop')."""
    res = extract_bilateral_eye_rois(sample_eye_crop_image, image_type="eye_crop")
    assert res.is_valid is True
    assert res.image_type == "eye_crop"
    assert res.od.eye_side == "OD"
    assert res.os.eye_side == "OS"
    assert res.od.bbox[0] < res.os.bbox[0]


def test_extract_bilateral_eye_rois_empty_image():
    """Kiểm tra xử lý an toàn khi ảnh rỗng hoặc không hợp lệ."""
    empty = np.zeros((0, 0, 3), dtype=np.uint8)
    res = extract_bilateral_eye_rois(empty, image_type="fullface")
    assert res.is_valid is False
    assert res.detection_method == "FAILED"

    none_res = extract_bilateral_eye_rois(None, image_type="eye_crop")
    assert none_res.is_valid is False
    assert none_res.detection_method == "FAILED"
