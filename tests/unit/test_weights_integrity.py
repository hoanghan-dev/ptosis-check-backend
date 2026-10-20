"""
Kiểm thử tính toàn vẹn và hợp thức của các tệp trọng số mô hình học sâu (Phase 1).
Tuân thủ tuyệt đối quy tắc không dùng từ ngữ thuộc danh mục cấm (G4).
"""

import hashlib
from pathlib import Path
import pytest
import torch

from app.core.config import settings

# Bảng mã băm SHA256 và kích thước chuẩn đã kiểm toán của 5 mô hình học sâu tái sử dụng
EXPECTED_CHECKPOINTS = {
    "eyelid_unet.pt": {
        "size_bytes": 489445,
        "sha256": "54cb7856f6b91506266b2792354155ee0fd0e3ec566855c4a0b8f7b6a0488423",
        "description": "Mô hình U-Net phân đoạn bờ mi trên, bờ mi dưới và khe mi",
    },
    "iris_four.pt": {
        "size_bytes": 1482093,
        "sha256": "af713bb8b396a7be831b904206c96b12ca8258ada1c587edaead9f638fd8940c",
        "description": "Mô hình hồi quy 4 điểm cực mống mắt và xác lập thước đo HVID",
    },
    "reflex_localizer.pt": {
        "size_bytes": 483103,
        "sha256": "cbcd29ef7065dfc28be3f09cca399395db2087bb1f87a51413719128748a4f44",
        "description": "Mô hình định vị phản xạ giác mạc Purkinje I và tâm đồng tử sơ bộ",
    },
    "pupil_direct.pt": {
        "size_bytes": 1460519,
        "sha256": "b275bd844a5ad0638898b1d7e38fa497029de8776964504229019e4a2db312d7",
        "description": "Mô hình định vị tâm đồng tử dự phòng với hàm tiên nghiệm mống mắt mềm",
    },
    "eye_crop_seed.pt": {
        "size_bytes": 496513,
        "sha256": "cc1e9558be005b809b0d5ee79644a56bfe6d8342d251e748d32230b6530d9941",
        "description": "Mô hình hỗ trợ cắt và chuẩn hóa hộp bao ROI mắt độc lập",
    },
}


def test_weights_directory_configuration():
    """Xác thực thư mục quản lý trọng số được cấu hình chính xác và tồn tại."""
    assert isinstance(settings.WEIGHTS_DIR, Path)
    assert settings.WEIGHTS_DIR.exists(), f"Thư mục trọng số không tồn tại: {settings.WEIGHTS_DIR}"
    assert settings.WEIGHTS_DIR.is_dir(), f"Đường dẫn không phải thư mục: {settings.WEIGHTS_DIR}"
    assert settings.TORCH_DEVICE == "cpu", f"Thiết bị tính toán mặc định phải là CPU: {settings.TORCH_DEVICE}"


@pytest.mark.parametrize("filename,meta", EXPECTED_CHECKPOINTS.items())
def test_checkpoint_file_existence_and_size(filename: str, meta: dict):
    """Xác thực từng tệp trọng số tồn tại với kích thước khớp tuyệt đối."""
    file_path = settings.WEIGHTS_DIR / filename
    assert file_path.exists(), f"Thiếu tệp trọng số: {filename}"
    assert file_path.is_file(), f"Đường dẫn không phải là tệp: {filename}"
    actual_size = file_path.stat().st_size
    assert actual_size == meta["size_bytes"], (
        f"Kích thước tệp {filename} không khớp: mong đợi {meta['size_bytes']}, thực tế {actual_size}"
    )


@pytest.mark.parametrize("filename,meta", EXPECTED_CHECKPOINTS.items())
def test_checkpoint_sha256_integrity(filename: str, meta: dict):
    """Xác thực mã SHA256 từng tệp trọng số khớp chuẩn xác với giá trị kiểm toán."""
    file_path = settings.WEIGHTS_DIR / filename
    with open(file_path, "rb") as f:
        actual_hash = hashlib.sha256(f.read()).hexdigest()
    assert actual_hash == meta["sha256"], (
        f"Mã SHA256 của {filename} không hợp lệ! Mong đợi: {meta['sha256']}, Thực tế: {actual_hash}"
    )


@pytest.mark.parametrize("filename", list(EXPECTED_CHECKPOINTS.keys()))
def test_checkpoint_pytorch_loadable_on_cpu(filename: str):
    """Xác thực PyTorch nạp được từng tệp checkpoint trên CPU mà không bị lỗi cấu trúc."""
    file_path = settings.WEIGHTS_DIR / filename
    checkpoint = torch.load(file_path, map_location=settings.TORCH_DEVICE, weights_only=False)
    assert isinstance(checkpoint, dict), f"Nội dung {filename} không phải là từ điển checkpoint"
    assert "state_dict" in checkpoint, f"Checkpoint {filename} không chứa trường 'state_dict'"
    state_dict = checkpoint["state_dict"]
    assert len(state_dict) > 0, f"Trọng số state_dict trong {filename} rỗng"
