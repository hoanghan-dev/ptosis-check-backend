"""
Cấu hình fixtures dùng chung cho bộ kiểm thử độc lập (Independent QA Test Suite).
Tuân thủ nguyên tắc:
- Không mock logic của hệ thống.
- Sử dụng dữ liệu thật hoặc ảnh tổng hợp trực tiếp trên RAM.
- Độc lập, tự dọn dẹp, không để lại file rác.
"""

import io
from pathlib import Path
import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app

_CURRENT_DIR = Path(__file__).resolve().parent
_FIXTURES_DIR = _CURRENT_DIR.parent / "tests" / "fixtures"


@pytest.fixture(scope="session")
def api_client() -> TestClient:
    """Khởi tạo TestClient của FastAPI."""
    return TestClient(app)


@pytest.fixture(scope="session")
def valid_face_bytes() -> bytes:
    """Tải ảnh khuôn mặt chuẩn từ fixtures (nếu có) hoặc sinh ảnh tổng hợp."""
    sample_path = _FIXTURES_DIR / "sample_face.jpg"
    if sample_path.exists():
        with open(sample_path, "rb") as f:
            return f.read()
    
    # Fallback sinh ảnh trong RAM
    img = np.full((400, 400, 3), 130, dtype=np.uint8)
    cv2.circle(img, (200, 200), 100, (180, 180, 180), -1)
    success, buf = cv2.imencode(".jpg", img)
    assert success
    return buf.tobytes()


@pytest.fixture
def make_solid_image_bytes():
    """Factory tạo ảnh màu đồng nhất theo độ sáng trong RAM."""
    def _create(brightness: int, width: int = 300, height: int = 300) -> bytes:
        img = np.full((height, width, 3), brightness, dtype=np.uint8)
        success, buf = cv2.imencode(".jpg", img)
        assert success
        return buf.tobytes()
    return _create
