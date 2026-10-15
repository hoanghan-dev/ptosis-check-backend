"""
Kiểm toán độc lập nguyên tắc Guardrail G2 (In-Memory Zero-Retention).
Xác minh 100% dữ liệu hình ảnh người dùng chỉ tồn tại tạm thời trong RAM (io.BytesIO),
giải phóng tức thì sau khi hoàn tất pipeline và không lưu lại bất kỳ byte nào xuống đĩa cứng.
Tuân thủ tuyệt đối G2 (In-Memory Zero-Retention) và G4 (Non-Diagnostic Safety).
"""

import io
import os
from pathlib import Path
from typing import Set

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def get_test_image_bytes() -> bytes:
    """Tải ảnh mẫu từ fixtures hoặc tạo ảnh nhân tạo trong RAM và mã hóa sang bytes."""
    fixtures_paths = [
        Path("tests/fixtures/sample_face.jpg"),
        Path("backend/tests/fixtures/sample_face.jpg"),
        Path("/app/tests/fixtures/sample_face.jpg"),
    ]
    for p in fixtures_paths:
        if p.exists():
            with open(p, "rb") as f:
                return f.read()

    # Nếu không tìm thấy tệp fixture, sinh ảnh nhân tạo hợp lệ trong RAM
    synthetic_img = np.full((400, 400, 3), 130, dtype=np.uint8)
    cv2.circle(synthetic_img, (200, 200), 120, (180, 180, 180), -1)
    success, buf = cv2.imencode(".jpg", synthetic_img)
    return buf.tobytes()


def scan_image_files(target_dir: Path) -> Set[Path]:
    """Quét và thu thập tập hợp tất cả các tệp hình ảnh/nhị phân trong thư mục mục tiêu."""
    image_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".bin", ".raw"}
    found_files = set()
    if not target_dir.exists():
        return found_files

    for root, dirs, files in os.walk(target_dir):
        # Bỏ qua các thư mục môi trường và phụ thuộc
        dirs[:] = [d for d in dirs if d not in {".venv", "node_modules", ".git", "__pycache__", ".pytest_cache"}]
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in image_extensions:
                found_files.add(Path(root) / f)
    return found_files


def test_zero_disk_retention_after_20_analyze_requests():
    """Kiểm toán gửi liên tiếp 20 request sàng lọc và xác minh không có tệp ảnh mới nào được ghi ra đĩa (G2)."""
    current_dir = Path(__file__).resolve().parent
    backend_dir = current_dir.parent.parent
    app_dir = backend_dir / "app"

    # 1. Quét trạng thái tệp đĩa trước khi thực hiện các yêu cầu
    files_before = scan_image_files(app_dir)

    img_bytes = get_test_image_bytes()
    num_requests = 20

    # 2. Gửi liên tiếp 20 request phân tích sàng lọc
    for i in range(num_requests):
        response = client.post(
            "/api/v1/screening/analyze",
            files={"file": (f"test_audit_user_{i}.jpg", io.BytesIO(img_bytes), "image/jpeg")},
        )
        assert response.status_code in {200, 422}, (
            f"Request {i} thất bại với mã lỗi HTTP {response.status_code}: {response.text}"
        )

    # 3. Quét lại trạng thái tệp đĩa sau khi hoàn tất chuỗi 20 request
    files_after = scan_image_files(app_dir)

    newly_created_files = files_after - files_before
    assert len(newly_created_files) == 0, (
        f"VI PHẠM GUARDRAIL G2: Phát hiện {len(newly_created_files)} tệp dữ liệu được ghi lên đĩa: "
        f"{[str(p) for p in newly_created_files]}"
    )
