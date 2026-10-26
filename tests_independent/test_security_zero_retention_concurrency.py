"""
Bộ kiểm thử độc lập cho Bảo mật, Zero-Retention trên Đĩa và Xử lý Đồng thời (Concurrency).
Nguồn chân lý (Oracle):
- docs/06-system-architecture.md: § 1 Nguyên tắc P1 (In-Memory Zero-Retention), § 10
- docs/02-requirements.md: FR-034, NFR-001, NFR-008
- Nghị định 13/2023/NĐ-CP (Bảo vệ dữ liệu cá nhân sinh trắc học)

Mục tiêu chính:
Khẳng định tuyệt đối không có tệp nào lưu lại trên đĩa cứng máy chủ sau khi phân tích,
và hệ thống xử lý ổn định khi nhận các request đồng thời mà không bị xung đột trạng thái hay deadlock.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
import io
import os
from pathlib import Path
from typing import Set
import pytest
from fastapi.testclient import TestClient

from app.main import app

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".bin", ".raw", ".tmp"}


def _scan_disk_files(target_dir: Path) -> Set[Path]:
    """Quét toàn bộ tệp trong thư mục mục tiêu (loại trừ cache và môi trường)."""
    found = set()
    if not target_dir.exists():
        return found
    for root, dirs, files in os.walk(target_dir):
        dirs[:] = [d for d in dirs if d not in {".venv", "node_modules", ".git", "__pycache__", ".pytest_cache"}]
        for f in files:
            p = Path(root) / f
            if p.suffix.lower() in _IMAGE_EXTENSIONS:
                found.add(p)
    return found


class TestSecurityZeroRetentionAndConcurrency:
    """Kiểm thử độ an toàn lưu trữ và độ bền đồng thời."""

    def test_zero_disk_retention_after_multiple_requests(self, api_client: TestClient, valid_face_bytes: bytes):
        """
        [IT-014 / BR-011 / FR-034]
        Khẳng định nguyên tắc Zero-Retention: Sau 10 request tải ảnh lên backend,
        không có bất kỳ tệp ảnh hoặc buffer nào được lưu lại trên đĩa cứng.
        """
        current_dir = Path(__file__).resolve().parent
        backend_dir = current_dir.parent

        files_before = _scan_disk_files(backend_dir / "app")

        # Gửi 10 request liên tiếp
        for i in range(10):
            resp = api_client.post(
                "/api/v1/screening/analyze",
                files={"file": (f"zero_retention_audit_{i}.jpg", io.BytesIO(valid_face_bytes), "image/jpeg")},
            )
            assert resp.status_code in {200, 422}

        files_after = _scan_disk_files(backend_dir / "app")
        new_files = files_after - files_before

        assert len(new_files) == 0, (
            f"VI PHẠM BẢO MẬT ZERO-RETENTION (FR-034): Phát hiện {len(new_files)} tệp "
            f"được ghi lên đĩa sau khi xử lý: {[str(p) for p in new_files]}"
        )

    def test_concurrent_screening_requests_stability(self):
        """
        [IT-019 / NFR-008]
        Kiểm tra tính an toàn luồng (Thread-Safety) khi xử lý đồng thời:
        Gửi các request phân tích đồng thời và xác minh máy chủ không bị crash bộ nhớ C++
        (Access Violation / Segfault) do các singleton MediaPipe không thread-safe.
        """
        import sys
        import os
        import subprocess
        from pathlib import Path

        backend_dir = str(Path(__file__).resolve().parent.parent)

        runner_script = f"""
import io
import sys
from pathlib import Path

backend_dir = r"{backend_dir}"
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from concurrent.futures import ThreadPoolExecutor
from fastapi.testclient import TestClient
from app.main import app

fixtures_path = Path("tests/fixtures/sample_face.jpg")
if not fixtures_path.exists():
    fixtures_path = Path("backend/tests/fixtures/sample_face.jpg")

with open(fixtures_path, "rb") as f:
    img_bytes = f.read()

def _worker(idx):
    client = TestClient(app)
    resp = client.post(
        "/api/v1/screening/analyze",
        files={{"file": (f"concurrent_{{idx}}.jpg", io.BytesIO(img_bytes), "image/jpeg")}},
    )
    return resp.status_code

with ThreadPoolExecutor(max_workers=3) as executor:
    results = list(executor.map(_worker, range(6)))

print("CONCURRENT_RESULTS:", results)
sys.exit(0)
"""
        env = dict(os.environ)
        env["PYTHONPATH"] = backend_dir
        proc = subprocess.run(
            [sys.executable, "-c", runner_script],
            capture_output=True,
            text=True,
            timeout=20,
            cwd=backend_dir,
            env=env,
        )

        assert proc.returncode == 0, (
            f"LỖI NGHIÊM TRỌNG (THREAD-SAFETY RACE CONDITION / ACCESS VIOLATION): "
            f"Tiến trình bị crash với mã lỗi {proc.returncode} khi có request đồng thời! "
            f"Nguyên nhân: MediaPipe FaceMesh là singleton toàn cục không an toàn đa luồng. "
            f"Stderr: {proc.stderr}"
        )
        assert "CONCURRENT_RESULTS:" in proc.stdout
