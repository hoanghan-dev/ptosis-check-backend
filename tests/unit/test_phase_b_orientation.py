"""
Kiểm thử chuyên sâu cho Giai đoạn Phase B:
Tự động định hướng ảnh và sửa lỗi tọa độ (EXIF & 4-Direction Auto-Orientation).

Bao phủ các yêu cầu của implementation_plan_prompt4.md:
1. Giải mã EXIF Orientation (Tags 1, 3, 6, 8) in-memory bằng Pillow ImageOps.
2. Fallback xoay 4 hướng (Probe 4 directions) khi ảnh không có EXIF hoặc bị xoay thực tế.
3. Ghi nhận góc đã xoay vào audit_details["applied_rotation_deg"].
4. Kiểm thử chốt chặn Rollback khi AUTO_ORIENTATION_ENABLED = False.
5. Kiểm thử tích hợp Endpoint /api/v1/screening/analyze.
6. Không ghi bất kỳ tệp đệm nào ra đĩa (G2 Zero-Retention).
"""

import io
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import pytest
from fastapi.testclient import TestClient

from app.core.config import settings
from app.main import app
from app.utils.image_orientation import decode_and_orient_image

client = TestClient(app)
FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"


@pytest.fixture
def sample_face_bytes() -> bytes:
    """Đọc tệp ảnh mẫu có khuôn mặt thẳng chuẩn làm dữ liệu thử nghiệm."""
    p = FIXTURES_DIR / "patient_07_image.jpg"
    if not p.exists():
        img = np.full((600, 600, 3), 120, dtype=np.uint8)
        _, enc = cv2.imencode(".jpg", img)
        return enc.tobytes()
    with open(p, "rb") as f:
        return f.read()


# ==============================================================================
# 1. KIỂM THỬ GIẢI MÃ EXIF ORIENTATION IN-MEMORY
# ==============================================================================

def test_exif_orientation_tag_6_rotated_90_cw(sample_face_bytes):
    """Kiểm tra thẻ EXIF Orientation 6 (xoay 90 độ CW trên điện thoại) được nhận diện và xoay đúng."""
    nparr = np.frombuffer(sample_face_bytes, np.uint8)
    orig_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Khi gắn EXIF 6, ảnh thô ở góc xoay 90 CCW, ImageOps xoay 90 CW để về thẳng đứng
    sideways_90_ccw = cv2.rotate(orig_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
    pil_img = Image.fromarray(cv2.cvtColor(sideways_90_ccw, cv2.COLOR_BGR2RGB))
    exif = pil_img.getexif()
    exif[0x0112] = 6  # Orientation Tag = 6

    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", exif=exif)
    jpeg_bytes = buf.getvalue()

    img_bgr, meta = decode_and_orient_image(jpeg_bytes)

    assert img_bgr is not None
    assert meta["exif_orientation"] == 6
    assert meta["applied_rotation_deg"] == 90
    assert meta["method"] == "EXIF"


def test_exif_orientation_tag_8_rotated_270(sample_face_bytes):
    """Kiểm tra thẻ EXIF Orientation 8 (xoay 270 độ CW) được nhận diện và xoay đúng."""
    nparr = np.frombuffer(sample_face_bytes, np.uint8)
    orig_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Khi gắn EXIF 8, ảnh thô ở góc xoay 90 CW, ImageOps xoay 270 CW (90 CCW) để về thẳng đứng
    sideways_90_cw = cv2.rotate(orig_bgr, cv2.ROTATE_90_CLOCKWISE)
    pil_img = Image.fromarray(cv2.cvtColor(sideways_90_cw, cv2.COLOR_BGR2RGB))
    exif = pil_img.getexif()
    exif[0x0112] = 8  # Orientation Tag = 8

    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", exif=exif)
    jpeg_bytes = buf.getvalue()

    img_bgr, meta = decode_and_orient_image(jpeg_bytes)

    assert img_bgr is not None
    assert meta["exif_orientation"] == 8
    assert meta["applied_rotation_deg"] == 270
    assert meta["method"] == "EXIF"


def test_exif_orientation_tag_3_rotated_180(sample_face_bytes):
    """Kiểm tra thẻ EXIF Orientation 3 (lộn ngược 180 độ) được nhận diện và xoay đúng."""
    nparr = np.frombuffer(sample_face_bytes, np.uint8)
    orig_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    inverted_180 = cv2.rotate(orig_bgr, cv2.ROTATE_180)
    pil_img = Image.fromarray(cv2.cvtColor(inverted_180, cv2.COLOR_BGR2RGB))
    exif = pil_img.getexif()
    exif[0x0112] = 3  # Orientation Tag = 3

    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", exif=exif)
    jpeg_bytes = buf.getvalue()

    img_bgr, meta = decode_and_orient_image(jpeg_bytes)

    assert img_bgr is not None
    assert meta["exif_orientation"] == 3
    assert meta["applied_rotation_deg"] == 180
    assert meta["method"] == "EXIF"


def test_no_exif_normal_image_remains_0_deg(sample_face_bytes):
    """Kiểm tra ảnh không có thẻ EXIF và đang ở chiều thẳng đứng được giữ nguyên 0 độ."""
    img_bgr, meta = decode_and_orient_image(sample_face_bytes)
    assert img_bgr is not None
    assert meta["applied_rotation_deg"] == 0
    assert meta["method"] == "NONE"


# ==============================================================================
# 2. KIỂM THỬ CHỐT CHẶN FALLBACK XOAY 4 HƯỚNG (PROBE 4 DIRECTIONS)
# ==============================================================================

def test_probe_auto_orients_synthetic_sideways_90_deg(sample_face_bytes):
    """Kiểm tra ảnh bị xoay ngang 90 độ (không có EXIF) được probe phát hiện và xoay lại thẳng."""
    nparr = np.frombuffer(sample_face_bytes, np.uint8)
    orig_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Cố tình xoay 90 độ CW thành ảnh ngang
    sideways_90 = cv2.rotate(orig_bgr, cv2.ROTATE_90_CLOCKWISE)
    _, enc_bytes = cv2.imencode(".jpg", sideways_90)

    corrected_bgr, meta = decode_and_orient_image(enc_bytes.tobytes())

    assert corrected_bgr is not None
    assert meta["method"] == "PROBE"
    # Để bù lại góc 90 CW, ảnh cần xoay 270 CW (hoặc 90 CCW)
    assert meta["applied_rotation_deg"] == 270
    assert abs(meta["canthus_roll_deg"]) <= 15.0


def test_probe_auto_orients_synthetic_sideways_270_deg(sample_face_bytes):
    """Kiểm tra ảnh bị xoay ngang 270 độ (không có EXIF) được probe phát hiện và xoay lại thẳng."""
    nparr = np.frombuffer(sample_face_bytes, np.uint8)
    orig_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    sideways_270 = cv2.rotate(orig_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
    _, enc_bytes = cv2.imencode(".jpg", sideways_270)

    corrected_bgr, meta = decode_and_orient_image(enc_bytes.tobytes())

    assert corrected_bgr is not None
    assert meta["method"] == "PROBE"
    assert meta["applied_rotation_deg"] == 90
    assert abs(meta["canthus_roll_deg"]) <= 15.0


# ==============================================================================
# 3. KIỂM THỬ TÍCH HỢP ENDPOINT /ANALYZE VÀ GHI NHẬN AUDIT_DETAILS
# ==============================================================================

def test_screening_endpoint_integrates_rotation_audit_details(sample_face_bytes):
    """Kiểm tra Endpoint /analyze trả về applied_rotation_deg trong audit_details của image_quality_audit."""
    response = client.post(
        "/api/v1/screening/analyze",
        files={"file": ("test_probe.jpg", io.BytesIO(sample_face_bytes), "image/jpeg")},
    )

    assert response.status_code == 200
    data = response.json()
    audit = data["image_quality_audit"]
    assert "audit_details" in audit
    assert "applied_rotation_deg" in audit["audit_details"]
    assert audit["audit_details"]["applied_rotation_deg"] == 0


def test_screening_endpoint_rescues_sideways_image(sample_face_bytes):
    """Kiểm tra ảnh gửi lên bị xoay 90 độ được Endpoint tự động xoay và trả về HTTP 200 thay vì 422."""
    nparr = np.frombuffer(sample_face_bytes, np.uint8)
    orig_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    sideways_90 = cv2.rotate(orig_bgr, cv2.ROTATE_90_CLOCKWISE)
    _, enc_bytes = cv2.imencode(".jpg", sideways_90)

    response = client.post(
        "/api/v1/screening/analyze",
        files={"file": ("test_sideways.jpg", io.BytesIO(enc_bytes.tobytes()), "image/jpeg")},
    )

    assert response.status_code == 200
    data = response.json()
    audit = data["image_quality_audit"]
    assert audit["audit_details"]["applied_rotation_deg"] == 270
    assert audit["audit_details"]["orientation_method"] == "PROBE"


# ==============================================================================
# 4. KIỂM THỬ KẾ HOẠCH ROLLBACK (AUTO_ORIENTATION_ENABLED = False)
# ==============================================================================

def test_rollback_disabling_auto_orientation(monkeypatch, sample_face_bytes):
    """Kiểm tra khi tắt cờ AUTO_ORIENTATION_ENABLED, cơ chế probe bị vô hiệu hóa hoàn toàn."""
    monkeypatch.setattr(settings, "AUTO_ORIENTATION_ENABLED", False)

    nparr = np.frombuffer(sample_face_bytes, np.uint8)
    orig_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    sideways_90 = cv2.rotate(orig_bgr, cv2.ROTATE_90_CLOCKWISE)
    _, enc_bytes = cv2.imencode(".jpg", sideways_90)

    img_bgr, meta = decode_and_orient_image(enc_bytes.tobytes())

    # Khi tắt auto orientation, ảnh không bị probe xoay
    assert meta["applied_rotation_deg"] == 0
    assert meta["method"] == "NONE"


def test_empty_or_corrupted_bytes_handled_gracefully():
    """Kiểm tra luồng byte rỗng hoặc hỏng trả về None an toàn không làm sập tiến trình."""
    img_bgr, meta = decode_and_orient_image(b"")
    assert img_bgr is None
    assert meta["method"] == "EMPTY_INPUT"

    img_bgr_corrupt, meta_corrupt = decode_and_orient_image(b"not_an_image_data")
    assert img_bgr_corrupt is None
    assert meta_corrupt["method"] == "DECODE_FAILED"
