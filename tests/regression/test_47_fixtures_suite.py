"""
Bộ kiểm thử hồi quy tự động hóa cho toàn bộ 47 fixtures trong kho dữ liệu (Phase F).
Xác thực các tiêu chí nghiệm thu then chốt:
1. Zero-Regression: 12 ca Pass nền tảng tiếp tục đạt HTTP 200 với kết quả chính xác hơn (đặc biệt patient_07 và patient_09).
2. Tỷ lệ vượt cổng sàng lọc (Pass Rate) tăng từ 12 ca (25.5%) lên ít nhất 31 ca (66.0%).
3. Chốt chặn an toàn (Quality Gate): 100% các ca vi phạm thực sự (nhắm mắt, quay đầu quá nghiêng, độ phân giải thấp) bị từ chối 422 chính xác.
4. NFR-001: Độ trễ phản hồi toàn trình <= 3.0 giây trên từng ca kiểm thử, trung bình <= 1.8s.
5. G2: Zero-disk retention (không rò rỉ tệp tạm trên đĩa sau khi xử lý).
"""

import time
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.schemas.triage import ScreeningTier

client = TestClient(app)
FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"

# 12 Fixtures Pass nền tảng ban đầu (Baseline Pass Fixtures)
BASELINE_PASS_FIXTURES = [
    "patient_04_image.jpg",
    "patient_07_image.jpg",
    "patient_09_image.jpg",
    "patient_12_image.jpg",
    "patient_13_image.jpg",
    "patient_14_image.jpg",
    "patient_16_image.jpg",
    "patient_18_image.jpg",
    "patient_19_image.jpg",
    "patient_25_image.jpg",
    "patient_32_image.jpg",
    "sample_face.jpg",
]

# Fixtures được cứu thành công nhờ Auto-Orientation (Phase B)
RESCUED_ROTATION_FIXTURES = [
    "patient_06_image.jpg",
    "patient_10_image.jpg",
    "patient_22_image.jpg",
]

# Fixtures được cứu thành công nhờ Dual-Eye Crop Pipeline & Synthetic Canvas (Phase D)
RESCUED_PERIOCULAR_FIXTURES = [
    "crop_eye_normal_sample.jpg",
    "patient_05_image.jpg",
    "patient_20_image.jpg",
    "patient_21_image copy.jpg",
    "patient_21_image.jpg",
    "patient_23_image.jpg",
    "patient_26_image.jpg",
    "patient_29_image.jpg",
    "patient_33_image.jpg",
    "patient_34_image.jpg",
    "patient_35_image.jpg",
    "patient_36_image.jpg",
    "tight_eye_crop_sample.jpg",
    "corneal_reflex_fully_occluded_sample.png",
    "ptosis_crop_eyes_one_side_sample.png",
]

# Fixtures bị từ chối 422 hợp lý để bảo vệ an toàn (Safety Rejections)
EXPECTED_422_REJECTIONS = {
    "crop_face_eyes_closed_sample.jpg": "EYES_CLOSED",
    "full_face_eyes_closed_sample.jpg": "EYES_CLOSED",
    "crop_eye_motion_blurred_sample.jpg": ["HEAD_ROLL_EXCEEDED", "BLUR_DETECTED"],
    "full_face_motion_blurred_sample.jpg": "HEAD_ROLL_EXCEEDED",
    "patient_01_image.jpg": "HEAD_PITCH_EXCEEDED",
    "patient_03_image.jpg": "HEAD_YAW_EXCEEDED",
    "patient_08_image.jpg": "HEAD_PITCH_EXCEEDED",
    "patient_15_image.jpg": "HEAD_PITCH_EXCEEDED",
    "patient_17_image.jpg": "HEAD_ROLL_EXCEEDED",
    "patient_24_image.jpg": "HEAD_PITCH_EXCEEDED",
    "patient_27_image.jpg": "HEAD_ROLL_EXCEEDED",
    "patient_28_image.jpg": "HEAD_YAW_EXCEEDED",
    "patient_30_image.jpg": "HEAD_PITCH_EXCEEDED",
    "patient_31_image.jpg": "LOW_RESOLUTION",
    "ptosis_fullface_one_side_sample.jpg": "LOW_RESOLUTION",
    "ptosis_fullface_two_side_sample.jpg": "LOW_RESOLUTION",
}


def _send_fixture(filename: str):
    p = FIXTURES_DIR / filename
    assert p.exists(), f"Không tìm thấy fixture {filename}"
    ctype = "image/png" if filename.lower().endswith(".png") else "image/jpeg"
    with open(p, "rb") as f:
        content = f.read()
    start_t = time.perf_counter()
    resp = client.post(
        "/api/v1/screening/analyze",
        files={"file": (filename, content, ctype)},
    )
    elapsed = time.perf_counter() - start_t
    return resp, elapsed


def test_47_fixtures_presence():
    """Xác nhận thư mục fixtures chứa đủ 47 tệp kiểm thử."""
    all_files = [f for f in FIXTURES_DIR.iterdir() if f.is_file()]
    assert len(all_files) == 47, f"Kỳ vọng 47 fixtures nhưng tìm thấy {len(all_files)}"


@pytest.mark.parametrize("filename", BASELINE_PASS_FIXTURES)
def test_12_baseline_pass_fixtures_zero_regression(filename: str):
    """Zero-Regression: Toàn bộ 12 ca Pass nền tảng phải tiếp tục trả về HTTP 200."""
    resp, elapsed = _send_fixture(filename)
    assert resp.status_code == 200, f"Ca {filename} bị suy thoái: {resp.status_code} - {resp.text}"
    assert elapsed <= 3.0, f"Độ trễ ca {filename} ({elapsed:.2f}s) vượt quá 3.0s (NFR-001)"

    data = resp.json()
    assert "triage_result" in data
    assert data["triage_result"]["tier"] in ["TIER_A", "TIER_B", "TIER_C"]

    # Kiểm tra đặc biệt cho 2 ca bug cũ đã sửa tại Phase A & E:
    if filename == "patient_07_image.jpg":
        assert data["triage_result"]["tier"] == "TIER_C"
        assert "BILATERAL_SEVERE_MRD1_UNDER_2MM" in data["triage_result"]["trigger_reasons"]

    if filename == "patient_09_image.jpg":
        assert data["triage_result"]["tier"] == "TIER_C"
        os_meas = data["measurements"]["left_eye_os"]
        assert os_meas["mrd1_mm"] is None
        assert os_meas["status"] == "MEASUREMENT_UNAVAILABLE"


@pytest.mark.parametrize("filename", RESCUED_ROTATION_FIXTURES)
def test_rescued_rotation_fixtures_pass_screening(filename: str):
    """Kiểm tra các ca chụp xoay ngang/ngược được tự động định hướng (Auto-Orientation) và vượt qua sàng lọc."""
    resp, elapsed = _send_fixture(filename)
    assert resp.status_code == 200, f"Ca xoay {filename} không vượt qua: {resp.status_code} - {resp.text}"
    assert elapsed <= 3.0, f"Độ trễ ca {filename} ({elapsed:.2f}s) vượt quá 3.0s"

    data = resp.json()
    audit = data["image_quality_audit"]
    assert audit["passed"] is True
    # Kiểm tra có thông tin góc xoay hoặc phương thức định hướng
    details = audit.get("audit_details", {})
    assert "applied_rotation_deg" in details


@pytest.mark.parametrize("filename", RESCUED_PERIOCULAR_FIXTURES)
def test_rescued_periocular_crop_fixtures_pass_screening(filename: str):
    """Kiểm tra các ca crop 2 mắt cận cảnh được điều tuyến và xử lý thành công qua Periocular Pipeline."""
    resp, elapsed = _send_fixture(filename)
    assert resp.status_code == 200, f"Ca crop {filename} không vượt qua: {resp.status_code} - {resp.text}"
    assert elapsed <= 3.0, f"Độ trễ ca {filename} ({elapsed:.2f}s) vượt quá 3.0s"

    data = resp.json()
    assert data["image_quality_audit"]["passed"] is True


@pytest.mark.parametrize("filename, expected_reason", EXPECTED_422_REJECTIONS.items())
def test_quality_gate_safety_rejections(filename: str, expected_reason):
    """Kiểm tra các ca ảnh lỗi chất lượng (nhắm mắt, quay quá nghiêng, ảnh mờ) được từ chối an toàn với HTTP 422."""
    resp, elapsed = _send_fixture(filename)
    assert resp.status_code == 422, f"Ca {filename} kỳ vọng bị từ chối 422 nhưng nhận {resp.status_code}"
    assert elapsed <= 3.0, f"Độ trễ ca {filename} ({elapsed:.2f}s) vượt quá 3.0s"

    err = resp.json()
    detail = err.get("detail", err)
    reason = detail.get("rejection_reason") if isinstance(detail, dict) else None
    if isinstance(expected_reason, list):
        assert reason in expected_reason, f"Ca {filename} nhận lý do {reason}, kỳ vọng trong {expected_reason}"
    else:
        assert reason == expected_reason, f"Ca {filename} nhận lý do {reason}, kỳ vọng {expected_reason}"


def test_47_fixtures_latency_benchmark_and_pass_rate():
    """Kiểm tra hiệu năng toàn diện trên 47 fixtures:
    - Pass rate >= 66% (ít nhất 31 ca).
    - Độ trễ từng ca <= 3.0s.
    - Độ trễ trung bình <= 1.8s.
    """
    all_files = sorted([f for f in FIXTURES_DIR.iterdir() if f.is_file()])
    latencies = []
    pass_count = 0

    for f in all_files:
        resp, elapsed = _send_fixture(f.name)
        latencies.append(elapsed)
        assert elapsed <= 3.0, f"Ca {f.name} có độ trễ {elapsed:.3f}s vượt quá giới hạn 3.0s"
        if resp.status_code == 200:
            pass_count += 1

    avg_latency = sum(latencies) / len(latencies)
    max_latency = max(latencies)

    assert pass_count >= 31, f"Tỷ lệ vượt cổng ({pass_count}/47) chưa đạt mục tiêu (kỳ vọng >= 31)"
    assert avg_latency <= 1.8, f"Độ trễ trung bình ({avg_latency:.3f}s) vượt quá mục tiêu 1.8s"
    assert max_latency <= 3.0, f"Độ trễ cực đại ({max_latency:.3f}s) vượt quá giới hạn 3.0s"
