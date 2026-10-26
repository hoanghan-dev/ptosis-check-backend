"""
Bộ kiểm thử độc lập cho Cổng Sàng lọc Chất lượng Hình ảnh Lâm sàng (Module M2).
Nguồn chân lý (Oracle):
- docs/04-measurement-specification.md: § 10 (Bảng tham số và ngưỡng Cổng Chất lượng)
- docs/02-requirements.md: FR-007, FR-008, FR-009, FR-010, FR-011

Mục tiêu chính:
Khẳng định mọi vi phạm chất lượng lâm sàng (tối, cháy sáng, nghiêng đầu, nhắm mắt, chạm mép)
đều bị Cổng Chất lượng từ chối với mã lỗi chính xác theo hợp đồng BR-001, BR-002, BR-003.
"""

from pathlib import Path
import cv2
import numpy as np
import pytest
from app.core.config import settings
from app.services.quality_gate_service import evaluate_quality

_FIXTURES_DIR = Path(__file__).resolve().parent.parent / "tests" / "fixtures"


class TestQualityGateInvariants:
    """Kiểm định tính nghiêm ngặt của Cổng Chất lượng M2."""

    def test_reject_empty_or_invalid_image_array(self):
        """Khẳng định mảng rỗng hoặc None bị từ chối với IMAGE_UNREADABLE."""
        res_none = evaluate_quality(None)
        assert res_none.passed is False
        assert res_none.rejection_reason == "IMAGE_UNREADABLE"

        empty_arr = np.array([], dtype=np.uint8)
        res_empty = evaluate_quality(empty_arr)
        assert res_empty.passed is False
        assert res_empty.rejection_reason == "IMAGE_UNREADABLE"

    def test_reject_low_resolution_image_dimensions(self):
        """Khẳng định ảnh có kích thước dưới 200x200 px bị từ chối với LOW_RESOLUTION."""
        tiny_img = np.full((150, 150, 3), 120, dtype=np.uint8)
        res = evaluate_quality(tiny_img)
        assert res.passed is False
        assert res.rejection_reason == "LOW_RESOLUTION"

    def test_reject_too_dark_illumination(self):
        """
        [IT-016 / BR-001]
        Ảnh có độ sáng trung bình < 40 (mean_brightness < MIN_BRIGHTNESS)
        bắt buộc bị từ chối với mã lý do TOO_DARK.
        """
        dark_img = np.full((300, 300, 3), 25, dtype=np.uint8)
        res = evaluate_quality(dark_img)
        assert res.passed is False
        assert res.rejection_reason == "TOO_DARK"
        assert res.mean_illumination is not None
        assert res.mean_illumination < settings.MIN_BRIGHTNESS

    def test_reject_too_bright_illumination(self):
        """
        [BR-001]
        Ảnh bị cháy sáng (mean_brightness > 220)
        bắt buộc bị từ chối với mã lý do TOO_BRIGHT.
        """
        bright_img = np.full((300, 300, 3), 240, dtype=np.uint8)
        res = evaluate_quality(bright_img)
        assert res.passed is False
        assert res.rejection_reason == "TOO_BRIGHT"
        assert res.mean_illumination is not None
        assert res.mean_illumination > settings.MAX_BRIGHTNESS

    def test_reject_eyes_closed_fixture(self):
        """
        [BR-002]
        Ảnh nhắm mắt từ fixture (full_face_eyes_closed_sample.jpg)
        bắt buộc bị Cổng chất lượng từ chối với lý do EYES_CLOSED.
        """
        p = _FIXTURES_DIR / "full_face_eyes_closed_sample.jpg"
        if not p.exists():
            pytest.skip("Fixture full_face_eyes_closed_sample.jpg không tồn tại")

        img = cv2.imread(str(p))
        assert img is not None, "Không thể đọc fixture ảnh nhắm mắt"

        res = evaluate_quality(img)
        assert res.passed is False
        assert res.rejection_reason == "EYES_CLOSED", (
            f"Kỳ vọng từ chối EYES_CLOSED, nhưng nhận: {res.rejection_reason}"
        )

    def test_reject_head_roll_exceeded_fixture(self):
        """
        [BR-001]
        Ảnh nghiêng đầu quá 5 độ (patient_17_image.jpg hoặc patient_27_image.jpg)
        bắt buộc bị từ chối với HEAD_ROLL_EXCEEDED.
        """
        p = _FIXTURES_DIR / "patient_17_image.jpg"
        if not p.exists():
            pytest.skip("Fixture patient_17_image.jpg không tồn tại")

        img = cv2.imread(str(p))
        assert img is not None

        res = evaluate_quality(img)
        assert res.passed is False
        assert res.rejection_reason == "HEAD_ROLL_EXCEEDED", (
            f"Kỳ vọng HEAD_ROLL_EXCEEDED, nhận: {res.rejection_reason}"
        )

    def test_reject_head_yaw_exceeded_fixture(self):
        """
        [BR-001]
        Ảnh quay đầu sang bên quá 10 độ (patient_03_image.jpg hoặc patient_28_image.jpg)
        bắt buộc bị từ chối với HEAD_YAW_EXCEEDED.
        """
        p = _FIXTURES_DIR / "patient_03_image.jpg"
        if not p.exists():
            pytest.skip("Fixture patient_03_image.jpg không tồn tại")

        img = cv2.imread(str(p))
        assert img is not None

        res = evaluate_quality(img)
        assert res.passed is False
        assert res.rejection_reason == "HEAD_YAW_EXCEEDED", (
            f"Kỳ vọng HEAD_YAW_EXCEEDED, nhận: {res.rejection_reason}"
        )

    def test_reject_head_pitch_exceeded_fixture(self):
        """
        [BR-001]
        Ảnh ngửa/cúi đầu quá 10 độ (patient_01_image.jpg hoặc patient_08_image.jpg)
        bắt buộc bị từ chối với HEAD_PITCH_EXCEEDED.
        """
        p = _FIXTURES_DIR / "patient_01_image.jpg"
        if not p.exists():
            pytest.skip("Fixture patient_01_image.jpg không tồn tại")

        img = cv2.imread(str(p))
        assert img is not None

        res = evaluate_quality(img)
        assert res.passed is False
        assert res.rejection_reason == "HEAD_PITCH_EXCEEDED", (
            f"Kỳ vọng HEAD_PITCH_EXCEEDED, nhận: {res.rejection_reason}"
        )
