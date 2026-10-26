"""
Bộ kiểm thử độc lập cho Khối Đo lường MRD1, Quy đổi Tỷ lệ & Hình học (Module M6).
Nguồn chân lý (Oracle):
- docs/04-measurement-specification.md: § 4 (MRD1), § 4.4, § 4.5, § 7 (PFH), § 8, § 13
- docs/02-requirements.md: FR-017, FR-018, FR-019, FR-031

Mục tiêu chính:
Khẳng định tính đúng đắn của công thức đo khoảng cách điểm ảnh, quy ước dấu Y từ trên xuống,
công thức quy đổi tỷ lệ mống mắt HVID, và tính toán biên dung sai +-0.5mm.
"""

import pytest
import numpy as np
from app.core.config import settings
from app.schemas.measurement import MeasurementStatusEnum, ReflexVisibilityEnum
from app.services.eyelid_margin_service import MarginResult
from app.services.landmark_service import LandmarkResult
from app.services.reflex_service import ReflexDetectionStatus, ReflexResult
from app.services.measurement_service import (
    compute_measurements,
    _calculate_single_eye_confidence,
    _determine_measurement_status,
)


def _build_dummy_landmarks(hvid_px: float = 40.0) -> LandmarkResult:
    """Tạo đối tượng LandmarkResult giả lập chuẩn cho test toán học."""
    dummy_px = np.zeros((478, 2), dtype=np.float64)
    scale = settings.ASSUMED_CORNEAL_DIAMETER_MM / hvid_px

    return LandmarkResult(
        landmarks_px=dummy_px,
        iris_center_od=(200.0, 150.0),
        iris_center_os=(350.0, 150.0),
        limbus_horizontal_od=((180.0, 150.0), (220.0, 150.0)),
        limbus_horizontal_os=((330.0, 150.0), (370.0, 150.0)),
        hvid_od_px=hvid_px,
        hvid_os_px=hvid_px,
        scale_factor_od=scale,
        scale_factor_os=scale,
        hvid_asymmetry_ratio=0.0,
        is_hvid_biologically_consistent=True,
        upper_eyelid_od=[(180.0, 140.0), (200.0, 138.0), (220.0, 140.0)],
        lower_eyelid_od=[(180.0, 160.0), (200.0, 162.0), (220.0, 160.0)],
        upper_eyelid_os=[(330.0, 140.0), (350.0, 142.0), (370.0, 140.0)],
        lower_eyelid_os=[(330.0, 160.0), (350.0, 162.0), (370.0, 160.0)],
        eyebrow_od=[(180.0, 110.0), (200.0, 108.0), (220.0, 110.0)],
        eyebrow_os=[(330.0, 110.0), (350.0, 108.0), (370.0, 110.0)],
        warning_flags=[],
    )


class TestMeasurementAndGeometryContract:
    """Kiểm thử tính chính xác của các thuật toán hình học và công thức đo lường."""

    def test_mrd1_pixel_and_millimeter_calculation(self):
        """
        [IT-007 / BR-004 / BR-005]
        Kiểm tra công thức:
        MRD1_px = y_ref - y_margin
        MRD1_mm = MRD1_px * (11.71 / HVID_px)
        y_ref = 150.0, y_margin = 138.0 -> MRD1_px = 12.0 px (> 0).
        Với HVID = 40.0 px -> scale = 11.71 / 40.0 = 0.29275 mm/px.
        MRD1_mm = 12.0 * 0.29275 = 3.513 mm -> làm tròn 3.51 mm.
        """
        landmarks = _build_dummy_landmarks(hvid_px=40.0)

        reflex_od = ReflexResult(
            reflex_point=(200.0, 150.0),
            reference_point_used="CORNEAL_LIGHT_REFLEX",
            confidence_score=1.0,
            has_fallback=False,
            status=ReflexDetectionStatus.SUCCESS,
        )
        margin_od = MarginResult(
            margin_point=(200.0, 138.0),
            y_init_spline=138.0,
            y_margin_refined=138.0,
            spline_curve_points=[(200.0, 138.0)],
            gradient_magnitude=15.0,
            is_gradient_refined=True,
        )

        reflex_os = ReflexResult(
            reflex_point=(350.0, 150.0),
            reference_point_used="CORNEAL_LIGHT_REFLEX",
            confidence_score=1.0,
            has_fallback=False,
            status=ReflexDetectionStatus.SUCCESS,
        )
        margin_os = MarginResult(
            margin_point=(350.0, 142.0),
            y_init_spline=142.0,
            y_margin_refined=142.0,
            spline_curve_points=[(350.0, 142.0)],
            gradient_magnitude=15.0,
            is_gradient_refined=True,
        )

        meas_container, bilateral = compute_measurements(
            landmarks=landmarks,
            reflex_od=reflex_od,
            reflex_os=reflex_os,
            margin_od=margin_od,
            margin_os=margin_os,
            sharpness_score=120.0,
        )

        od = meas_container.right_eye_od
        assert od.mrd1_px == 12.0
        assert od.mrd1_mm == pytest.approx(3.51, abs=0.02)
        assert od.mrd1_estimated_range is not None
        assert " - " in od.mrd1_estimated_range
        assert "mm" in od.mrd1_estimated_range
        assert od.status == MeasurementStatusEnum.VALID

        # Kiểm tra Mắt trái (OS): y_margin = 142 -> MRD1_px = 150 - 142 = 8.0 px
        os_meas = meas_container.left_eye_os
        assert os_meas.mrd1_px == 8.0
        assert os_meas.mrd1_mm == pytest.approx(2.34, abs=0.02)

        # Kiểm tra so sánh hai bên
        assert bilateral.delta_mrd1_mm == pytest.approx(abs(3.51 - 2.34), abs=0.02)
        assert bilateral.signed_delta_mrd1_mm == pytest.approx(2.34 - 3.51, abs=0.02)

    def test_negative_mrd1_droop_guard(self):
        """
        [IT-008 / MS § 4.4]
        Khi bờ mi sa trễ xuống thấp hơn tâm phản quang (y_margin > y_ref):
        y_ref = 150.0, y_margin = 155.0 -> raw MRD1 = -5.0 px.
        Hệ thống phải bắt cờ sụp che lấp (ANATOMICAL_NEGATIVE_SUSPECTED)
        và chuyển status sang MEASUREMENT_UNAVAILABLE, reflex_visibility sang OCCLUDED_BY_LID.
        """
        landmarks = _build_dummy_landmarks(hvid_px=40.0)

        reflex_od = ReflexResult(
            reflex_point=(200.0, 150.0),
            reference_point_used="CORNEAL_LIGHT_REFLEX",
            confidence_score=1.0,
            has_fallback=False,
            status=ReflexDetectionStatus.SUCCESS,
        )
        # Bờ mi thấp hơn tâm phản quang 5 pixel (sụp nặng che đồng tử)
        margin_od = MarginResult(
            margin_point=(200.0, 155.0),
            y_init_spline=155.0,
            y_margin_refined=155.0,
            spline_curve_points=[(200.0, 155.0)],
            gradient_magnitude=10.0,
            is_gradient_refined=True,
        )

        reflex_os = ReflexResult(
            reflex_point=(350.0, 150.0),
            reference_point_used="CORNEAL_LIGHT_REFLEX",
            confidence_score=1.0,
            has_fallback=False,
            status=ReflexDetectionStatus.SUCCESS,
        )
        margin_os = MarginResult(
            margin_point=(350.0, 138.0),
            y_init_spline=138.0,
            y_margin_refined=138.0,
            spline_curve_points=[(350.0, 138.0)],
            gradient_magnitude=15.0,
            is_gradient_refined=True,
        )

        meas_container, bilateral = compute_measurements(
            landmarks=landmarks,
            reflex_od=reflex_od,
            reflex_os=reflex_os,
            margin_od=margin_od,
            margin_os=margin_os,
        )

        od = meas_container.right_eye_od
        assert od.mrd1_mm is None, "Khi MRD1 âm, mrd1_mm phải chuyển sang None để bảo vệ hiển thị"
        assert od.status == MeasurementStatusEnum.MEASUREMENT_UNAVAILABLE
        assert od.reflex_visibility == ReflexVisibilityEnum.OCCLUDED_BY_LID
        assert "ANATOMICAL_NEGATIVE_SUSPECTED" in od.warning_flags

    def test_confidence_score_model_invariants(self):
        """
        [MS § 13]
        Kiểm tra tính bất biến của mô hình chấm điểm tin cậy:
        C_MRD1 phải luôn nằm trong khoảng [0.0, 1.0].
        Ánh xạ trạng thái:
        - C >= 0.70 -> VALID
        - 0.50 <= C < 0.70 -> LOW_CONFIDENCE
        - C < 0.50 -> INVALID
        """
        assert _determine_measurement_status(0.95) == MeasurementStatusEnum.VALID
        assert _determine_measurement_status(0.70) == MeasurementStatusEnum.VALID
        assert _determine_measurement_status(0.69) == MeasurementStatusEnum.LOW_CONFIDENCE
        assert _determine_measurement_status(0.50) == MeasurementStatusEnum.LOW_CONFIDENCE
        assert _determine_measurement_status(0.49) == MeasurementStatusEnum.INVALID
        assert _determine_measurement_status(0.0) == MeasurementStatusEnum.INVALID
