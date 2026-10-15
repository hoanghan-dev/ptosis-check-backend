"""
Kiểm thử đơn vị cấu hình tham số V2 và tính hợp lệ của lược đồ dữ liệu (Phase 1).
Xác thực tương thích ngược 100% (Guardrail G5) và an toàn thuật ngữ phi can thiệp chuyên sâu (G4).
"""

import pytest
from app.core.config import settings
from app.schemas.measurement import (
    EyeAnnotationPoints,
    EyeFloatPair,
    ImageQualityAudit,
    MeasurementsContainer,
    MeasurementPayload,
    BilateralComparison,
    ScaleReference,
    SingleEyeMeasurement,
    PointCoordinates,
    AnnotatedOverlayContainer,
)
from app.schemas.quality import PoseAudit


def test_v2_configuration_constants():
    """Kiểm tra các hằng số ngưỡng thuật toán V2 được nạp chính xác từ settings."""
    # Phản quang giác mạc nâng cao (M4)
    assert settings.REFLEX_MAX_ASPECT_RATIO == 2.8
    assert settings.REFLEX_MAX_AREA_PX == 60
    assert settings.REFLEX_MAX_AREA_IRIS_RATIO == 0.05
    assert settings.REFLEX_CLUSTER_DISTANCE_PX == 8.0
    assert settings.REFLEX_CLUSTER_DISTANCE_IRIS_RATIO == 0.20
    assert settings.REFLEX_CLUSTER_SCORE_GAP == 3.0

    # Định vị đồng tử độc lập (M4.5)
    assert settings.PUPIL_SEARCH_RADIUS_FACTOR == 0.85
    assert settings.PUPIL_MIN_CONTRAST_DIFF == 12.0
    assert settings.PUPIL_MIN_RADIUS_RATIO == 0.15
    assert settings.PUPIL_MAX_RADIUS_RATIO == 0.75
    assert settings.PUPIL_MAX_OFFSET_FROM_IRIS == 0.25
    assert settings.PUPIL_MIN_CIRCULARITY == 0.35
    assert settings.PUPIL_FALLBACK_CONFIDENCE == 0.75
    assert settings.IRIS_CENTER_FALLBACK_CONFIDENCE == 0.60

    # Tính toàn vẹn hình thái học mống mắt hai bên
    assert settings.IRIS_MAX_BILATERAL_DIFF_RATIO == 0.15

    # An toàn biên khung hình (M2)
    assert settings.OCULAR_EDGE_MARGIN_PX == 2


def test_single_eye_measurement_backward_compatibility():
    """Xác thực SingleEyeMeasurement hoạt động khi không truyền các trường V2 mới (tương thích ngược G5)."""
    item = SingleEyeMeasurement(
        mrd1_px=24.5,
        mrd1_mm=3.2,
        mrd1_estimated_range="[2.7 - 3.7 mm]",
        reference_point_used="CORNEAL_LIGHT_REFLEX",
        status="VALID",
        confidence_score=1.0,
        warning_flags=[],
    )

    # Các trường mở rộng mới phải mặc định là None
    assert item.glint_aspect_ratio is None
    assert item.pupil_diameter_px is None
    assert item.fallback_type is None

    dumped = item.model_dump()
    assert dumped["glint_aspect_ratio"] is None
    assert dumped["pupil_diameter_px"] is None
    assert dumped["fallback_type"] is None

    # Khôi phục từ dict
    restored = SingleEyeMeasurement.model_validate(dumped)
    assert restored.mrd1_mm == 3.2
    assert restored.glint_aspect_ratio is None


def test_single_eye_measurement_with_v2_fields():
    """Xác thực SingleEyeMeasurement khi được cung cấp đầy đủ các trường hình thái học V2."""
    item = SingleEyeMeasurement(
        mrd1_px=22.0,
        mrd1_mm=2.8,
        mrd1_estimated_range="[2.3 - 3.3 mm]",
        reference_point_used="PUPIL_CENTER_FALLBACK",
        reference_coordinates=PointCoordinates(x=120.5, y=140.2),
        eyelid_margin_point=PointCoordinates(x=120.5, y=118.2),
        status="VALID",
        confidence_score=0.75,
        warning_flags=["REFLEX_FALLBACK_TO_PUPIL"],
        glint_aspect_ratio=1.45,
        pupil_diameter_px=18.6,
        fallback_type="PUPIL_DETECTOR",
    )

    assert item.reference_point_used == "PUPIL_CENTER_FALLBACK"
    assert item.glint_aspect_ratio == 1.45
    assert item.pupil_diameter_px == 18.6
    assert item.fallback_type == "PUPIL_DETECTOR"

    dumped = item.model_dump()
    assert dumped["glint_aspect_ratio"] == 1.45
    assert dumped["fallback_type"] == "PUPIL_DETECTOR"


def test_measurement_payload_end_to_end_schema():
    """Xác thực toàn bộ cây dữ liệu MeasurementPayload vẫn bảo toàn cấu trúc khi chứa các trường V2."""
    payload = MeasurementPayload(
        session_id="session_test_phase1",
        timestamp="2026-10-16T00:00:00Z",
        image_quality_audit=ImageQualityAudit(
            passed=True,
            pose=PoseAudit(yaw=1.0, pitch=-2.0, roll=0.5),
            sharpness_score=120.0,
            mean_illumination=140.0,
        ),
        scale_reference=ScaleReference(
            hvid_pixels=EyeFloatPair(right_eye=48.0, left_eye=48.5),
            scale_factor_mm_per_px=EyeFloatPair(right_eye=0.244, left_eye=0.241),
        ),
        measurements=MeasurementsContainer(
            right_eye_od=SingleEyeMeasurement(
                mrd1_px=25.0,
                mrd1_mm=3.1,
                mrd1_estimated_range="[2.6 - 3.6 mm]",
                reference_point_used="CORNEAL_LIGHT_REFLEX",
                glint_aspect_ratio=1.1,
                pupil_diameter_px=None,
                fallback_type=None,
            ),
            left_eye_os=SingleEyeMeasurement(
                mrd1_px=20.0,
                mrd1_mm=2.4,
                mrd1_estimated_range="[1.9 - 2.9 mm]",
                reference_point_used="PUPIL_CENTER_FALLBACK",
                glint_aspect_ratio=None,
                pupil_diameter_px=19.2,
                fallback_type="PUPIL_DETECTOR",
            ),
        ),
        bilateral_comparison=BilateralComparison(
            delta_mrd1_mm=0.7,
            signed_delta_mrd1_mm=0.7,
            asymmetry_classification="PHYSIOLOGICAL_SYMMETRY",
        ),
        annotated_overlay_coordinates=AnnotatedOverlayContainer(
            right_eye=EyeAnnotationPoints(
                corneal_reflex=[100.0, 100.0],
                pupil_center=[100.0, 100.0],
            ),
            left_eye=EyeAnnotationPoints(
                corneal_reflex=[200.0, 100.0],
                pupil_center=[200.0, 100.0],
            ),
        ),
    )

    data = payload.model_dump()
    assert data["measurements"]["right_eye_od"]["glint_aspect_ratio"] == 1.1
    assert data["measurements"]["left_eye_os"]["fallback_type"] == "PUPIL_DETECTOR"
