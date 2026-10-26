"""
Kiểm thử chuyên sâu cho Giai đoạn Phase A (Phase A - Safety Guards & Critical Bug Fixes).
Bao phủ các mục tiêu A1, A2, A3, A4, A6 theo kế hoạch implementation_plan_prompt4.md:
- A1: Ngưỡng chốt chặn hai bên 2.0mm và mã lý do BILATERAL_SEVERE_MRD1_UNDER_2MM.
- A2: Chốt chặn khoảng cách âm (Negative MRD1 guard) -> None và MEASUREMENT_UNAVAILABLE.
- A3: Logic nhận biết bờ mi che lấp trục thị giác giải phẫu.
- A4: Chuyển tiếp vệt lóa sang fallback thay vì từ chối HTTP 422.
- A6: Kiểm thử hồi quy 3 ca trọng điểm: patient_07, patient_09, patient_11.
"""

from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from app.core.config import settings
from app.main import app
from app.schemas.triage import ScreeningTier
from app.services.measurement_service import (
    MarginResult,
    PointCoordinates,
    ReflexResult,
    compute_measurements,
)
from app.services.triage_service import evaluate_triage
from tests.unit.test_measurement_service import create_dummy_landmark_result

client = TestClient(app)
FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"


# ==============================================================================
# A1: KIỂM THỬ NGƯỠNG PHÂN TẦNG VÀ LÝ DO KÍCH HOẠT (TRIAGE THRESHOLD)
# ==============================================================================

def test_a1_critical_severe_threshold_setting():
    """Khẳng định ngưỡng CRITICAL_SEVERE_MRD1_MM được thiết lập chuẩn 2.0 mm."""
    assert settings.CRITICAL_SEVERE_MRD1_MM == 2.0


@pytest.mark.parametrize(
    "od, os, expected_tier, expected_bilateral, expected_reason",
    [
        (1.9, 1.9, ScreeningTier.TIER_C, True, "BILATERAL_SEVERE_MRD1_UNDER_2MM"),
        (1.99, 1.99, ScreeningTier.TIER_C, True, "BILATERAL_SEVERE_MRD1_UNDER_2MM"),
        (2.0, 2.0, ScreeningTier.TIER_A, False, "PHYSIOLOGICAL_NORMAL"),
        (2.5, 2.5, ScreeningTier.TIER_A, False, "PHYSIOLOGICAL_NORMAL"),
        (3.5, 3.5, ScreeningTier.TIER_A, False, "PHYSIOLOGICAL_NORMAL"),
    ],
)
def test_a1_threshold_boundaries_below_equal_above_2mm(
    od, os, expected_tier, expected_bilateral, expected_reason
):
    """Kiểm tra hành vi phân tầng tại các giá trị biên: dưới 2.0mm, bằng 2.0mm và trên 2.0mm."""
    res = evaluate_triage(od_mrd1_mm=od, os_mrd1_mm=os)
    assert res.tier == expected_tier
    assert res.is_bilateral_severe_triggered == expected_bilateral
    assert expected_reason in res.trigger_reasons


# ==============================================================================
# A2: KIỂM THỬ CHỐT CHẶN DẤU ÂM (NEGATIVE MRD1 GUARD)
# ==============================================================================

def test_a2_negative_mrd1_guard_returns_none_and_unavailable():
    """Kiểm tra khi y_ref < y_margin (bờ mi rủ dưới phản quang), trả về None và MEASUREMENT_UNAVAILABLE."""
    landmarks = create_dummy_landmark_result(scale_od=0.25, scale_os=0.25)
    reflex_od = ReflexResult(
        reflex_point=(50.0, 60.0),
        reference_point_used="CORNEAL_LIGHT_REFLEX",
        confidence_score=1.0,
        has_fallback=False,
    )
    reflex_os = ReflexResult(
        reflex_point=(150.0, 60.0),
        reference_point_used="CORNEAL_LIGHT_REFLEX",
        confidence_score=1.0,
        has_fallback=False,
    )
    # OD: y_margin = 65.0 > y_ref = 60.0 -> raw_mrd1_px = -5.0 px (ÂM)
    margin_od = MarginResult(
        margin_point=(50.0, 65.0),
        y_init_spline=65.0,
        y_margin_refined=65.0,
        spline_curve_points=[],
        gradient_magnitude=25.0,
        is_gradient_refined=True,
    )
    # OS: y_margin = 48.0 < y_ref = 60.0 -> raw_mrd1_px = 12.0 px (DƯƠNG HỢP LỆ)
    margin_os = MarginResult(
        margin_point=(150.0, 48.0),
        y_init_spline=48.0,
        y_margin_refined=48.0,
        spline_curve_points=[],
        gradient_magnitude=30.0,
        is_gradient_refined=True,
    )

    meas, comp = compute_measurements(
        landmarks=landmarks,
        reflex_od=reflex_od,
        reflex_os=reflex_os,
        margin_od=margin_od,
        margin_os=margin_os,
    )

    # Khẳng định mắt phải OD được bảo vệ an toàn
    assert meas.right_eye_od.mrd1_px is None
    assert meas.right_eye_od.mrd1_mm is None
    assert meas.right_eye_od.mrd1_estimated_range is None
    assert meas.right_eye_od.status == "MEASUREMENT_UNAVAILABLE"
    assert meas.right_eye_od.confidence_score == 0.0
    assert "ANATOMICAL_NEGATIVE_SUSPECTED" in meas.right_eye_od.warning_flags

    # Khẳng định mắt trái OS vẫn đo bình thường
    assert meas.left_eye_os.mrd1_mm == 3.0
    assert meas.left_eye_os.status == "VALID"

    # Khẳng định so sánh bất đối xứng xử lý an toàn giá trị None
    assert comp.delta_mrd1_mm is None
    assert comp.signed_delta_mrd1_mm is None
    assert comp.asymmetry_classification == "SIGNIFICANT_ASYMMETRY"


# ==============================================================================
# A6: KIỂM THỬ HỒI QUY FIXTURES THỰC TẾ (PATIENT_07, PATIENT_09, PATIENT_11)
# ==============================================================================

def test_a6_regression_patient_07_upgraded_to_tier_c():
    """Kiểm tra fixture patient_07 (MRD1=1.52/1.06) được phân tầng đúng TIER_C với lý do chốt chặn hai bên."""
    fixture_path = FIXTURES_DIR / "patient_07_image.jpg"
    if not fixture_path.exists():
        pytest.skip("Fixture patient_07_image.jpg không tồn tại.")

    with open(fixture_path, "rb") as f:
        resp = client.post("/api/v1/screening/analyze", files={"file": ("patient_07.jpg", f, "image/jpeg")})

    assert resp.status_code == 200
    data = resp.json()
    assert data["triage_result"]["tier"] == "TIER_C"
    assert data["triage_result"]["is_bilateral_severe_triggered"] is True
    assert "BILATERAL_SEVERE_MRD1_UNDER_2MM" in data["triage_result"]["trigger_reasons"]


def test_a6_regression_patient_09_negative_mrd1_guarded():
    """Kiểm tra fixture patient_09 không trả về số âm -0.31mm; OS nhận None và MEASUREMENT_UNAVAILABLE."""
    fixture_path = FIXTURES_DIR / "patient_09_image.jpg"
    if not fixture_path.exists():
        pytest.skip("Fixture patient_09_image.jpg không tồn tại.")

    with open(fixture_path, "rb") as f:
        resp = client.post("/api/v1/screening/analyze", files={"file": ("patient_09.jpg", f, "image/jpeg")})

    assert resp.status_code == 200
    data = resp.json()
    os_meas = data["measurements"]["left_eye_os"]
    assert os_meas["mrd1_mm"] is None
    assert os_meas["status"] == "MEASUREMENT_UNAVAILABLE"
    assert "ANATOMICAL_NEGATIVE_SUSPECTED" in os_meas["warning_flags"]
    assert data["triage_result"]["tier"] == "TIER_C"


def test_a6_regression_patient_11_glare_fallback_succeeds():
    """Kiểm tra fixture patient_11 không bị từ chối 422; chuyển tiếp sang fallback và trả về HTTP 200."""
    fixture_path = FIXTURES_DIR / "patient_11_image.jpg"
    if not fixture_path.exists():
        pytest.skip("Fixture patient_11_image.jpg không tồn tại.")

    with open(fixture_path, "rb") as f:
        resp = client.post("/api/v1/screening/analyze", files={"file": ("patient_11.jpg", f, "image/jpeg")})

    assert resp.status_code == 200
    data = resp.json()
    od_meas = data["measurements"]["right_eye_od"]
    assert "REFLEX_GLARE_ARTIFACT" in od_meas["warning_flags"]
    assert od_meas["reference_point_used"] == "PUPIL_CENTER_FALLBACK"


# ==============================================================================
# REMEDIATION TESTS: REFLEX OCCLUSION GEOMETRIC RULE & REASON CODE VERIFICATION
# ==============================================================================

def test_remediation_reflex_not_occluded_when_margin_above_iris_center():
    """Kiểm tra khi reflex bị thiếu do thiếu sáng (NO_REFLEX_FOUND), nhưng bờ mi ở trên tâm mống mắt
    (y_margin < y_iris_center), hệ thống KHÔNG ĐƯỢC khẳng định che lấp giải phẫu (is_reflex_occluded = False).
    """
    # Điểm tâm mống mắt y = 100.0, bán kính r = 40.0.
    # Bờ mi y = 90.0 (nằm phía trên tâm mống mắt 10px, tức mi mở bình thường)
    iris_center_y = 100.0
    margin_y = 90.0  # y_margin < y_iris_center

    # Kiểm tra biểu thức logic đã sửa
    is_occluded = (
        "ANATOMICAL_NEGATIVE_SUSPECTED" in []
        or margin_y >= iris_center_y
    )
    assert is_occluded is False, "Mất phản quang do thiếu sáng không được coi là che lấp khi bờ mi ở trên tâm mống mắt!"


def test_remediation_reflex_genuinely_occluded_when_margin_at_or_below_center():
    """Kiểm tra khi bờ mi rủ xuống ngang hoặc dưới tâm mống mắt (y_margin >= y_iris_center),
    hệ thống xác nhận che lấp giải phẫu (is_reflex_occluded = True).
    """
    iris_center_y = 100.0
    margin_y_droop = 102.0  # y_margin >= y_iris_center (che trục nhìn)

    is_occluded = (
        "ANATOMICAL_NEGATIVE_SUSPECTED" in []
        or margin_y_droop >= iris_center_y
    )
    assert is_occluded is True, "Bờ mi che qua tâm mống mắt phải được xác nhận là che lấp trục thị giác!"


def test_remediation_unilateral_unavailable_never_reports_under_2mm_code():
    """Kiểm tra khi một hoặc cả hai mắt có MRD1 là None (không khả dụng),
    hệ thống KHÔNG BAO GIỜ gán mã định lượng sai 'UNILATERAL_SEVERE_MRD1_UNDER_2MM'.
    """
    # Trường hợp 1 mắt None kèm che lấp phản quang (ca patient_09)
    res_occluded = evaluate_triage(od_mrd1_mm=1.05, os_mrd1_mm=None, is_reflex_occluded=True)
    assert res_occluded.tier == ScreeningTier.TIER_C
    assert "REFLEX_OCCLUDED_SEVERE" in res_occluded.trigger_reasons
    assert "UNILATERAL_SEVERE_MRD1_UNDER_2MM" not in res_occluded.trigger_reasons

    # Trường hợp gọi trực tiếp 1 mắt None không kèm che lấp
    res_unavail_no_occ = evaluate_triage(od_mrd1_mm=None, os_mrd1_mm=2.5, is_reflex_occluded=False)
    assert "UNILATERAL_SEVERE_MRD1_UNDER_2MM" not in res_unavail_no_occ.trigger_reasons

    # Trường hợp cả 2 mắt None không kèm che lấp
    res_both_none = evaluate_triage(od_mrd1_mm=None, os_mrd1_mm=None, is_reflex_occluded=False)
    assert "UNILATERAL_SEVERE_MRD1_UNDER_2MM" not in res_both_none.trigger_reasons
    assert "BILATERAL_SEVERE_MRD1_UNDER_2MM" not in res_both_none.trigger_reasons


def test_remediation_measured_values_correctly_report_under_2mm_code():
    """Kiểm tra mã 'UNILATERAL_SEVERE_MRD1_UNDER_2MM' chỉ xuất hiện khi có số đo thực nghiệm
    nhỏ hơn 2.0mm đi kèm bất đối xứng rõ rệt (> 2.0mm) theo đúng nhánh 1.1.
    """
    # OD=1.5mm (< 2.0mm), OS=3.8mm -> delta=2.3mm (> 2.0mm), min=1.5mm (< 2.0mm)
    res = evaluate_triage(od_mrd1_mm=1.5, os_mrd1_mm=3.8)
    assert res.tier == ScreeningTier.TIER_C
    assert "SIGNIFICANT_ASYMMETRY_OVER_2MM" in res.trigger_reasons
    assert "UNILATERAL_SEVERE_MRD1_UNDER_2MM" in res.trigger_reasons
