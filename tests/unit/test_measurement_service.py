"""
Kiểm thử đơn vị và tích hợp cho Module M6: Measurement Service (MRD1, Bất đối xứng, Độ tin cậy).
Kiểm tra công thức hình học MRD1, quy đổi mm qua S_iris, phân loại bất đối xứng,
chiều cao khe mi PFH, điểm tin cậy tổng hợp và loại bỏ hoàn toàn số hardcode.
Tuyệt đối tuân thủ G2: Xử lý trong RAM.
Tuyệt đối tuân thủ G4: Không dùng từ ngữ thuộc danh mục cấm.
"""

from pathlib import Path
import cv2
from fastapi.testclient import TestClient
import numpy as np
import pytest

from app.core.config import settings
from app.main import app
from app.schemas.measurement import MeasurementsContainer, BilateralComparison
from app.services.eyelid_margin_service import MarginResult
from app.services.landmark_service import LandmarkResult
from app.services.measurement_service import compute_measurements
from app.services.reflex_service import ReflexResult

client = TestClient(app)

from tests.test_helpers import get_sample_image_path

SAMPLE_IMAGE_PATH = get_sample_image_path()


def create_dummy_landmark_result(
    scale_od: float = 0.25,
    scale_os: float = 0.25,
    is_consistent: bool = True,
) -> LandmarkResult:
    """Tạo đối tượng LandmarkResult giả lập phục vụ kiểm thử đơn vị."""
    return LandmarkResult(
        landmarks_px=np.zeros((478, 2)),
        iris_center_od=(50.0, 60.0),
        iris_center_os=(150.0, 60.0),
        limbus_horizontal_od=((26.5, 60.0), (73.5, 60.0)),
        limbus_horizontal_os=((126.5, 60.0), (173.5, 60.0)),
        hvid_od_px=46.84,
        hvid_os_px=46.84,
        scale_factor_od=scale_od,
        scale_factor_os=scale_os,
        hvid_asymmetry_ratio=0.0,
        is_hvid_biologically_consistent=is_consistent,
        upper_eyelid_od=[(30.0, 48.0), (50.0, 44.0), (70.0, 48.0)],
        lower_eyelid_od=[(30.0, 72.0), (50.0, 76.0), (70.0, 72.0)],
        upper_eyelid_os=[(130.0, 48.0), (150.0, 44.0), (170.0, 48.0)],
        lower_eyelid_os=[(130.0, 72.0), (150.0, 76.0), (170.0, 72.0)],
        eyebrow_od=[(30.0, 25.0), (40.0, 22.0), (50.0, 20.0), (60.0, 22.0), (70.0, 25.0)],
        eyebrow_os=[(130.0, 25.0), (140.0, 22.0), (150.0, 20.0), (160.0, 22.0), (170.0, 25.0)],
        warning_flags=[],
    )


# ==============================================================================
# 1. Kiểm thử công thức hình học MRD1 và quy ước dấu
# ==============================================================================

def test_compute_measurements_mrd1_formula():
    """
    Kiểm tra công thức MRD1_px = y_ref - y_margin và quy đổi mm:
    y_ref = 60.0, y_margin = 44.0 -> MRD1_px = 16.0 px.
    scale = 0.25 mm/px -> MRD1_mm = 4.0 mm.
    Khoảng ước lượng: '3.5 - 4.5 mm'.
    """
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
    margin_od = MarginResult(
        margin_point=(50.0, 44.0),
        y_init_spline=44.0,
        y_margin_refined=44.0,
        spline_curve_points=[],
        gradient_magnitude=35.0,
        is_gradient_refined=True,
    )
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
        sharpness_score=100.0,
    )

    # OD: y_ref=60, y_margin=44 -> mrd1_px = 16.0 px, mrd1_mm = 16 * 0.25 = 4.0 mm
    assert meas.right_eye_od.mrd1_px == 16.0
    assert meas.right_eye_od.mrd1_mm == 4.0
    assert meas.right_eye_od.mrd1_estimated_range == "3.5 - 4.5 mm"

    # OS: y_ref=60, y_margin=48 -> mrd1_px = 12.0 px, mrd1_mm = 12 * 0.25 = 3.0 mm
    assert meas.left_eye_os.mrd1_px == 12.0
    assert meas.left_eye_os.mrd1_mm == 3.0
    assert meas.left_eye_os.mrd1_estimated_range == "2.5 - 3.5 mm"

    # Delta: |3.0 - 4.0| = 1.0 mm, signed = 3.0 - 4.0 = -1.0 mm
    assert comp.delta_mrd1_mm == 1.0
    assert comp.signed_delta_mrd1_mm == -1.0
    assert comp.asymmetry_classification == "PHYSIOLOGICAL_SYMMETRY"


def test_mrd1_negative_sign_convention():
    """
    Kiểm tra quy ước dấu âm khi bờ mi trên rủ xuống thấp hơn phản quang:
    y_margin = 64.0 > y_ref = 60.0 -> MRD1_px = -4.0 px -> MRD1_mm = -1.0 mm.
    """
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
    margin_od = MarginResult(
        margin_point=(50.0, 64.0),
        y_init_spline=64.0,
        y_margin_refined=64.0,
        spline_curve_points=[],
        gradient_magnitude=25.0,
        is_gradient_refined=True,
    )
    margin_os = MarginResult(
        margin_point=(150.0, 48.0),
        y_init_spline=48.0,
        y_margin_refined=48.0,
        spline_curve_points=[],
        gradient_magnitude=30.0,
        is_gradient_refined=True,
    )

    meas, _ = compute_measurements(
        landmarks=landmarks,
        reflex_od=reflex_od,
        reflex_os=reflex_os,
        margin_od=margin_od,
        margin_os=margin_os,
    )

    assert meas.right_eye_od.mrd1_px == -4.0
    assert meas.right_eye_od.mrd1_mm == -1.0
    assert meas.right_eye_od.mrd1_estimated_range == "-1.5 - -0.5 mm"


# ==============================================================================
# 2. Kiểm thử phân loại đối xứng (Asymmetry Classification)
# ==============================================================================

@pytest.mark.parametrize(
    "od_margin_y, os_margin_y, expected_class, expected_delta",
    [
        (44.0, 44.0, "PHYSIOLOGICAL_SYMMETRY", 0.0),      # Cân bằng hoàn hảo
        (44.0, 48.0, "PHYSIOLOGICAL_SYMMETRY", 1.0),      # Chênh 1.0 mm (sinh lý)
        (44.0, 49.0, "PHYSIOLOGICAL_SYMMETRY", 1.25),     # Chênh 1.25 mm (sinh lý)
        (44.0, 50.0, "PHYSIOLOGICAL_SYMMETRY", 1.5),      # Chênh 1.50 mm (ngưỡng biên sinh lý)
        (44.0, 51.0, "BORDERLINE_ASYMMETRY", 1.75),       # Chênh 1.75 mm (ranh giới 1.5 - 2.0mm)
        (44.0, 52.0, "BORDERLINE_ASYMMETRY", 2.0),        # Chênh 2.0 mm (ngưỡng biên ranh giới)
        (44.0, 53.0, "SIGNIFICANT_ASYMMETRY", 2.25),      # Chênh 2.25 mm (> 2.0 mm: rõ rệt)
    ],
)
def test_asymmetry_classification_thresholds(
    od_margin_y, os_margin_y, expected_class, expected_delta
):
    landmarks = create_dummy_landmark_result(scale_od=0.25, scale_os=0.25)
    reflex = ReflexResult((50.0, 60.0), "CORNEAL_LIGHT_REFLEX", 1.0, False)
    reflex_os = ReflexResult((150.0, 60.0), "CORNEAL_LIGHT_REFLEX", 1.0, False)

    margin_od = MarginResult((50.0, od_margin_y), od_margin_y, od_margin_y, [], 20.0, True)
    margin_os = MarginResult((150.0, os_margin_y), os_margin_y, os_margin_y, [], 20.0, True)

    _, comp = compute_measurements(landmarks, reflex, reflex_os, margin_od, margin_os)

    assert comp.asymmetry_classification == expected_class
    assert abs(comp.delta_mrd1_mm - expected_delta) < 1e-4


# ==============================================================================
# 3. Kiểm thử điểm tin cậy tổng hợp và trạng thái
# ==============================================================================

def test_composite_confidence_score_and_status():
    """Kiểm tra tính điểm tin cậy C_MRD1 khi ảnh tối ưu vs khi fallback."""
    landmarks = create_dummy_landmark_result(is_consistent=True)

    # 1. Điều kiện tối ưu: ảnh nét (Q=1.0), bờ mi tinh chỉnh (C=1.0), có flash (C=1.0), HVID nhất quán (G=1.0)
    # C = 0.25*1.0 + 0.35*1.0 + 0.25*1.0 + 0.15*1.0 = 1.0
    reflex_opt = ReflexResult((50.0, 60.0), "CORNEAL_LIGHT_REFLEX", 1.0, False)
    margin_opt = MarginResult((50.0, 45.0), 45.0, 45.0, [], 30.0, True)

    meas_opt, _ = compute_measurements(
        landmarks, reflex_opt, reflex_opt, margin_opt, margin_opt, sharpness_score=100.0
    )
    assert meas_opt.right_eye_od.confidence_score == 1.0
    assert meas_opt.right_eye_od.status == "VALID"

    # 2. Điều kiện suy giảm: fallback sang đồng tử (C_phan_quang=0.65), không bắt được gradient (C_bo_mi=0.70)
    # C = 0.25*1.0 + 0.35*0.70 + 0.25*0.65 + 0.15*1.0 = 0.25 + 0.245 + 0.1625 + 0.15 = 0.81 (vẫn >= 0.70 VALID)
    reflex_degraded = ReflexResult((50.0, 60.0), "PUPIL_CENTER", 0.65, True)
    margin_degraded = MarginResult((50.0, 45.0), 45.0, 45.0, [], 0.0, False)

    meas_deg, _ = compute_measurements(
        landmarks, reflex_degraded, reflex_degraded, margin_degraded, margin_degraded, sharpness_score=50.0
    )
    # Với ảnh mờ (sharpness=50 -> Q=0.50):
    # C = 0.25*0.50 + 0.35*0.70 + 0.25*0.65 + 0.15*1.0 = 0.125 + 0.245 + 0.1625 + 0.15 = 0.68
    # 0.50 <= C < 0.70 -> LOW_CONFIDENCE
    assert 0.50 <= meas_deg.right_eye_od.confidence_score < 0.70
    assert meas_deg.right_eye_od.status == "LOW_CONFIDENCE"
    assert "LOW_MEASUREMENT_CONFIDENCE" in meas_deg.right_eye_od.warning_flags


# ==============================================================================
# 4. Kiểm thử tích hợp toàn diện trên ảnh thật và API
# ==============================================================================

def test_compute_measurements_real_image_dynamic_values():
    """Kiểm tra chạy trên ảnh benchmark thực tế và loại bỏ số hardcode."""
    if not SAMPLE_IMAGE_PATH.exists():
        pytest.skip("Không tìm thấy tệp ảnh mẫu thực tế.")

    with open(SAMPLE_IMAGE_PATH, "rb") as f:
        file_bytes = f.read()

    response = client.post(
        "/api/v1/screening/analyze",
        files={"file": ("test_benchmark.jpg", file_bytes, "image/jpeg")},
    )

    assert response.status_code == 200
    data = response.json()

    od = data["measurements"]["right_eye_od"]
    os_eye = data["measurements"]["left_eye_os"]
    comp = data["bilateral_comparison"]

    # Đảm bảo các con số là số thực động được tính toán, KHÔNG PHẢI MOCK CŨ (3.42, 1.94, 1.48)
    # Tọa độ thực tế của ảnh benchmark:
    mrd1_od = od["mrd1_mm"]
    mrd1_os = os_eye["mrd1_mm"]
    delta = comp["delta_mrd1_mm"]

    # Kiểm tra tính nhất quán toán học: delta = |os - od|
    assert abs(delta - abs(mrd1_os - mrd1_od)) < 0.02
    assert od["mrd1_px"] > 0
    assert os_eye["mrd1_px"] > 0
    assert od["status"] in ["VALID", "LOW_CONFIDENCE"]
    assert os_eye["status"] in ["VALID", "LOW_CONFIDENCE"]

    # Kiểm tra chiều cao khe mi PFH
    assert od["pfh_mm"] is not None and od["pfh_mm"] > 0
    assert os_eye["pfh_mm"] is not None and os_eye["pfh_mm"] > 0

    # Kiểm tra khoảng dung sai hiển thị
    tol = settings.NOMINAL_TOLERANCE_MM
    expected_range_od = f"{mrd1_od - tol:.1f} - {mrd1_od + tol:.1f} mm"
    assert od["mrd1_estimated_range"] == expected_range_od
