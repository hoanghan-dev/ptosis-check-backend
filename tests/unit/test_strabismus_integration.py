"""
Kiểm thử tích hợp chuyên sâu cho các thuật toán lâm sàng chuyển giao từ dự án Lác/Lé (Hirschberg V2):
1. Khử răng cưa bờ mi (Anti-zigzag spike & Savitzky-Golay envelope smoothing).
2. Định vị mống mắt kháng lác (Strabismus-resilient dark cornea component recentering).
3. Bắt đốm phản xạ Purkinje I lệch trục nhãn cầu (Decoupled reflex detection).
4. Chốt chặn an toàn đo lường MRD1 sinh lý học (Physiological limit safeguards).
Tuân thủ tuyệt đối G2 (In-memory zero retention) và G4 (Không dùng từ ngữ thuộc danh mục cấm).
"""

import math
import numpy as np
import pytest

from app.schemas.measurement import MeasurementsContainer, BilateralComparison
from app.services.eyelid_model_service import _smooth_eyelid_curve, EyelidSegmentationResult
from app.services.iris_model_service import detect_iris_geometry, IrisGeometryResult
from app.services.measurement_service import compute_measurements
from app.services.reflex_model_service import _extract_optical_glint, ReflexModelResult, locate_reference_point
from app.services.reflex_service import ReflexResult


def test_eyelid_contour_spike_elimination():
    """Kiểm tra thuật toán khử gai nhọn zigzag trên bờ mi mắt."""
    # Tạo đường cong bờ mi ban đầu có các gai nhọn bất thường (spikes) do lông mi / bóng đổ
    xs = np.linspace(20.0, 180.0, 30)
    # Đường cong parabol tự nhiên: y = 0.002 * (x - 100)^2 + 50
    natural_y = 0.002 * ((xs - 100.0) ** 2) + 50.0
    
    noisy_pts = []
    for i, (x, y) in enumerate(zip(xs, natural_y)):
        pt_y = y
        # Thêm gai nhọn tại điểm thứ 8 và 20
        if i == 8:
            pt_y -= 15.0  # Gai nhọn hướng lên 15 px
        elif i == 20:
            pt_y += 12.0  # Gai nhọn hướng xuống 12 px
        noisy_pts.append((float(x), float(pt_y)))

    # Làm mịn bằng thuật toán Savitzky-Golay / envelope smoothing
    smoothed = _smooth_eyelid_curve(noisy_pts, is_upper=True)
    assert len(smoothed) == 30

    # Kiểm tra đạo hàm bậc 2 / độ dốc cực đại không bị nhảy vọt (không có zigzag spike)
    for i in range(1, len(smoothed) - 1):
        prev_y = smoothed[i - 1][1]
        curr_y = smoothed[i][1]
        next_y = smoothed[i + 1][1]
        # Độ chênh lệch giữa các điểm liền kề trên lưới 30 điểm phải trơn mượt (< 6 px)
        assert abs(curr_y - prev_y) < 6.0
        assert abs(next_y - curr_y) < 6.0

    # Khóe mắt trong và ngoài (canthi endpoints) phải được bảo toàn
    assert abs(smoothed[0][0] - noisy_pts[0][0]) < 1e-3
    assert abs(smoothed[0][1] - noisy_pts[0][1]) < 1e-3
    assert abs(smoothed[-1][0] - noisy_pts[-1][0]) < 1e-3
    assert abs(smoothed[-1][1] - noisy_pts[-1][1]) < 1e-3


def test_strabismus_iris_recentering():
    """Kiểm tra mống mắt bị lác trong (esotropia) lệch sát mép góc trong được tái định vị chính xác."""
    # Tạo ảnh crop mắt nhân tạo kích thước 120 x 180
    h, w = 120, 180
    crop_bgr = np.ones((h, w, 3), dtype=np.uint8) * 180  # Vùng củng mạc / da sáng
    
    # Giả lập giác mạc/mống mắt tối lệch hẳn về phía góc trong (x = 45, y = 60, r = 24)
    # Trong khi tâm crop là x = 90, y = 60 (độ lệch 45 px > 0.15 * 180 = 27 px)
    cv2_center = (45, 60)
    cv2_radius = 24
    import cv2
    cv2.circle(crop_bgr, cv2_center, cv2_radius, (35, 30, 25), -1)

    box = (100, 100, 100 + w, 100 + h)
    full_image = np.ones((300, 400, 3), dtype=np.uint8) * 180
    full_image[100:100+h, 100:100+w] = crop_bgr

    iris_res = detect_iris_geometry(full_image, box, eye_side="OD")
    
    # Tâm mống mắt toàn cục phải nằm gần x = 100 + 45 = 145, y = 100 + 60 = 160
    assert abs(iris_res.iris_center[0] - 145.0) < 15.0
    assert abs(iris_res.iris_center[1] - 160.0) < 15.0
    # Phải kích hoạt cờ cảnh báo mắt lác/lé lệch trục
    assert "STRABISMUS_RECENTERED_IRIS" in iris_res.warning_flags


def test_decoupled_corneal_reflex_extraction():
    """Kiểm tra đốm phản xạ Purkinje I lệch trục mống mắt do lác không bị loại bỏ."""
    import cv2
    crop_h, crop_w = 100, 100
    crop_bgr = np.ones((crop_h, crop_w, 3), dtype=np.uint8) * 80
    
    # Tâm mống mắt tại (50, 50), bán kính R = 30
    # Đặt đốm sáng tại (75, 50), khoảng cách tới tâm mống là 25 px
    # (25 px / 30 px = 0.83 * R_iris, ngoài vùng 0.4*R_iris nhưng nằm trong 1.3*R_iris)
    crop_bgr[50, 75] = (250, 250, 250)
    crop_bgr[50, 76] = (240, 240, 240)
    crop_bgr[51, 75] = (235, 235, 235)

    box = (0, 0, 100, 100)
    iris_center = (50.0, 50.0)
    iris_radius = 30.0
    pupil_point = (50.0, 50.0)

    glint = _extract_optical_glint(
        crop_bgr=crop_bgr,
        box=box,
        iris_center=iris_center,
        iris_radius=iris_radius,
        pupil_point=pupil_point,
        eyelid_mask_crop=None,
    )

    assert glint is not None
    point, aspect = glint
    assert abs(point[0] - 75.3) < 1.0
    assert abs(point[1] - 50.3) < 1.0
    assert aspect <= 2.8


def test_mrd1_physiological_limits_safeguards():
    """Kiểm tra các chốt chặn an toàn khi MRD1 vượt ngưỡng sinh lý lâm sàng."""
    # Tạo dữ liệu đo lường với MRD1 bị phóng đại bất thường (MRD1 = 7.5 mm > 6.0 mm)
    from app.services.landmark_service import LandmarkResult
    
    landmarks = LandmarkResult(
        landmarks_px=np.zeros((478, 2)),
        iris_center_od=(50.0, 60.0),
        iris_center_os=(150.0, 60.0),
        limbus_horizontal_od=((26.5, 60.0), (73.5, 60.0)),
        limbus_horizontal_os=((126.5, 60.0), (173.5, 60.0)),
        hvid_od_px=46.84,
        hvid_os_px=46.84,
        scale_factor_od=0.25,
        scale_factor_os=0.25,
        hvid_asymmetry_ratio=0.0,
        is_hvid_biologically_consistent=True,
        upper_eyelid_od=[(30.0, 30.0), (50.0, 30.0), (70.0, 30.0)],
        lower_eyelid_od=[(30.0, 75.0), (50.0, 75.0), (70.0, 75.0)],
        upper_eyelid_os=[(130.0, 48.0), (150.0, 44.0), (170.0, 48.0)],
        lower_eyelid_os=[(130.0, 72.0), (150.0, 76.0), (170.0, 72.0)],
        eyebrow_od=[],
        eyebrow_os=[],
        warning_flags=[],
    )

    # OD: y_ref = 60.0, y_margin = 30.0 -> mrd1_px = 30 px * 0.25 = 7.5 mm (> 6.0 mm)
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
    from app.services.eyelid_margin_service import MarginResult
    margin_od = MarginResult(
        margin_point=(50.0, 30.0),
        y_init_spline=30.0,
        y_margin_refined=30.0,
        spline_curve_points=[],
        gradient_magnitude=30.0,
        is_gradient_refined=True,
    )
    margin_os = MarginResult(
        margin_point=(150.0, 44.0),
        y_init_spline=44.0,
        y_margin_refined=44.0,
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
        sharpness_score=100.0,
    )

    # OD phải bị gắn cờ và hạ cấp độ tin cậy
    assert meas.right_eye_od.mrd1_mm == 7.5
    assert "PHYSIOLOGICAL_LIMIT_EXCEEDED_MRD1" in meas.right_eye_od.warning_flags
    assert meas.right_eye_od.status == "LOW_CONFIDENCE"
    assert meas.right_eye_od.confidence_score <= 0.49

    # OS bình thường: MRD1 = (60 - 44) * 0.25 = 4.0 mm
    assert meas.left_eye_os.mrd1_mm == 4.0
    assert "PHYSIOLOGICAL_LIMIT_EXCEEDED_MRD1" not in meas.left_eye_os.warning_flags
    assert meas.left_eye_os.status == "VALID"


def test_reflex_below_lower_margin_safeguard():
    """Kiểm tra khi điểm phản xạ trôi xuống dưới bờ mi dưới."""
    from app.services.landmark_service import LandmarkResult
    from app.services.eyelid_margin_service import MarginResult

    landmarks = LandmarkResult(
        landmarks_px=np.zeros((478, 2)),
        iris_center_od=(50.0, 80.0),
        iris_center_os=(150.0, 60.0),
        limbus_horizontal_od=((26.5, 80.0), (73.5, 80.0)),
        limbus_horizontal_os=((126.5, 60.0), (173.5, 60.0)),
        hvid_od_px=46.84,
        hvid_os_px=46.84,
        scale_factor_od=0.25,
        scale_factor_os=0.25,
        hvid_asymmetry_ratio=0.0,
        is_hvid_biologically_consistent=True,
        upper_eyelid_od=[(30.0, 40.0), (50.0, 40.0), (70.0, 40.0)],
        lower_eyelid_od=[(30.0, 70.0), (50.0, 70.0), (70.0, 70.0)],
        upper_eyelid_os=[(130.0, 48.0), (150.0, 44.0), (170.0, 48.0)],
        lower_eyelid_os=[(130.0, 72.0), (150.0, 76.0), (170.0, 72.0)],
        eyebrow_od=[],
        eyebrow_os=[],
        warning_flags=[],
    )

    # OD: y_ref = 80.0 > y_lower = 70.0 (phản xạ nằm dưới cả bờ mi dưới)
    reflex_od = ReflexResult(
        reflex_point=(50.0, 80.0),
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
        margin_point=(50.0, 40.0),
        y_init_spline=40.0,
        y_margin_refined=40.0,
        spline_curve_points=[],
        gradient_magnitude=30.0,
        is_gradient_refined=True,
    )
    margin_os = MarginResult(
        margin_point=(150.0, 44.0),
        y_init_spline=44.0,
        y_margin_refined=44.0,
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
        sharpness_score=100.0,
    )

    assert "REFLEX_BELOW_LOWER_MARGIN" in meas.right_eye_od.warning_flags
    assert meas.right_eye_od.status == "LOW_CONFIDENCE"
