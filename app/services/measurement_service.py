"""
Dịch vụ Khối Đo lường MRD1, Độ Bất đối xứng & Độ Tin cậy (Module M6: Measurement Service).
Tính toán toàn bộ các thông số hình thái mi mắt từ dữ liệu mốc giải phẫu:
- MRD1 (Margin Reflex Distance 1): MRD1_px = y_ref - y_margin, quy đổi sang mm qua S_iris.
- Khoảng ước lượng dung sai: [MRD1_mm - 0.5, MRD1_mm + 0.5] mm.
- Chiều cao khe mi PFH (Palpebral Fissure Height): PFH_px = y_lower - y_upper, quy đổi mm.
- So sánh bất đối xứng hai bên: delta_MRD1, signed_delta_MRD1, delta_PFH, delta_Brow (cung mày).
- Phân loại đối xứng: PHYSIOLOGICAL_SYMMETRY, BORDERLINE_ASYMMETRY, SIGNIFICANT_ASYMMETRY.
- Điểm tin cậy tổng hợp C_MRD1 = 0.25*Q_anh + 0.35*C_bo_mi + 0.25*C_phan_quang + 0.15*G_nhat_quan.
- Phân định trạng thái: VALID, LOW_CONFIDENCE, INVALID, NOT_DETECTABLE.
Tuân thủ tuyệt đối G2 (In-Memory Zero-Retention) và G4 (Non-Diagnostic Safety).
"""

from typing import List, Optional, Tuple
import numpy as np

from app.core.config import settings
from app.schemas.measurement import (
    BilateralComparison,
    MeasurementsContainer,
    PointCoordinates,
    SingleEyeMeasurement,
)
from app.services.eyelid_margin_service import MarginResult
from app.services.landmark_service import LandmarkResult
from app.services.reflex_service import ReflexResult
from app.utils.spline_interpolator import EyelidSplineInterpolator


def _interpolate_lower_eyelid_y(
    landmarks_lower: List[Tuple[float, float]],
    x_target: float,
) -> float:
    """Nội suy tung độ bờ mi dưới tại hoành độ x_target."""
    if not landmarks_lower or len(landmarks_lower) < 2:
        return 0.0
    sorted_pts = sorted(landmarks_lower, key=lambda p: float(p[0]))
    xs = [float(p[0]) for p in sorted_pts]
    ys = [float(p[1]) for p in sorted_pts]
    return float(np.interp(x_target, xs, ys))


def _calculate_single_eye_confidence(
    sharpness_score: Optional[float],
    margin_res: MarginResult,
    reflex_res: ReflexResult,
    is_hvid_consistent: bool,
    hvid_asymmetry_ratio: float,
) -> float:
    """
    Tính điểm tin cậy tổng hợp:
    C_MRD1 = 0.25*Q_anh + 0.35*C_bo_mi + 0.25*C_phan_quang + 0.15*G_nhat_quan
    """
    # 1. Chất lượng ảnh (Q_anh)
    sharp = sharpness_score if sharpness_score is not None else 100.0
    q_anh = float(np.clip(sharp / 100.0, 0.50, 1.0))

    # 2. Độ tin cậy bờ mi (C_bo_mi)
    c_bo_mi = 1.0 if margin_res.is_gradient_refined else 0.70

    # 3. Độ tin cậy phản quang (C_phan_quang)
    c_phan_quang = float(reflex_res.confidence_score)

    # 4. Tính nhất quán sinh học mống mắt (G_nhat_quan)
    if is_hvid_consistent:
        g_nhat_quan = 1.0
    else:
        g_nhat_quan = float(np.clip(1.0 - hvid_asymmetry_ratio * 4.0, 0.40, 0.90))

    composite_score = (
        0.25 * q_anh +
        0.35 * c_bo_mi +
        0.25 * c_phan_quang +
        0.15 * g_nhat_quan
    )
    return round(float(np.clip(composite_score, 0.0, 1.0)), 2)


def _determine_measurement_status(confidence_score: float) -> str:
    """Xác định trạng thái phép đo dựa trên điểm tin cậy."""
    if confidence_score >= settings.CONFIDENCE_VALID_THRESHOLD:
        return "VALID"
    elif confidence_score >= settings.CONFIDENCE_LOW_THRESHOLD:
        return "LOW_CONFIDENCE"
    else:
        return "INVALID"


def compute_measurements(
    landmarks: LandmarkResult,
    reflex_od: ReflexResult,
    reflex_os: ReflexResult,
    margin_od: MarginResult,
    margin_os: MarginResult,
    sharpness_score: Optional[float] = None,
) -> Tuple[MeasurementsContainer, BilateralComparison]:
    """
    Tính toán toàn diện các chỉ số đo lường hình thái hai mắt:
    - MRD1 (px, mm, khoảng ước lượng)
    - PFH (px, mm)
    - So sánh bất đối xứng hai bên và độ chênh lệch cung mày
    - Điểm tin cậy tổng hợp và phân loại trạng thái

    Returns:
        Tuple chứa MeasurementsContainer và BilateralComparison.
    """
    # -------------------------------------------------------------------------
    # 1. MẮT PHẢI (OD)
    # -------------------------------------------------------------------------
    x_ref_od = reflex_od.reflex_point[0]
    y_ref_od = reflex_od.reflex_point[1]
    y_margin_od = margin_od.margin_point[1]

    # Quy ước dấu: Y tăng dần từ trên xuống dưới
    # Bờ mi trên ở phía trên phản quang (y_margin < y_ref) -> MRD1 > 0
    mrd1_px_od = y_ref_od - y_margin_od
    mrd1_mm_od = mrd1_px_od * landmarks.scale_factor_od

    tol = settings.NOMINAL_TOLERANCE_MM
    range_od = f"{mrd1_mm_od - tol:.1f} - {mrd1_mm_od + tol:.1f} mm"

    # Chiều cao khe mi PFH
    y_lower_od = _interpolate_lower_eyelid_y(landmarks.lower_eyelid_od, x_ref_od)
    pfh_px_od = max(0.0, y_lower_od - y_margin_od)
    pfh_mm_od = pfh_px_od * landmarks.scale_factor_od

    # Điểm tin cậy tổng hợp OD
    conf_od = _calculate_single_eye_confidence(
        sharpness_score=sharpness_score,
        margin_res=margin_od,
        reflex_res=reflex_od,
        is_hvid_consistent=landmarks.is_hvid_biologically_consistent,
        hvid_asymmetry_ratio=landmarks.hvid_asymmetry_ratio,
    )
    status_od = _determine_measurement_status(conf_od)

    flags_od = list(dict.fromkeys(landmarks.warning_flags + reflex_od.warning_flags + margin_od.warning_flags))
    if status_od == "LOW_CONFIDENCE":
        flags_od.append("LOW_MEASUREMENT_CONFIDENCE")

    single_od = SingleEyeMeasurement(
        mrd1_px=round(mrd1_px_od, 2),
        mrd1_mm=round(mrd1_mm_od, 2),
        mrd1_estimated_range=range_od,
        pfh_px=round(pfh_px_od, 2),
        pfh_mm=round(pfh_mm_od, 2),
        reference_point_used=reflex_od.reference_point_used,
        reference_coordinates=PointCoordinates(x=x_ref_od, y=y_ref_od),
        eyelid_margin_point=PointCoordinates(x=margin_od.margin_point[0], y=y_margin_od),
        status=status_od,
        confidence_score=conf_od,
        warning_flags=flags_od,
    )

    # -------------------------------------------------------------------------
    # 2. MẮT TRÁI (OS)
    # -------------------------------------------------------------------------
    x_ref_os = reflex_os.reflex_point[0]
    y_ref_os = reflex_os.reflex_point[1]
    y_margin_os = margin_os.margin_point[1]

    mrd1_px_os = y_ref_os - y_margin_os
    mrd1_mm_os = mrd1_px_os * landmarks.scale_factor_os
    range_os = f"{mrd1_mm_os - tol:.1f} - {mrd1_mm_os + tol:.1f} mm"

    y_lower_os = _interpolate_lower_eyelid_y(landmarks.lower_eyelid_os, x_ref_os)
    pfh_px_os = max(0.0, y_lower_os - y_margin_os)
    pfh_mm_os = pfh_px_os * landmarks.scale_factor_os

    conf_os = _calculate_single_eye_confidence(
        sharpness_score=sharpness_score,
        margin_res=margin_os,
        reflex_res=reflex_os,
        is_hvid_consistent=landmarks.is_hvid_biologically_consistent,
        hvid_asymmetry_ratio=landmarks.hvid_asymmetry_ratio,
    )
    status_os = _determine_measurement_status(conf_os)

    flags_os = list(dict.fromkeys(landmarks.warning_flags + reflex_os.warning_flags + margin_os.warning_flags))
    if status_os == "LOW_CONFIDENCE":
        flags_os.append("LOW_MEASUREMENT_CONFIDENCE")

    single_os = SingleEyeMeasurement(
        mrd1_px=round(mrd1_px_os, 2),
        mrd1_mm=round(mrd1_mm_os, 2),
        mrd1_estimated_range=range_os,
        pfh_px=round(pfh_px_os, 2),
        pfh_mm=round(pfh_mm_os, 2),
        reference_point_used=reflex_os.reference_point_used,
        reference_coordinates=PointCoordinates(x=x_ref_os, y=y_ref_os),
        eyelid_margin_point=PointCoordinates(x=margin_os.margin_point[0], y=y_margin_os),
        status=status_os,
        confidence_score=conf_os,
        warning_flags=flags_os,
    )

    measurements_container = MeasurementsContainer(
        right_eye_od=single_od,
        left_eye_os=single_os,
    )

    # -------------------------------------------------------------------------
    # 3. SO SÁNH BẤT ĐỐI XỨNG HAI BÊN (BILATERAL COMPARISON)
    # -------------------------------------------------------------------------
    delta_mrd1_mm = abs(mrd1_mm_os - mrd1_mm_od)
    signed_delta_mrd1_mm = mrd1_mm_os - mrd1_mm_od
    delta_pfh_mm = abs(pfh_mm_os - pfh_mm_od)

    # Phân loại đối xứng
    if delta_mrd1_mm <= settings.PHYSIOLOGICAL_ASYMMETRY_MAX_MM:
        asymmetry_classification = "PHYSIOLOGICAL_SYMMETRY"
    elif delta_mrd1_mm <= settings.BORDERLINE_ASYMMETRY_MAX_MM:
        asymmetry_classification = "BORDERLINE_ASYMMETRY"
    else:
        asymmetry_classification = "SIGNIFICANT_ASYMMETRY"

    # Chênh lệch độ cao cung mày (Eyebrow Height Discrepancy)
    # Khoảng cách từ tâm mống mắt / phản quang lên điểm giữa cung mày
    brow_center_od_y = landmarks.eyebrow_od[2][1] if len(landmarks.eyebrow_od) > 2 else y_ref_od
    brow_center_os_y = landmarks.eyebrow_os[2][1] if len(landmarks.eyebrow_os) > 2 else y_ref_os

    brow_height_od = y_ref_od - brow_center_od_y
    brow_height_os = y_ref_os - brow_center_os_y
    delta_brow_px = abs(brow_height_os - brow_height_od)

    bilateral_comparison = BilateralComparison(
        delta_mrd1_mm=round(delta_mrd1_mm, 2),
        signed_delta_mrd1_mm=round(signed_delta_mrd1_mm, 2),
        asymmetry_classification=asymmetry_classification,
        delta_pfh_mm=round(delta_pfh_mm, 2),
        delta_brow_px=round(delta_brow_px, 2),
        eyebrow_height_discrepancy_px=round(delta_brow_px, 2),
    )

    return measurements_container, bilateral_comparison
