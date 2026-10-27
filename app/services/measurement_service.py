"""
Dịch vụ Khối Đo lường MRD1, Độ Bất đối xứng & Độ Tin cậy (Module M6: Measurement Service - Phase 6).
Tính toán toàn bộ các thông số hình thái mi mắt từ dữ liệu mốc giải phẫu và mô hình học sâu:
- MRD1 (Margin Reflex Distance 1): MRD1_px = y_ref - y_margin, quy đổi sang mm qua S_iris.
- Khoảng ước lượng dung sai: [MRD1_mm - 0.5, MRD1_mm + 0.5] mm.
- Chiều cao khe mi PFH (Palpebral Fissure Height): PFH_px = y_lower - y_upper, quy đổi mm.
- So sánh bất đối xứng hai bên: delta_MRD1, signed_delta_MRD1, delta_PFH, delta_Brow (cung mày).
- Phân loại đối xứng: PHYSIOLOGICAL_SYMMETRY, BORDERLINE_ASYMMETRY, SIGNIFICANT_ASYMMETRY.
- Điểm tin cậy tổng hợp C_MRD1 = 0.25*Q_anh + 0.35*C_bo_mi + 0.25*C_phan_quang + 0.15*G_nhat_quan.
- Phân định trạng thái: VALID, LOW_CONFIDENCE, INVALID, NOT_DETECTABLE.
Hỗ trợ liền mạch cả kết quả từ mô hình học sâu V2 (EyelidUNet, IrisFour, ReflexModel)
lẫn đối tượng mốc giải phẫu chuẩn trước đây (Backward Compatibility 100%).
Tuân thủ tuyệt đối G2 (In-Memory Zero-Retention) và G4 (Non-Diagnostic Safety).
"""

from typing import Any, List, Optional, Tuple, Union
import numpy as np

from app.core.config import settings
from app.schemas.measurement import (
    BilateralComparison,
    MeasurementStatusEnum,
    MeasurementsContainer,
    PointCoordinates,
    ReflexVisibilityEnum,
    SingleEyeMeasurement,
)
from app.services.eyelid_margin_service import MarginResult
from app.services.eyelid_model_service import EyelidSegmentationResult
from app.services.iris_model_service import IrisGeometryResult
from app.services.landmark_service import LandmarkResult
from app.services.reflex_model_service import ReflexModelResult
from app.services.reflex_service import ReflexResult

PHYSIOLOGICAL_MRD1_MAX_MM: float = 6.0
PHYSIOLOGICAL_MRD1_MIN_MM: float = -2.0


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
    c_bo_mi: float = 1.0,
    c_phan_quang: float = 1.0,
    is_hvid_consistent: bool = True,
    hvid_asymmetry_ratio: float = 0.0,
    *,
    margin_res: Any = None,
    reflex_res: Any = None,
) -> float:
    """
    Tính điểm tin cậy tổng hợp:
    C_MRD1 = 0.25*Q_anh + 0.35*C_bo_mi + 0.25*C_phan_quang + 0.15*G_nhat_quan
    """
    if margin_res is not None:
        c_bo_mi = getattr(margin_res, "confidence_score", 1.0 if getattr(margin_res, "is_gradient_refined", False) else 0.70)
    if reflex_res is not None:
        c_phan_quang = getattr(reflex_res, "confidence_score", 1.0)
    # 1. Chất lượng ảnh (Q_anh)
    sharp = sharpness_score if sharpness_score is not None else 100.0
    q_anh = float(np.clip(sharp / 100.0, 0.50, 1.0))

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


def _determine_measurement_status(confidence_score: float) -> MeasurementStatusEnum:
    """Xác định trạng thái phép đo dựa trên điểm tin cậy."""
    if confidence_score >= settings.CONFIDENCE_VALID_THRESHOLD:
        return MeasurementStatusEnum.VALID
    elif confidence_score >= settings.CONFIDENCE_LOW_THRESHOLD:
        return MeasurementStatusEnum.LOW_CONFIDENCE
    else:
        return MeasurementStatusEnum.INVALID


def compute_measurements(
    landmarks: Optional[LandmarkResult] = None,
    reflex_od: Optional[Union[ReflexResult, ReflexModelResult]] = None,
    reflex_os: Optional[Union[ReflexResult, ReflexModelResult]] = None,
    margin_od: Optional[Union[MarginResult, EyelidSegmentationResult]] = None,
    margin_os: Optional[Union[MarginResult, EyelidSegmentationResult]] = None,
    sharpness_score: Optional[float] = None,
    iris_od: Optional[IrisGeometryResult] = None,
    iris_os: Optional[IrisGeometryResult] = None,
) -> Tuple[MeasurementsContainer, BilateralComparison]:
    """
    Tính toán toàn diện các chỉ số đo lường hình thái hai mắt:
    - MRD1 (px, mm, khoảng ước lượng)
    - PFH (px, mm)
    - So sánh bất đối xứng hai bên và độ chênh lệch cung mày
    - Điểm tin cậy tổng hợp và phân loại trạng thái

    Hỗ trợ cả đối tượng từ pipeline mới (IrisGeometryResult, ReflexModelResult, EyelidSegmentationResult)
    và pipeline trước đó (LandmarkResult, ReflexResult, MarginResult).

    Returns:
        Tuple chứa MeasurementsContainer và BilateralComparison.
    """
    # -------------------------------------------------------------------------
    # 0. Xác định hệ số thang đo (Scale Factor) và độ bất đối xứng mống mắt
    # -------------------------------------------------------------------------
    if iris_od is not None and iris_os is not None:
        scale_od = iris_od.scale_factor_mm_per_px
        scale_os = iris_os.scale_factor_mm_per_px
        hvid_diff = abs(iris_od.hvid_px - iris_os.hvid_px)
        max_hvid = max(iris_od.hvid_px, iris_os.hvid_px, 1.0)
        hvid_asymmetry_ratio = hvid_diff / max_hvid
        is_hvid_consistent = (hvid_asymmetry_ratio <= settings.IRIS_MAX_BILATERAL_DIFF_RATIO)
        flags_scale: List[str] = []
        if not is_hvid_consistent:
            flags_scale.append("HVID_ASYMMETRY_EXCEEDED")
    elif landmarks is not None:
        scale_od = landmarks.scale_factor_od
        scale_os = landmarks.scale_factor_os
        hvid_asymmetry_ratio = landmarks.hvid_asymmetry_ratio
        is_hvid_consistent = landmarks.is_hvid_biologically_consistent
        flags_scale = list(landmarks.warning_flags)
    else:
        # Trường hợp dự phòng an toàn tuyệt đối
        scale_od = 0.25
        scale_os = 0.25
        hvid_asymmetry_ratio = 0.0
        is_hvid_consistent = True
        flags_scale = []

    # -------------------------------------------------------------------------
    # 1. MẮT PHẢI (OD)
    # -------------------------------------------------------------------------
    # Trích xuất thông số phản quang / điểm tham chiếu OD
    if isinstance(reflex_od, ReflexModelResult):
        x_ref_od = reflex_od.reference_point[0]
        y_ref_od = reflex_od.reference_point[1]
        ref_used_od = reflex_od.reference_point_used
        fallback_type_od = reflex_od.fallback_type
        conf_reflex_od = reflex_od.confidence_score
        flags_reflex_od = list(reflex_od.warning_flags)
        glint_aspect_od = reflex_od.glint_aspect_ratio
    elif isinstance(reflex_od, ReflexResult):
        x_ref_od = reflex_od.reflex_point[0]
        y_ref_od = reflex_od.reflex_point[1]
        ref_used_od = reflex_od.reference_point_used
        fallback_type_od = getattr(reflex_od, "fallback_type", None)
        conf_reflex_od = reflex_od.confidence_score
        flags_reflex_od = list(reflex_od.warning_flags)
        glint_aspect_od = reflex_od.glint_aspect_ratio
    else:
        x_ref_od, y_ref_od = (0.0, 0.0)
        ref_used_od = "CORNEAL_LIGHT_REFLEX"
        fallback_type_od = None
        conf_reflex_od = 1.0
        flags_reflex_od = []
        glint_aspect_od = None

    # Trích xuất thông số bờ mi OD
    if isinstance(margin_od, EyelidSegmentationResult):
        x_margin_od = margin_od.margin_point[0]
        y_margin_od = margin_od.margin_point[1]
        y_lower_od = margin_od.lower_margin_point[1]
        pfh_px_od = margin_od.pfh_px
        conf_margin_od = margin_od.confidence_score
        flags_margin_od = list(margin_od.warning_flags)
    elif isinstance(margin_od, MarginResult):
        x_margin_od = margin_od.margin_point[0]
        y_margin_od = margin_od.margin_point[1]
        if landmarks is not None and landmarks.lower_eyelid_od:
            y_lower_od = _interpolate_lower_eyelid_y(landmarks.lower_eyelid_od, x_ref_od)
        else:
            y_lower_od = y_margin_od + 20.0
        pfh_px_od = max(0.0, y_lower_od - y_margin_od)
        conf_margin_od = 1.0 if margin_od.is_gradient_refined else 0.70
        flags_margin_od = list(margin_od.warning_flags)
    else:
        x_margin_od, y_margin_od = (x_ref_od, y_ref_od - 16.0)
        y_lower_od = y_ref_od + 16.0
        pfh_px_od = 32.0
        conf_margin_od = 0.90
        flags_margin_od = []

    # Quy ước dấu: Y tăng dần từ trên xuống dưới
    # Bờ mi trên ở phía trên phản quang (y_margin < y_ref) -> MRD1 > 0
    raw_mrd1_px_od = y_ref_od - y_margin_od
    tol = settings.NOMINAL_TOLERANCE_MM

    flags_od = list(dict.fromkeys((landmarks.warning_flags if landmarks else []) + reflex_od.warning_flags + margin_od.warning_flags))

    # Chiều cao khe mi PFH
    if landmarks is not None and landmarks.lower_eyelid_od:
        y_lower_od = _interpolate_lower_eyelid_y(landmarks.lower_eyelid_od, x_ref_od)
        pfh_px_od = max(0.0, y_lower_od - y_margin_od)
    pfh_mm_od = pfh_px_od * scale_od

    if raw_mrd1_px_od < 0:
        # Chốt chặn dấu MRD1 âm (Phase A - A2)
        mrd1_px_od = None
        mrd1_mm_od = None
        range_od = None
        status_od = MeasurementStatusEnum.MEASUREMENT_UNAVAILABLE
        conf_od = 0.0
        ref_coords_od = None
        reflex_vis_od = ReflexVisibilityEnum.OCCLUDED_BY_LID
        if "ANATOMICAL_NEGATIVE_SUSPECTED" not in flags_od:
            flags_od.append("ANATOMICAL_NEGATIVE_SUSPECTED")
    else:
        mrd1_px_od = round(raw_mrd1_px_od, 2)
        mrd1_mm_od = round(raw_mrd1_px_od * scale_od, 2)
        range_od = f"{mrd1_mm_od - tol:.1f} - {mrd1_mm_od + tol:.1f} mm"

        conf_od = _calculate_single_eye_confidence(
            sharpness_score=sharpness_score,
            c_bo_mi=conf_margin_od,
            c_phan_quang=conf_reflex_od,
            is_hvid_consistent=is_hvid_consistent,
            hvid_asymmetry_ratio=hvid_asymmetry_ratio,
            margin_res=margin_od,
            reflex_res=reflex_od,
        )
        status_od = _determine_measurement_status(conf_od)
        if status_od == MeasurementStatusEnum.LOW_CONFIDENCE and "LOW_MEASUREMENT_CONFIDENCE" not in flags_od:
            flags_od.append("LOW_MEASUREMENT_CONFIDENCE")
        ref_coords_od = PointCoordinates(x=x_ref_od, y=y_ref_od)
        reflex_vis_od = getattr(reflex_od, "reflex_visibility", ReflexVisibilityEnum.VISIBLE)

    # Chốt chặn kiểm tra ngưỡng sinh lý lâm sàng và hình thái học
    PHYSIOLOGICAL_MRD1_MAX_MM = 6.0
    PHYSIOLOGICAL_MRD1_MIN_MM = 0.0

    if mrd1_mm_od is not None and (mrd1_mm_od > PHYSIOLOGICAL_MRD1_MAX_MM or mrd1_mm_od < PHYSIOLOGICAL_MRD1_MIN_MM):
        if "PHYSIOLOGICAL_LIMIT_EXCEEDED_MRD1" not in flags_od:
            flags_od.append("PHYSIOLOGICAL_LIMIT_EXCEEDED_MRD1")
        conf_od = min(conf_od, 0.49)
        status_od = MeasurementStatusEnum.LOW_CONFIDENCE

    if y_ref_od > y_lower_od:
        if "REFLEX_BELOW_LOWER_MARGIN" not in flags_od:
            flags_od.append("REFLEX_BELOW_LOWER_MARGIN")
        conf_od = min(conf_od, 0.40)
        status_od = MeasurementStatusEnum.LOW_CONFIDENCE

    if status_od == MeasurementStatusEnum.LOW_CONFIDENCE and "LOW_MEASUREMENT_CONFIDENCE" not in flags_od:
        flags_od.append("LOW_MEASUREMENT_CONFIDENCE")

    single_od = SingleEyeMeasurement(
        mrd1_px=mrd1_px_od,
        mrd1_mm=mrd1_mm_od,
        mrd1_estimated_range=range_od,
        pfh_px=round(pfh_px_od, 2),
        pfh_mm=round(pfh_mm_od, 2),
        reference_point_used=reflex_od.reference_point_used,
        reference_coordinates=ref_coords_od,
        eyelid_margin_point=PointCoordinates(x=margin_od.margin_point[0], y=y_margin_od),
        status=status_od,
        confidence_score=conf_od,
        warning_flags=flags_od,
        reflex_visibility=reflex_vis_od,
    )

    # -------------------------------------------------------------------------
    # 2. MẮT TRÁI (OS)
    # -------------------------------------------------------------------------
    if isinstance(reflex_os, ReflexModelResult):
        x_ref_os = reflex_os.reference_point[0]
        y_ref_os = reflex_os.reference_point[1]
        ref_used_os = reflex_os.reference_point_used
        fallback_type_os = reflex_os.fallback_type
        conf_reflex_os = reflex_os.confidence_score
        flags_reflex_os = list(reflex_os.warning_flags)
        glint_aspect_os = reflex_os.glint_aspect_ratio
    elif isinstance(reflex_os, ReflexResult):
        x_ref_os = reflex_os.reflex_point[0]
        y_ref_os = reflex_os.reflex_point[1]
        ref_used_os = reflex_os.reference_point_used
        fallback_type_os = getattr(reflex_os, "fallback_type", None)
        conf_reflex_os = reflex_os.confidence_score
        flags_reflex_os = list(reflex_os.warning_flags)
        glint_aspect_os = reflex_os.glint_aspect_ratio
    else:
        x_ref_os, y_ref_os = (0.0, 0.0)
        ref_used_os = "CORNEAL_LIGHT_REFLEX"
        fallback_type_os = None
        conf_reflex_os = 1.0
        flags_reflex_os = []
        glint_aspect_os = None

    if isinstance(margin_os, EyelidSegmentationResult):
        x_margin_os = margin_os.margin_point[0]
        y_margin_os = margin_os.margin_point[1]
        y_lower_os = margin_os.lower_margin_point[1]
        pfh_px_os = margin_os.pfh_px
        conf_margin_os = margin_os.confidence_score
        flags_margin_os = list(margin_os.warning_flags)
    elif isinstance(margin_os, MarginResult):
        x_margin_os = margin_os.margin_point[0]
        y_margin_os = margin_os.margin_point[1]
        if landmarks is not None and landmarks.lower_eyelid_os:
            y_lower_os = _interpolate_lower_eyelid_y(landmarks.lower_eyelid_os, x_ref_os)
        else:
            y_lower_os = y_margin_os + 20.0
        pfh_px_os = max(0.0, y_lower_os - y_margin_os)
        conf_margin_os = 1.0 if margin_os.is_gradient_refined else 0.70
        flags_margin_os = list(margin_os.warning_flags)
    else:
        x_margin_os, y_margin_os = (x_ref_os, y_ref_os - 16.0)
        y_lower_os = y_ref_os + 16.0
        pfh_px_os = 32.0
        conf_margin_os = 0.90
        flags_margin_os = []

    raw_mrd1_px_os = y_ref_os - y_margin_os
    flags_os = list(dict.fromkeys((landmarks.warning_flags if landmarks else []) + reflex_os.warning_flags + margin_os.warning_flags))

    if landmarks is not None and landmarks.lower_eyelid_os:
        y_lower_os = _interpolate_lower_eyelid_y(landmarks.lower_eyelid_os, x_ref_os)
        pfh_px_os = max(0.0, y_lower_os - y_margin_os)
    pfh_mm_os = pfh_px_os * scale_os

    if raw_mrd1_px_os < 0:
        # Chốt chặn dấu MRD1 âm (Phase A - A2)
        mrd1_px_os = None
        mrd1_mm_os = None
        range_os = None
        status_os = MeasurementStatusEnum.MEASUREMENT_UNAVAILABLE
        conf_os = 0.0
        ref_coords_os = None
        reflex_vis_os = ReflexVisibilityEnum.OCCLUDED_BY_LID
        if "ANATOMICAL_NEGATIVE_SUSPECTED" not in flags_os:
            flags_os.append("ANATOMICAL_NEGATIVE_SUSPECTED")
    else:
        mrd1_px_os = round(raw_mrd1_px_os, 2)
        mrd1_mm_os = round(raw_mrd1_px_os * scale_os, 2)
        range_os = f"{mrd1_mm_os - tol:.1f} - {mrd1_mm_os + tol:.1f} mm"

        conf_os = _calculate_single_eye_confidence(
            sharpness_score=sharpness_score,
            c_bo_mi=conf_margin_os,
            c_phan_quang=conf_reflex_os,
            is_hvid_consistent=is_hvid_consistent,
            hvid_asymmetry_ratio=hvid_asymmetry_ratio,
            margin_res=margin_os,
            reflex_res=reflex_os,
        )
        status_os = _determine_measurement_status(conf_os)
        if status_os == MeasurementStatusEnum.LOW_CONFIDENCE and "LOW_MEASUREMENT_CONFIDENCE" not in flags_os:
            flags_os.append("LOW_MEASUREMENT_CONFIDENCE")
        ref_coords_os = PointCoordinates(x=x_ref_os, y=y_ref_os)
        reflex_vis_os = getattr(reflex_os, "reflex_visibility", ReflexVisibilityEnum.VISIBLE)

    if mrd1_mm_os is not None and (mrd1_mm_os > PHYSIOLOGICAL_MRD1_MAX_MM or mrd1_mm_os < PHYSIOLOGICAL_MRD1_MIN_MM):
        if "PHYSIOLOGICAL_LIMIT_EXCEEDED_MRD1" not in flags_os:
            flags_os.append("PHYSIOLOGICAL_LIMIT_EXCEEDED_MRD1")
        conf_os = min(conf_os, 0.49)
        status_os = MeasurementStatusEnum.LOW_CONFIDENCE

    if y_ref_os > y_lower_os:
        if "REFLEX_BELOW_LOWER_MARGIN" not in flags_os:
            flags_os.append("REFLEX_BELOW_LOWER_MARGIN")
        conf_os = min(conf_os, 0.40)
        status_os = MeasurementStatusEnum.LOW_CONFIDENCE

    if status_os == MeasurementStatusEnum.LOW_CONFIDENCE and "LOW_MEASUREMENT_CONFIDENCE" not in flags_os:
        flags_os.append("LOW_MEASUREMENT_CONFIDENCE")

    single_os = SingleEyeMeasurement(
        mrd1_px=mrd1_px_os,
        mrd1_mm=mrd1_mm_os,
        mrd1_estimated_range=range_os,
        pfh_px=round(pfh_px_os, 2),
        pfh_mm=round(pfh_mm_os, 2),
        reference_point_used=reflex_os.reference_point_used,
        reference_coordinates=ref_coords_os,
        eyelid_margin_point=PointCoordinates(x=margin_os.margin_point[0], y=y_margin_os),
        status=status_os,
        confidence_score=conf_os,
        warning_flags=flags_os,
        reflex_visibility=reflex_vis_os,
    )

    measurements_container = MeasurementsContainer(
        right_eye_od=single_od,
        left_eye_os=single_os,
    )

    # -------------------------------------------------------------------------
    # 3. SO SÁNH BẤT ĐỐI XỨNG HAI BÊN (BILATERAL COMPARISON)
    # -------------------------------------------------------------------------
    if mrd1_mm_od is not None and mrd1_mm_os is not None:
        delta_mrd1_mm = round(abs(mrd1_mm_os - mrd1_mm_od), 2)
        signed_delta_mrd1_mm = round(mrd1_mm_os - mrd1_mm_od, 2)
        if delta_mrd1_mm <= settings.PHYSIOLOGICAL_ASYMMETRY_MAX_MM:
            asymmetry_classification = "PHYSIOLOGICAL_SYMMETRY"
        elif delta_mrd1_mm <= settings.BORDERLINE_ASYMMETRY_MAX_MM:
            asymmetry_classification = "BORDERLINE_ASYMMETRY"
        else:
            asymmetry_classification = "SIGNIFICANT_ASYMMETRY"
    else:
        delta_mrd1_mm = None
        signed_delta_mrd1_mm = None
        asymmetry_classification = "SIGNIFICANT_ASYMMETRY"

    delta_pfh_mm = abs(pfh_mm_os - pfh_mm_od)

    # Chênh lệch độ cao cung mày (Eyebrow Height Discrepancy)
    if landmarks is not None and len(landmarks.eyebrow_od) > 2 and len(landmarks.eyebrow_os) > 2:
        brow_center_od_y = landmarks.eyebrow_od[2][1]
        brow_center_os_y = landmarks.eyebrow_os[2][1]
        brow_height_od = y_ref_od - brow_center_od_y
        brow_height_os = y_ref_os - brow_center_os_y
        delta_brow_px = abs(brow_height_os - brow_height_od)
    else:
        delta_brow_px = 0.0

    bilateral_comparison = BilateralComparison(
        delta_mrd1_mm=delta_mrd1_mm,
        signed_delta_mrd1_mm=signed_delta_mrd1_mm,
        asymmetry_classification=asymmetry_classification,
        delta_pfh_mm=round(delta_pfh_mm, 2),
        delta_brow_px=round(delta_brow_px, 2),
        eyebrow_height_discrepancy_px=round(delta_brow_px, 2),
    )

    return measurements_container, bilateral_comparison
