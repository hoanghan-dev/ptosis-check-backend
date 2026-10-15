"""
Endpoint cấu hình thông số và ngưỡng sàng lọc lâm sàng.
Tuyệt đối tuân thủ G4: Không dùng từ ngữ thuộc danh mục cấm.
"""

from typing import Dict, Any
from fastapi import APIRouter
from app.core.config import settings

router = APIRouter()


@router.get("/thresholds", response_model=Dict[str, Any])
def get_screening_thresholds() -> Dict[str, Any]:
    """
    Cung cấp bảng thông số ngưỡng cấu hình sàng lọc hình thái cho Client hiển thị.
    """
    return {
        "absolute_severe_mrd1_mm": settings.ABSOLUTE_SEVERE_MRD1_MM,
        "tier_a_min_mrd1_mm": settings.TIER_A_MIN_MRD1_MM,
        "tier_b_min_mrd1_mm": settings.TIER_B_MIN_MRD1_MM,
        "physiological_asymmetry_max_mm": settings.PHYSIOLOGICAL_ASYMMETRY_MAX_MM,
        "borderline_asymmetry_max_mm": settings.BORDERLINE_ASYMMETRY_MAX_MM,
        "brow_asymmetry_threshold_px": settings.BROW_ASYMMETRY_THRESHOLD_PX,
        "max_yaw_deg": settings.MAX_YAW_DEG,
        "max_pitch_deg": settings.MAX_PITCH_DEG,
        "max_roll_deg": settings.MAX_ROLL_DEG,
        "min_eye_width_px": settings.MIN_EYE_WIDTH_PX,
        "min_iris_diameter_px": settings.MIN_IRIS_DIAMETER_PX,
        "min_brightness": settings.MIN_BRIGHTNESS,
        "max_brightness": settings.MAX_BRIGHTNESS,
        "max_glare_ratio": settings.MAX_GLARE_RATIO,
        "min_eye_aperture_ratio": settings.MIN_EYE_APERTURE_RATIO,
        "laplacian_sharp_threshold": settings.LAPLACIAN_SHARP_THRESHOLD,
        "assumed_corneal_diameter_mm": settings.ASSUMED_CORNEAL_DIAMETER_MM,
        "nominal_tolerance_mm": settings.NOMINAL_TOLERANCE_MM,
    }
