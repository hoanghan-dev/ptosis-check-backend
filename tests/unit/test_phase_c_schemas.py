"""
Kiểm thử chuyên sâu cho Giai đoạn Phase C:
Tiến hóa Lược đồ Dữ liệu & Hoàn thiện Cỗ máy Trạng thái (State Machine & Schema Evolution).

Bao phủ các yêu cầu của implementation_plan_prompt4.md:
1. Xác thực 5 Mock Payloads đại diện tại Mục 3:
   - Payload 1: Full-face tiêu chuẩn (TIER_A, MRD1 hợp lệ)
   - Payload 2: Không flash, Fallback mống mắt/đồng tử (TIER_B)
   - Payload 3: Che lấp phản quang hoàn toàn do bờ mi (TIER_C, mrd1=null, status=MEASUREMENT_UNAVAILABLE)
   - Payload 4: Dual-Eye crop hợp lệ (TIER_A)
   - Payload 5: Dual-Eye crop mất phản quang do chất lượng ảnh (TIER_B, mrd1=null)
2. Kiểm tra tính năng serialize/deserialize JSON với Pydantic model_validate và model_dump.
3. Kiểm tra tính tương thích của MeasurementStatusEnum và ReflexVisibilityEnum.
"""

import json
import pytest
from app.schemas.measurement import (
    MeasurementPayload,
    MeasurementStatusEnum,
    ReflexVisibilityEnum,
    SingleEyeMeasurement,
    EyeAnnotationPoints,
)

# ------------------------------------------------------------------------------
# 5 MOCK PAYLOADS TỪ MỤC 3 CỦA IMPLEMENTATION_PLAN_PROMPT4.MD
# ------------------------------------------------------------------------------

PAYLOAD_1_FULLFACE_NORMAL = {
    "session_id": "c4b12345-789a-4bc0-a123-456789abcdef",
    "timestamp": "2026-10-26T07:30:00.000Z",
    "image_quality_audit": {
        "passed": True,
        "rejection_reason": None,
        "pose": {"yaw": -1.2, "pitch": 2.1, "roll": 0.5},
        "sharpness_score": 145.2,
        "mean_illumination": 138.4,
    },
    "scale_reference": {
        "method": "HVID_WTW_POPULATION_AVERAGE",
        "assumed_corneal_diameter_mm": 11.71,
        "hvid_pixels": {"right_eye": 182.0, "left_eye": 184.0},
        "scale_factor_mm_per_px": {"right_eye": 0.0643, "left_eye": 0.0636},
    },
    "measurements": {
        "right_eye_od": {
            "mrd1_px": 54.4,
            "mrd1_mm": 3.5,
            "mrd1_estimated_range": "3.0 - 4.0 mm",
            "pfh_px": 142.1,
            "pfh_mm": 9.14,
            "reference_point_used": "CORNEAL_LIGHT_REFLEX",
            "reference_coordinates": {"x": 420.5, "y": 512.3},
            "eyelid_margin_point": {"x": 420.5, "y": 457.9},
            "status": "VALID",
            "confidence_score": 0.95,
            "warning_flags": [],
            "reflex_visibility": "VISIBLE",
        },
        "left_eye_os": {
            "mrd1_px": 52.8,
            "mrd1_mm": 3.36,
            "mrd1_estimated_range": "2.9 - 3.9 mm",
            "pfh_px": 139.5,
            "pfh_mm": 8.87,
            "reference_point_used": "CORNEAL_LIGHT_REFLEX",
            "reference_coordinates": {"x": 680.2, "y": 511.8},
            "eyelid_margin_point": {"x": 680.2, "y": 459.0},
            "status": "VALID",
            "confidence_score": 0.94,
            "warning_flags": [],
            "reflex_visibility": "VISIBLE",
        },
    },
    "bilateral_comparison": {
        "delta_mrd1_mm": 0.14,
        "signed_delta_mrd1_mm": -0.14,
        "asymmetry_classification": "PHYSIOLOGICAL_SYMMETRY",
        "delta_pfh_mm": 0.27,
        "delta_brow_px": 1.5,
        "eyebrow_height_discrepancy_px": 1.5,
    },
    "triage_result": {
        "tier": "TIER_A",
        "trigger_reasons": ["PHYSIOLOGICAL_NORMAL"],
        "description_vi": "Không phát hiện dấu hiệu bất thường.",
        "recommendation_vi": "Duy trì sinh hoạt lành mạnh.",
        "safety_disclaimer_vi": "Kết quả chỉ mang tính sàng lọc.",
        "is_bilateral_severe_triggered": False,
    },
    "annotated_overlay_coordinates": {
        "right_eye": {
            "corneal_reflex": [420.5, 512.3],
            "pupil_center": [420.2, 511.9],
            "iris_limbus_contour": [[329.5, 512.3], [511.5, 512.3]],
            "upper_eyelid_contour": [[330.0, 480.0], [420.5, 457.9], [510.0, 485.0]],
        },
        "left_eye": {
            "corneal_reflex": [680.2, 511.8],
            "pupil_center": [680.0, 511.5],
            "iris_limbus_contour": [[588.2, 511.8], [772.2, 511.8]],
            "upper_eyelid_contour": [[590.0, 483.0], [680.2, 459.0], [770.0, 482.0]],
        },
    },
}

PAYLOAD_2_FALLBACK_NO_FLASH = {
    "session_id": "e8a98765-4321-4def-b456-789012abcdef",
    "timestamp": "2026-10-26T07:35:00.000Z",
    "image_quality_audit": {
        "passed": True,
        "rejection_reason": None,
        "pose": {"yaw": 2.4, "pitch": -3.1, "roll": 1.2},
        "sharpness_score": 110.5,
        "mean_illumination": 88.0,
    },
    "scale_reference": {
        "method": "HVID_WTW_POPULATION_AVERAGE",
        "assumed_corneal_diameter_mm": 11.71,
        "hvid_pixels": {"right_eye": 140.0, "left_eye": 142.0},
        "scale_factor_mm_per_px": {"right_eye": 0.0836, "left_eye": 0.0824},
    },
    "measurements": {
        "right_eye_od": {
            "mrd1_px": 35.0,
            "mrd1_mm": 2.93,
            "mrd1_estimated_range": "2.4 - 3.4 mm",
            "pfh_px": 98.0,
            "pfh_mm": 8.19,
            "reference_point_used": "PUPIL_CENTER_FALLBACK",
            "reference_coordinates": {"x": 310.2, "y": 405.0},
            "eyelid_margin_point": {"x": 310.2, "y": 370.0},
            "status": "LOW_CONFIDENCE",
            "confidence_score": 0.65,
            "warning_flags": ["REFLEX_FALLBACK_TO_PUPIL", "LOW_MEASUREMENT_CONFIDENCE"],
            "reflex_visibility": "NOT_VISIBLE_QUALITY",
        },
        "left_eye_os": {
            "mrd1_px": 34.0,
            "mrd1_mm": 2.80,
            "mrd1_estimated_range": "2.3 - 3.3 mm",
            "pfh_px": 96.0,
            "pfh_mm": 7.91,
            "reference_point_used": "PUPIL_CENTER_FALLBACK",
            "reference_coordinates": {"x": 520.4, "y": 404.2},
            "eyelid_margin_point": {"x": 520.4, "y": 370.2},
            "status": "LOW_CONFIDENCE",
            "confidence_score": 0.65,
            "warning_flags": ["REFLEX_FALLBACK_TO_PUPIL", "LOW_MEASUREMENT_CONFIDENCE"],
            "reflex_visibility": "NOT_VISIBLE_QUALITY",
        },
    },
    "bilateral_comparison": {
        "delta_mrd1_mm": 0.13,
        "signed_delta_mrd1_mm": 0.13,
        "asymmetry_classification": "PHYSIOLOGICAL_SYMMETRY",
        "delta_pfh_mm": 0.28,
        "delta_brow_px": 0.5,
        "eyebrow_height_discrepancy_px": 0.5,
    },
    "triage_result": {
        "tier": "TIER_B",
        "trigger_reasons": ["PUPIL_FALLBACK_ACTIVE"],
        "description_vi": "Ảnh thiếu phản quang flash.",
        "recommendation_vi": "Bật flash chụp lại.",
        "safety_disclaimer_vi": "Chỉ mang tính sàng lọc.",
        "is_bilateral_severe_triggered": False,
    },
    "annotated_overlay_coordinates": {
        "right_eye": {
            "corneal_reflex": [310.2, 405.0],
            "pupil_center": [310.2, 405.0],
            "iris_limbus_contour": [[240.2, 405.0], [380.2, 405.0]],
            "upper_eyelid_contour": [[242.0, 390.0], [310.2, 376.5], [378.0, 392.0]],
        },
        "left_eye": {
            "corneal_reflex": [520.4, 404.2],
            "pupil_center": [520.4, 404.2],
            "iris_limbus_contour": [[449.4, 404.2], [591.4, 404.2]],
            "upper_eyelid_contour": [[451.0, 391.0], [520.4, 378.0], [589.0, 393.0]],
        },
    },
}

PAYLOAD_3_OCCLUDED_SEVERE = {
    "session_id": "f1c54321-1234-4567-890a-bcdef1234567",
    "timestamp": "2026-10-26T07:40:00.000Z",
    "image_quality_audit": {
        "passed": True,
        "rejection_reason": None,
        "pose": {"yaw": 0.8, "pitch": 1.5, "roll": -0.4},
        "sharpness_score": 128.0,
        "mean_illumination": 115.0,
    },
    "scale_reference": {
        "method": "HVID_WTW_POPULATION_AVERAGE",
        "assumed_corneal_diameter_mm": 11.71,
        "hvid_pixels": {"right_eye": 160.0, "left_eye": 162.0},
        "scale_factor_mm_per_px": {"right_eye": 0.0732, "left_eye": 0.0723},
    },
    "measurements": {
        "right_eye_od": {
            "mrd1_px": None,
            "mrd1_mm": None,
            "mrd1_estimated_range": None,
            "pfh_px": 42.0,
            "pfh_mm": 3.07,
            "reference_point_used": "NONE",
            "reference_coordinates": None,
            "eyelid_margin_point": {"x": 380.0, "y": 465.2},
            "status": "MEASUREMENT_UNAVAILABLE",
            "confidence_score": 0.0,
            "warning_flags": [
                "REFLEX_OCCLUDED_BY_UPPER_LID",
                "VISUAL_AXIS_OBSTRUCTION_SUSPECTED",
            ],
            "reflex_visibility": "OCCLUDED_BY_LID",
        },
        "left_eye_os": {
            "mrd1_px": 48.2,
            "mrd1_mm": 3.48,
            "mrd1_estimated_range": "3.0 - 4.0 mm",
            "pfh_px": 130.5,
            "pfh_mm": 9.44,
            "reference_point_used": "CORNEAL_LIGHT_REFLEX",
            "reference_coordinates": {"x": 640.5, "y": 450.0},
            "eyelid_margin_point": {"x": 640.5, "y": 401.8},
            "status": "VALID",
            "confidence_score": 0.93,
            "warning_flags": [],
            "reflex_visibility": "VISIBLE",
        },
    },
    "bilateral_comparison": {
        "delta_mrd1_mm": None,
        "signed_delta_mrd1_mm": None,
        "asymmetry_classification": "SIGNIFICANT_ASYMMETRY",
        "delta_pfh_mm": 6.37,
        "delta_brow_px": 14.5,
        "eyebrow_height_discrepancy_px": 14.5,
    },
    "triage_result": {
        "tier": "TIER_C",
        "trigger_reasons": [
            "REFLEX_OCCLUDED_SEVERE",
            "UNILATERAL_VISUAL_AXIS_OBSTRUCTION",
        ],
        "description_vi": "Che phủ bờ mi.",
        "recommendation_vi": "Khám chuyên khoa mắt.",
        "safety_disclaimer_vi": "Chỉ mang tính sàng lọc.",
        "is_bilateral_severe_triggered": False,
    },
    "annotated_overlay_coordinates": {
        "right_eye": {
            "corneal_reflex": None,
            "pupil_center": None,
            "iris_limbus_contour": [[300.0, 450.0], [460.0, 450.0]],
            "upper_eyelid_contour": [[305.0, 470.0], [380.0, 465.2], [455.0, 472.0]],
        },
        "left_eye": {
            "corneal_reflex": [640.5, 450.0],
            "pupil_center": [640.3, 449.8],
            "iris_limbus_contour": [[559.5, 450.0], [721.5, 450.0]],
            "upper_eyelid_contour": [[560.0, 425.0], [640.5, 401.8], [720.0, 428.0]],
        },
    },
}

PAYLOAD_4_DUAL_EYE_CROP = {
    "session_id": "a9e11223-3344-5566-7788-99aabbccdde0",
    "timestamp": "2026-10-26T07:45:00.000Z",
    "image_quality_audit": {
        "passed": True,
        "rejection_reason": None,
        "pose": {"yaw": 0.0, "pitch": 0.0, "roll": 1.1},
        "sharpness_score": 210.4,
        "mean_illumination": 142.0,
    },
    "scale_reference": {
        "method": "HVID_WTW_POPULATION_AVERAGE",
        "assumed_corneal_diameter_mm": 11.71,
        "hvid_pixels": {"right_eye": 245.0, "left_eye": 248.0},
        "scale_factor_mm_per_px": {"right_eye": 0.0478, "left_eye": 0.0472},
    },
    "measurements": {
        "right_eye_od": {
            "mrd1_px": 78.5,
            "mrd1_mm": 3.75,
            "mrd1_estimated_range": "3.3 - 4.3 mm",
            "pfh_px": 212.0,
            "pfh_mm": 10.13,
            "reference_point_used": "CORNEAL_LIGHT_REFLEX",
            "reference_coordinates": {"x": 340.2, "y": 285.4},
            "eyelid_margin_point": {"x": 340.2, "y": 206.9},
            "status": "VALID",
            "confidence_score": 0.96,
            "warning_flags": ["PROCESSED_VIA_PERIOCULAR_PIPELINE"],
            "reflex_visibility": "VISIBLE",
        },
        "left_eye_os": {
            "mrd1_px": 75.0,
            "mrd1_mm": 3.54,
            "mrd1_estimated_range": "3.0 - 4.0 mm",
            "pfh_px": 208.5,
            "pfh_mm": 9.84,
            "reference_point_used": "CORNEAL_LIGHT_REFLEX",
            "reference_coordinates": {"x": 860.5, "y": 284.0},
            "eyelid_margin_point": {"x": 860.5, "y": 209.0},
            "status": "VALID",
            "confidence_score": 0.95,
            "warning_flags": ["PROCESSED_VIA_PERIOCULAR_PIPELINE"],
            "reflex_visibility": "VISIBLE",
        },
    },
    "bilateral_comparison": {
        "delta_mrd1_mm": 0.21,
        "signed_delta_mrd1_mm": -0.21,
        "asymmetry_classification": "PHYSIOLOGICAL_SYMMETRY",
        "delta_pfh_mm": 0.29,
        "delta_brow_px": 0.0,
        "eyebrow_height_discrepancy_px": 0.0,
    },
    "triage_result": {
        "tier": "TIER_A",
        "trigger_reasons": ["PHYSIOLOGICAL_NORMAL"],
        "description_vi": "Bình thường.",
        "recommendation_vi": "Kiểm tra định kỳ.",
        "safety_disclaimer_vi": "Chỉ mang tính sàng lọc.",
        "is_bilateral_severe_triggered": False,
    },
    "annotated_overlay_coordinates": {
        "right_eye": {
            "corneal_reflex": [340.2, 285.4],
            "pupil_center": [340.0, 285.0],
            "iris_limbus_contour": [[217.7, 285.4], [462.7, 285.4]],
            "upper_eyelid_contour": [[220.0, 240.0], [340.2, 206.9], [460.0, 245.0]],
        },
        "left_eye": {
            "corneal_reflex": [860.5, 284.0],
            "pupil_center": [860.2, 283.8],
            "iris_limbus_contour": [[736.5, 284.0], [984.5, 284.0]],
            "upper_eyelid_contour": [[740.0, 242.0], [860.5, 209.0], [980.0, 246.0]],
        },
    },
}

PAYLOAD_5_CROP_LOW_QUALITY = {
    "session_id": "d7c88990-1122-3344-5566-778899aabbcc",
    "timestamp": "2026-10-26T07:50:00.000Z",
    "image_quality_audit": {
        "passed": True,
        "rejection_reason": None,
        "pose": {"yaw": 0.0, "pitch": 0.0, "roll": 2.5},
        "sharpness_score": 62.0,
        "mean_illumination": 74.0,
    },
    "scale_reference": {
        "method": "HVID_WTW_POPULATION_AVERAGE",
        "assumed_corneal_diameter_mm": 11.71,
        "hvid_pixels": {"right_eye": 190.0, "left_eye": 192.0},
        "scale_factor_mm_per_px": {"right_eye": 0.0616, "left_eye": 0.0610},
    },
    "measurements": {
        "right_eye_od": {
            "mrd1_px": None,
            "mrd1_mm": None,
            "mrd1_estimated_range": None,
            "pfh_px": 110.0,
            "pfh_mm": 6.78,
            "reference_point_used": "NONE",
            "reference_coordinates": None,
            "eyelid_margin_point": {"x": 280.0, "y": 210.0},
            "status": "MEASUREMENT_UNAVAILABLE",
            "confidence_score": 0.0,
            "warning_flags": ["REFLEX_NOT_VISIBLE_QUALITY", "IMAGE_UNDEREXPOSED_OR_DIFFUSE"],
            "reflex_visibility": "NOT_VISIBLE_QUALITY",
        },
        "left_eye_os": {
            "mrd1_px": None,
            "mrd1_mm": None,
            "mrd1_estimated_range": None,
            "pfh_px": 112.5,
            "pfh_mm": 6.86,
            "reference_point_used": "NONE",
            "reference_coordinates": None,
            "eyelid_margin_point": {"x": 620.0, "y": 208.5},
            "status": "MEASUREMENT_UNAVAILABLE",
            "confidence_score": 0.0,
            "warning_flags": ["REFLEX_NOT_VISIBLE_QUALITY", "IMAGE_UNDEREXPOSED_OR_DIFFUSE"],
            "reflex_visibility": "NOT_VISIBLE_QUALITY",
        },
    },
    "bilateral_comparison": {
        "delta_mrd1_mm": None,
        "signed_delta_mrd1_mm": None,
        "asymmetry_classification": "BORDERLINE_ASYMMETRY",
        "delta_pfh_mm": 0.08,
        "delta_brow_px": 0.0,
        "eyebrow_height_discrepancy_px": 0.0,
    },
    "triage_result": {
        "tier": "TIER_B",
        "trigger_reasons": ["REFLEX_NOT_VISIBLE_INDETERMINATE_QUALITY"],
        "description_vi": "Thiếu sáng.",
        "recommendation_vi": "Chụp lại có flash.",
        "safety_disclaimer_vi": "Chỉ mang tính sàng lọc.",
        "is_bilateral_severe_triggered": False,
    },
    "annotated_overlay_coordinates": {
        "right_eye": {
            "corneal_reflex": None,
            "pupil_center": [280.0, 245.0],
            "iris_limbus_contour": [[185.0, 245.0], [375.0, 245.0]],
            "upper_eyelid_contour": [[190.0, 230.0], [280.0, 210.0], [370.0, 232.0]],
        },
        "left_eye": {
            "corneal_reflex": None,
            "pupil_center": [620.0, 244.0],
            "iris_limbus_contour": [[524.0, 244.0], [716.0, 244.0]],
            "upper_eyelid_contour": [[528.0, 229.0], [620.0, 208.5], [712.0, 231.0]],
        },
    },
}


# ==============================================================================
# KIỂM THỬ XÁC THỰC PYDANTIC CHO TỪNG PAYLOAD MẪU
# ==============================================================================

@pytest.mark.parametrize(
    "payload_dict,expected_tier,expected_od_mrd1,expected_od_status,expected_reflex_vis",
    [
        (PAYLOAD_1_FULLFACE_NORMAL, "TIER_A", 3.5, MeasurementStatusEnum.VALID, ReflexVisibilityEnum.VISIBLE),
        (PAYLOAD_2_FALLBACK_NO_FLASH, "TIER_B", 2.93, MeasurementStatusEnum.LOW_CONFIDENCE, ReflexVisibilityEnum.NOT_VISIBLE_QUALITY),
        (PAYLOAD_3_OCCLUDED_SEVERE, "TIER_C", None, MeasurementStatusEnum.MEASUREMENT_UNAVAILABLE, ReflexVisibilityEnum.OCCLUDED_BY_LID),
        (PAYLOAD_4_DUAL_EYE_CROP, "TIER_A", 3.75, MeasurementStatusEnum.VALID, ReflexVisibilityEnum.VISIBLE),
        (PAYLOAD_5_CROP_LOW_QUALITY, "TIER_B", None, MeasurementStatusEnum.MEASUREMENT_UNAVAILABLE, ReflexVisibilityEnum.NOT_VISIBLE_QUALITY),
    ],
)
def test_mock_payload_pydantic_validation(
    payload_dict, expected_tier, expected_od_mrd1, expected_od_status, expected_reflex_vis
):
    """Kiểm tra Pydantic deserialize 100% thành công không có ngoại lệ validation."""
    model = MeasurementPayload.model_validate(payload_dict)

    assert model.session_id == payload_dict["session_id"]
    assert model.triage_result.tier == expected_tier
    assert model.measurements.right_eye_od.mrd1_mm == expected_od_mrd1
    assert model.measurements.right_eye_od.status == expected_od_status
    assert model.measurements.right_eye_od.reflex_visibility == expected_reflex_vis

    # Kiểm tra vòng đời Re-serialization -> Deserialization
    dumped = model.model_dump()
    json_str = json.dumps(dumped)
    loaded = json.loads(json_str)
    revalidated = MeasurementPayload.model_validate(loaded)
    assert revalidated.session_id == model.session_id


def test_nullable_mrd1_fields_and_null_reflex():
    """Kiểm tra cấu trúc chấp nhận mrd1_mm = None, corneal_reflex = None mà không crash."""
    meas = SingleEyeMeasurement(
        mrd1_px=None,
        mrd1_mm=None,
        mrd1_estimated_range=None,
        status=MeasurementStatusEnum.MEASUREMENT_UNAVAILABLE,
        confidence_score=0.0,
        warning_flags=["ANATOMICAL_NEGATIVE_SUSPECTED"],
        reflex_visibility=ReflexVisibilityEnum.OCCLUDED_BY_LID,
    )
    assert meas.mrd1_mm is None
    assert meas.status == "MEASUREMENT_UNAVAILABLE"
    assert meas.reflex_visibility == "OCCLUDED_BY_LID"

    overlay = EyeAnnotationPoints(
        corneal_reflex=None,
        pupil_center=None,
        iris_limbus_contour=[],
        upper_eyelid_contour=[],
    )
    assert overlay.corneal_reflex is None
    assert overlay.pupil_center is None
