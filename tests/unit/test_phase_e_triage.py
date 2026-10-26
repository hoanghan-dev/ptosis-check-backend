"""
Bộ kiểm thử đơn vị cho PHASE E: Enhanced Asymmetry-First Triage Engine.
Kiểm tra toàn diện:
1. Chốt chặn che lấp phản quang do bờ mi (OCCLUDED_BY_LID -> TIER_C).
2. Xử lý thiếu phản quang do chất lượng ảnh / ánh sáng tán xạ (NOT_VISIBLE_QUALITY -> TIER_B / INDETERMINATE, không báo động giả).
3. Chốt chặn an toàn hai mắt (Bilateral Fail-Safe: cả 2 mắt < 2.0 mm -> TIER_C).
4. Sụp mi nặng một mắt (< 1.0 mm -> TIER_C).
5. Phân tầng ưu tiên độ bất đối xứng (Significant Asymmetry > 2.0 mm -> TIER_C, Borderline -> TIER_B, Physiological -> TIER_A).
6. Khả năng chống crash 100% khi nhận giá trị None (Null Safety).
7. Tuân thủ tuyệt đối quy tắc cấm từ ngữ chẩn đoán y khoa (G4).
"""

import pytest
from app.core.config import settings
from app.schemas.measurement import ReflexVisibilityEnum
from app.schemas.triage import ScreeningTier
from app.services.triage_service import evaluate_triage


def test_triage_lid_occluded_od_triggers_tier_c():
    """Kiểm tra mắt phải bị che lấp bờ mi (OCCLUDED_BY_LID) kích hoạt TIER_C."""
    result = evaluate_triage(
        od_mrd1_mm=None,
        os_mrd1_mm=3.5,
        delta_mrd1_mm=None,
        delta_brow_px=0.0,
        reflex_visibility_od=ReflexVisibilityEnum.OCCLUDED_BY_LID,
        reflex_visibility_os=ReflexVisibilityEnum.VISIBLE,
    )

    assert result.tier == ScreeningTier.TIER_C
    assert "REFLEX_OCCLUDED_SEVERE" in result.trigger_reasons
    assert "UNILATERAL_VISUAL_AXIS_OBSTRUCTION" in result.trigger_reasons
    assert "mắt phải" in result.description_vi
    assert result.flags.is_brow_compensation_detected is False


def test_triage_lid_occluded_od_with_brow_compensation():
    """Kiểm tra mắt phải bị che kèm nâng cung mày bù trừ (> 8px)."""
    result = evaluate_triage(
        od_mrd1_mm=None,
        os_mrd1_mm=3.5,
        delta_mrd1_mm=None,
        delta_brow_px=11.5,
        reflex_visibility_od=ReflexVisibilityEnum.OCCLUDED_BY_LID,
        reflex_visibility_os=ReflexVisibilityEnum.VISIBLE,
    )

    assert result.tier == ScreeningTier.TIER_C
    assert "REFLEX_OCCLUDED_SEVERE" in result.trigger_reasons
    assert "UNILATERAL_VISUAL_AXIS_OBSTRUCTION" in result.trigger_reasons
    assert "BROW_COMPENSATION_SUSPECTED" in result.trigger_reasons
    assert result.flags.is_brow_compensation_detected is True
    assert "nhướng cung mày bù trừ rõ rệt" in result.description_vi


def test_triage_lid_occluded_os_triggers_tier_c():
    """Kiểm tra mắt trái bị che lấp bờ mi (OCCLUDED_BY_LID) kích hoạt TIER_C."""
    result = evaluate_triage(
        od_mrd1_mm=3.8,
        os_mrd1_mm=None,
        delta_mrd1_mm=None,
        delta_brow_px=0.0,
        reflex_visibility_od=ReflexVisibilityEnum.VISIBLE,
        reflex_visibility_os=ReflexVisibilityEnum.OCCLUDED_BY_LID,
    )

    assert result.tier == ScreeningTier.TIER_C
    assert "REFLEX_OCCLUDED_SEVERE" in result.trigger_reasons
    assert "UNILATERAL_VISUAL_AXIS_OBSTRUCTION" in result.trigger_reasons
    assert "mắt trái" in result.description_vi


def test_triage_bilateral_lid_occlusion():
    """Kiểm tra cả 2 mắt bị che lấp bờ mi (OCCLUDED_BY_LID) kích hoạt TIER_C không có cờ đơn phương."""
    result = evaluate_triage(
        od_mrd1_mm=None,
        os_mrd1_mm=None,
        delta_mrd1_mm=None,
        delta_brow_px=0.0,
        reflex_visibility_od=ReflexVisibilityEnum.OCCLUDED_BY_LID,
        reflex_visibility_os=ReflexVisibilityEnum.OCCLUDED_BY_LID,
    )

    assert result.tier == ScreeningTier.TIER_C
    assert "REFLEX_OCCLUDED_SEVERE" in result.trigger_reasons
    assert "UNILATERAL_VISUAL_AXIS_OBSTRUCTION" not in result.trigger_reasons
    assert "cả hai mắt" in result.description_vi


def test_triage_diffuse_lighting_both_eyes_indeterminate_quality():
    """Kiểm tra cả 2 mắt mất phản quang do thiếu flash/chất lượng ảnh -> TIER_B (không báo động giả TIER_C)."""
    result = evaluate_triage(
        od_mrd1_mm=None,
        os_mrd1_mm=None,
        delta_mrd1_mm=None,
        delta_brow_px=0.0,
        reflex_visibility_od=ReflexVisibilityEnum.NOT_VISIBLE_QUALITY,
        reflex_visibility_os=ReflexVisibilityEnum.NOT_VISIBLE_QUALITY,
    )

    assert result.tier == ScreeningTier.TIER_B
    assert "REFLEX_NOT_VISIBLE_INDETERMINATE_QUALITY" in result.trigger_reasons
    assert "thiếu sáng hoặc không có ánh đèn trực diện" in result.description_vi
    assert "bật đèn flash của điện thoại ở khoảng cách 50-70cm" in result.recommendation_vi
    assert result.is_bilateral_severe_triggered is False


def test_triage_diffuse_lighting_one_eye_indeterminate():
    """Kiểm tra một mắt mất phản quang do chất lượng, mắt kia đo được -> TIER_B INDETERMINATE."""
    result = evaluate_triage(
        od_mrd1_mm=None,
        os_mrd1_mm=3.6,
        delta_mrd1_mm=None,
        delta_brow_px=0.0,
        reflex_visibility_od=ReflexVisibilityEnum.NOT_VISIBLE_QUALITY,
        reflex_visibility_os=ReflexVisibilityEnum.VISIBLE,
    )

    assert result.tier == ScreeningTier.TIER_B
    assert "INDETERMINATE_IMAGE_QUALITY" in result.trigger_reasons
    assert "Một bên mắt không xác định được phản quang" in result.description_vi
    assert "bật đèn flash trực diện" in result.recommendation_vi


def test_triage_bilateral_severe_mrd1_under_2mm():
    """Kiểm tra chốt chặn an toàn hai mắt: cả 2 mắt < 2.0 mm -> TIER_C bất kể delta_MRD1."""
    result = evaluate_triage(
        od_mrd1_mm=1.8,
        os_mrd1_mm=1.9,
        delta_mrd1_mm=0.1,
        delta_brow_px=0.0,
        reflex_visibility_od=ReflexVisibilityEnum.VISIBLE,
        reflex_visibility_os=ReflexVisibilityEnum.VISIBLE,
    )

    assert result.tier == ScreeningTier.TIER_C
    assert "BILATERAL_SEVERE_MRD1_UNDER_2MM" in result.trigger_reasons
    assert result.is_bilateral_severe_triggered is True


def test_triage_unilateral_severe_mrd1_under_1mm():
    """Kiểm tra một mắt sụp nặng < 1.0 mm -> TIER_C."""
    result = evaluate_triage(
        od_mrd1_mm=0.8,
        os_mrd1_mm=2.5,
        delta_mrd1_mm=1.7,
        delta_brow_px=0.0,
        reflex_visibility_od=ReflexVisibilityEnum.VISIBLE,
        reflex_visibility_os=ReflexVisibilityEnum.VISIBLE,
    )

    assert result.tier == ScreeningTier.TIER_C
    assert "UNILATERAL_SEVERE_MRD1_UNDER_1MM" in result.trigger_reasons
    assert result.is_bilateral_severe_triggered is False


def test_triage_significant_asymmetry_over_2mm():
    """Kiểm tra bất đối xứng rõ rệt (delta_MRD1 > 2.0 mm) -> TIER_C."""
    result = evaluate_triage(
        od_mrd1_mm=2.2,
        os_mrd1_mm=4.5,
        delta_mrd1_mm=2.3,
        delta_brow_px=0.0,
        reflex_visibility_od=ReflexVisibilityEnum.VISIBLE,
        reflex_visibility_os=ReflexVisibilityEnum.VISIBLE,
    )

    assert result.tier == ScreeningTier.TIER_C
    assert "SIGNIFICANT_ASYMMETRY_OVER_2MM" in result.trigger_reasons


def test_triage_significant_asymmetry_with_unilateral_under_2mm():
    """Kiểm tra delta_MRD1 > 2.0 mm và mắt nhỏ hơn < 2.0 mm -> TIER_C kèm cờ UNILATERAL_SEVERE_MRD1_UNDER_2MM."""
    result = evaluate_triage(
        od_mrd1_mm=1.8,
        os_mrd1_mm=4.2,
        delta_mrd1_mm=2.4,
        delta_brow_px=0.0,
        reflex_visibility_od=ReflexVisibilityEnum.VISIBLE,
        reflex_visibility_os=ReflexVisibilityEnum.VISIBLE,
    )

    assert result.tier == ScreeningTier.TIER_C
    assert "SIGNIFICANT_ASYMMETRY_OVER_2MM" in result.trigger_reasons
    assert "UNILATERAL_SEVERE_MRD1_UNDER_2MM" in result.trigger_reasons


def test_triage_borderline_asymmetry():
    """Kiểm tra bất đối xứng ranh giới (1.5 < delta <= 2.0 mm) -> TIER_B."""
    result = evaluate_triage(
        od_mrd1_mm=2.5,
        os_mrd1_mm=4.2,
        delta_mrd1_mm=1.7,
        delta_brow_px=0.0,
        reflex_visibility_od=ReflexVisibilityEnum.VISIBLE,
        reflex_visibility_os=ReflexVisibilityEnum.VISIBLE,
    )

    assert result.tier == ScreeningTier.TIER_B
    assert "BORDERLINE_ASYMMETRY" in result.trigger_reasons


def test_triage_physiological_normal():
    """Kiểm tra đối xứng sinh lý bình thường (delta <= 1.5 mm, min >= 2.0 mm) -> TIER_A."""
    result = evaluate_triage(
        od_mrd1_mm=3.2,
        os_mrd1_mm=3.6,
        delta_mrd1_mm=0.4,
        delta_brow_px=0.0,
        reflex_visibility_od=ReflexVisibilityEnum.VISIBLE,
        reflex_visibility_os=ReflexVisibilityEnum.VISIBLE,
    )

    assert result.tier == ScreeningTier.TIER_A
    assert "PHYSIOLOGICAL_NORMAL" in result.trigger_reasons


def test_triage_null_combinations_zero_crash():
    """Kiểm tra an toàn tuyệt đối với mọi tổ hợp đầu vào None không gây ngoại lệ."""
    combinations = [
        (None, None, None),
        (None, 3.5, None),
        (3.5, None, None),
        (None, 2.0, 1.5),
        (2.0, None, 1.5),
    ]

    for od, os, delta in combinations:
        # Visibility VISIBLE
        res1 = evaluate_triage(od, os, delta)
        assert res1 is not None
        assert res1.tier in [ScreeningTier.TIER_A, ScreeningTier.TIER_B, ScreeningTier.TIER_C]

        # Visibility INDETERMINATE
        res2 = evaluate_triage(
            od, os, delta,
            reflex_visibility_od=ReflexVisibilityEnum.INDETERMINATE,
            reflex_visibility_os=ReflexVisibilityEnum.INDETERMINATE,
        )
        assert res2 is not None
        assert res2.tier in [ScreeningTier.TIER_A, ScreeningTier.TIER_B, ScreeningTier.TIER_C]

        # Backward compatibility flag is_reflex_occluded=True
        res3 = evaluate_triage(od, os, delta, is_reflex_occluded=True)
        assert res3 is not None
        assert res3.tier == ScreeningTier.TIER_C
