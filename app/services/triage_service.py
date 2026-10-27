"""
Dịch vụ Phân tầng Nguy cơ Ưu tiên Bất đối xứng (Module M7: Enhanced Asymmetry-First Triage Engine).
Triển khai cây quyết định lâm sàng 3 mức (TIER_A, TIER_B, TIER_C) dựa trên:
- Ngoại lệ chốt chặn an toàn (Fail-Safe Overrides):
  + Bờ mi che lấp trục thị giác / che lấp phản quang (OCCLUDED_BY_LID hoặc MRD1 <= 0) -> Mức C.
  + Sụp mi nặng che lấp trục nhìn cực hạn ở một hoặc hai mắt: min(MRD1) < 1.0 mm -> Mức C.
  + Thiếu phản quang do chất lượng ảnh / thiếu flash (NOT_VISIBLE_QUALITY / INDETERMINATE):
    Không tự động gán TIER_C (tránh báo động giả per Risk 4 / Payload 5) -> Mức B (khuyến nghị chụp lại có flash).
- Căn cứ ưu tiên cao nhất (Bất đối xứng hai bên delta_MRD1):
  + Tiêu chí ưu tiên: delta_MRD1 > 2.0 mm -> Mức C (lúc này mới xét chi tiết MRD1 2 mắt để phân định mức độ).
  + Tiêu chí ranh giới: 1.5 mm < delta_MRD1 <= 2.0 mm HOẶC delta_Brow > 8.0 px -> Mức B.
  + Sinh lý bình thường: delta_MRD1 <= 1.5 mm VÀ min(MRD1) >= 1.0 mm -> Mức A (hoặc Mức B nếu nhướng mày bù trừ).

Tuân thủ tuyệt đối:
- G3 (Asymmetry-First Priority & Clinical Fail-Safe)
- G4 (Non-Diagnostic Safety: Tuyệt đối không dùng từ ngữ thuộc danh mục cấm)
"""

from typing import List, Optional
from app.core.config import settings
from app.schemas.measurement import ReflexVisibilityEnum
from app.schemas.triage import ScreeningTier, TriageFlags, TriageResult

# Văn bản giải thích và khuyến nghị chuẩn đại chúng (Tuân thủ tuyệt đối G4)
DESCRIPTION_TIER_A = (
    "Không phát hiện dấu hiệu bất thường về độ mở mi hoặc độ bất đối xứng hai mắt. "
    "Các chỉ số nằm trong giới hạn bình thường."
)
RECOMMENDATION_TIER_A = (
    "Hiện tại chưa ghi nhận dấu hiệu cần can thiệp. "
    "Bạn nên tiếp tục duy trì thói quen sinh hoạt lành mạnh và kiểm tra mắt định kỳ."
)

DESCRIPTION_TIER_B = (
    "Ghi nhận độ cao mi ở mức ranh giới hoặc có sự chênh lệch nhẹ giữa hai mắt, "
    "hoặc có dấu hiệu nâng cung mày bù trừ."
)
RECOMMENDATION_TIER_B = (
    "Khuyến nghị bạn theo dõi thêm và cân nhắc thăm khám chuyên khoa mắt khi có dịp "
    "để kiểm tra tình trạng mi mắt chi tiết hơn."
)

DESCRIPTION_TIER_C = (
    "Ghi nhận bờ mi trên che phủ đáng kể trục thị giác hoặc có sự bất đối xứng rõ rệt giữa hai mắt."
)
RECOMMENDATION_TIER_C = (
    "Khuyến nghị bạn sớm đến cơ sở y tế chuyên khoa mắt để được thăm khám trực tiếp và tư vấn phù hợp."
)

SAFETY_DISCLAIMER_VI = (
    "Kết quả này chỉ mang tính sàng lọc hình thái ban đầu, không thay thế "
    "thăm khám y khoa chuyên sâu. Vui lòng tham vấn chuyên gia hoặc bác sĩ "
    "chuyên khoa mắt khi có băn khoăn."
)


def evaluate_triage(
    od_mrd1_mm: Optional[float],
    os_mrd1_mm: Optional[float],
    delta_mrd1_mm: Optional[float] = None,
    delta_brow_px: float = 0.0,
    reflex_visibility_od: ReflexVisibilityEnum = ReflexVisibilityEnum.VISIBLE,
    reflex_visibility_os: ReflexVisibilityEnum = ReflexVisibilityEnum.VISIBLE,
    is_pupil_fallback_used: bool = False,
    is_reflex_occluded: Optional[bool] = None,
) -> TriageResult:
    """
    Đánh giá phân tầng nguy cơ sàng lọc hình thái mi mắt theo cây quyết định 3 mức:
    Mức A (Không phát hiện bất thường), Mức B (Dấu hiệu nghi ngờ / Cần chụp lại), Mức C (Khuyến nghị khám chuyên khoa).

    Args:
        od_mrd1_mm: Chỉ số MRD1 mắt phải (milimét hoặc None nếu không khả dụng).
        os_mrd1_mm: Chỉ số MRD1 mắt trái (milimét hoặc None nếu không khả dụng).
        delta_mrd1_mm: Độ chênh lệch tuyệt đối |OS - OD| (mm). Tự tính nếu là None.
        delta_brow_px: Độ chênh lệch độ cao cung mày hai bên (pixels).
        reflex_visibility_od: Trạng thái hiển thị phản quang mắt phải.
        reflex_visibility_os: Trạng thái hiển thị phản quang mắt trái.
        is_pupil_fallback_used: Cờ báo hệ thống đã phải dự phòng sang tâm đồng tử.
        is_reflex_occluded: Cờ báo bờ mi che lấp hoàn toàn phản quang (tương thích ngược).

    Returns:
        TriageResult: Đối tượng chứa mức phân tầng, mã lý do kích hoạt, mô tả, khuyến nghị và cờ an toàn.
    """
    od_val = float(od_mrd1_mm) if od_mrd1_mm is not None else None
    os_val = float(os_mrd1_mm) if os_mrd1_mm is not None else None

    # Tính toán an toàn độ chênh lệch delta_mrd1_mm
    if delta_mrd1_mm is None:
        if od_val is not None and os_val is not None:
            delta_mrd1_mm = round(abs(os_val - od_val), 2)
        else:
            delta_mrd1_mm = None
    else:
        delta_mrd1_mm = round(float(delta_mrd1_mm), 2)

    min_mrd1 = None
    if od_val is not None and os_val is not None:
        min_mrd1 = min(od_val, os_val)
    elif od_val is not None:
        min_mrd1 = od_val
    elif os_val is not None:
        min_mrd1 = os_val

    # Xác định bằng chứng giải phẫu che lấp bờ mi (OCCLUDED_BY_LID)
    is_lid_occluded_od = reflex_visibility_od == ReflexVisibilityEnum.OCCLUDED_BY_LID
    is_lid_occluded_os = reflex_visibility_os == ReflexVisibilityEnum.OCCLUDED_BY_LID

    # Tương thích ngược: nếu cờ is_reflex_occluded được truyền vào là True
    if is_reflex_occluded is True:
        if not is_lid_occluded_od and not is_lid_occluded_os:
            if od_val is None and os_val is not None:
                is_lid_occluded_od = True
            elif os_val is None and od_val is not None:
                is_lid_occluded_os = True
            else:
                is_lid_occluded_od = True

    # Ngoại lệ lập tức để mức C: bờ mi che lấp reflex hoặc MRD1 âm / bằng 0
    is_any_lid_occluded = is_lid_occluded_od or is_lid_occluded_os or (min_mrd1 is not None and min_mrd1 <= 0.0)

    # =========================================================================
    # GIAI ĐOẠN 0: NGOẠI LỆ CHỐT CHẶN AN TOÀN TUYỆT ĐỐI (FAIL-SAFE OVERRIDE)
    # =========================================================================

    # -------------------------------------------------------------------------
    # Ngoại lệ 0.1: Bờ mi che lấp hoàn toàn phản quang / trục thị giác (MRD1 <= 0 hoặc OCCLUDED) -> MỨC C
    # -------------------------------------------------------------------------
    if is_any_lid_occluded:
        trigger_reasons = ["REFLEX_OCCLUDED_SEVERE"]
        is_brow_compensation = delta_brow_px > settings.BROW_ASYMMETRY_THRESHOLD_PX

        if is_lid_occluded_od != is_lid_occluded_os:
            trigger_reasons.append("UNILATERAL_VISUAL_AXIS_OBSTRUCTION")

        if is_brow_compensation:
            trigger_reasons.append("BROW_COMPENSATION_SUSPECTED")

        if is_lid_occluded_od and not is_lid_occluded_os:
            comp_suffix = ", kèm dấu hiệu nhướng cung mày bù trừ rõ rệt." if is_brow_compensation else "."
            desc = (
                f"Ghi nhận bờ mi trên mắt phải che phủ đáng kể qua trục thị giác khiến không thể xác định "
                f"phản quang giác mạc{comp_suffix}"
            )
        elif is_lid_occluded_os and not is_lid_occluded_od:
            comp_suffix = ", kèm dấu hiệu nhướng cung mày bù trừ rõ rệt." if is_brow_compensation else "."
            desc = (
                f"Ghi nhận bờ mi trên mắt trái che phủ đáng kể qua trục thị giác khiến không thể xác định "
                f"phản quang giác mạc{comp_suffix}"
            )
        else:
            comp_suffix = " Kèm dấu hiệu nhướng cung mày bù trừ rõ rệt." if is_brow_compensation else ""
            desc = (
                f"Ghi nhận bờ mi trên che phủ đáng kể qua trục thị giác ở cả hai mắt khiến không thể xác định "
                f"phản quang giác mạc.{comp_suffix}"
            )

        recommendation = (
            "Khuyến nghị bạn sớm đến cơ sở y tế chuyên khoa mắt để được thăm khám trực tiếp "
            "và đánh giá chuyên sâu chức năng mở mi."
        )

        is_bilateral_severe = bool(
            (od_val is not None and os_val is not None and od_val <= 0.0 and os_val <= 0.0)
            or (is_lid_occluded_od and is_lid_occluded_os)
        )

        flags = TriageFlags(
            is_bilateral_severe_triggered=is_bilateral_severe,
            is_brow_compensation_detected=is_brow_compensation,
            is_pupil_fallback_used=is_pupil_fallback_used,
        )
        return TriageResult(
            tier=ScreeningTier.TIER_C,
            trigger_reasons=trigger_reasons,
            description_vi=desc,
            recommendation_vi=recommendation,
            safety_disclaimer_vi=SAFETY_DISCLAIMER_VI,
            is_bilateral_severe_triggered=is_bilateral_severe,
            flags=flags,
        )

    # -------------------------------------------------------------------------
    # Ngoại lệ 0.2: Mất phản quang do chất lượng chụp / thiếu flash (Risk 4 / Payload 5) -> MỨC B
    # -------------------------------------------------------------------------
    is_quality_od = (
        reflex_visibility_od in [ReflexVisibilityEnum.NOT_VISIBLE_QUALITY, ReflexVisibilityEnum.INDETERMINATE]
        or od_val is None
    )
    is_quality_os = (
        reflex_visibility_os in [ReflexVisibilityEnum.NOT_VISIBLE_QUALITY, ReflexVisibilityEnum.INDETERMINATE]
        or os_val is None
    )

    if is_quality_od and is_quality_os:
        desc = (
            "Không phát hiện điểm phản xạ ánh sáng trên giác mạc ở cả hai mắt do thiếu sáng hoặc "
            "không có ánh đèn trực diện, bờ mi vẫn ở vị trí bình thường."
        )
        rec = (
            "Vui lòng chụp lại ảnh có bật đèn flash của điện thoại ở khoảng cách 50-70cm để "
            "thuật toán đo đạc chính xác phản quang."
        )
        flags = TriageFlags(
            is_bilateral_severe_triggered=False,
            is_brow_compensation_detected=False,
            is_pupil_fallback_used=is_pupil_fallback_used,
        )
        return TriageResult(
            tier=ScreeningTier.TIER_B,
            trigger_reasons=["REFLEX_NOT_VISIBLE_INDETERMINATE_QUALITY"],
            description_vi=desc,
            recommendation_vi=rec,
            safety_disclaimer_vi=SAFETY_DISCLAIMER_VI,
            is_bilateral_severe_triggered=False,
            flags=flags,
        )

    if is_quality_od or is_quality_os:
        desc = (
            "Một bên mắt không xác định được phản quang rõ ràng do góc sáng hoặc thiếu tương phản. "
            "Mắt còn lại ở giới hạn ghi nhận được."
        )
        rec = (
            "Vui lòng chụp lại ảnh có bật đèn flash trực diện để hệ thống đo đạc chính xác cả hai bên mắt."
        )
        flags = TriageFlags(
            is_bilateral_severe_triggered=False,
            is_brow_compensation_detected=False,
            is_pupil_fallback_used=is_pupil_fallback_used,
        )
        return TriageResult(
            tier=ScreeningTier.TIER_B,
            trigger_reasons=["INDETERMINATE_IMAGE_QUALITY"],
            description_vi=desc,
            recommendation_vi=rec,
            safety_disclaimer_vi=SAFETY_DISCLAIMER_VI,
            is_bilateral_severe_triggered=False,
            flags=flags,
        )

    # -------------------------------------------------------------------------
    # Ngoại lệ 0.3: Sụp mi nặng che lấp trục nhìn cực hạn (< 1.0 mm hoặc cả 2 mắt < 1.5 mm) -> MỨC C
    # -------------------------------------------------------------------------
    is_bilateral_severe = bool(
        od_val is not None
        and os_val is not None
        and (
            (od_val < 1.0 and os_val < 1.0)
            or (od_val < 1.5 and os_val < 1.5 and delta_mrd1_mm is not None and delta_mrd1_mm <= 1.5)
        )
    )

    if is_bilateral_severe:
        reason = "BILATERAL_SEVERE_MRD1_UNDER_1MM" if (od_val < 1.0 and os_val < 1.0) else "BILATERAL_SEVERE_MRD1_UNDER_2MM"
        flags = TriageFlags(
            is_bilateral_severe_triggered=True,
            is_brow_compensation_detected=False,
            is_pupil_fallback_used=is_pupil_fallback_used,
        )
        return TriageResult(
            tier=ScreeningTier.TIER_C,
            trigger_reasons=[reason],
            description_vi=DESCRIPTION_TIER_C,
            recommendation_vi=RECOMMENDATION_TIER_C,
            safety_disclaimer_vi=SAFETY_DISCLAIMER_VI,
            is_bilateral_severe_triggered=True,
            flags=flags,
        )

    if min_mrd1 is not None and min_mrd1 < 1.0:
        flags = TriageFlags(
            is_bilateral_severe_triggered=False,
            is_brow_compensation_detected=False,
            is_pupil_fallback_used=is_pupil_fallback_used,
        )
        return TriageResult(
            tier=ScreeningTier.TIER_C,
            trigger_reasons=["UNILATERAL_SEVERE_MRD1_UNDER_1MM"],
            description_vi=DESCRIPTION_TIER_C,
            recommendation_vi=RECOMMENDATION_TIER_C,
            safety_disclaimer_vi=SAFETY_DISCLAIMER_VI,
            is_bilateral_severe_triggered=False,
            flags=flags,
        )

    # =========================================================================
    # GIAI ĐOẠN 1: ĐÁNH GIÁ CĂN CỨ ƯU TIÊN CAO NHẤT - BẤT ĐỐI XỨNG HAI BÊN (ΔMRD1)
    # =========================================================================

    # -------------------------------------------------------------------------
    # Nhánh 1.1: BẤT ĐỐI XỨNG RÕ RỆT (delta_MRD1 > 2.0 mm) -> MỨC C
    # Lúc này MỚI XÉT đến chỉ số MRD1 của 2 mắt để phân định mức độ
    # -------------------------------------------------------------------------
    if delta_mrd1_mm is not None and delta_mrd1_mm > settings.BORDERLINE_ASYMMETRY_MAX_MM:
        trigger_reasons = ["SIGNIFICANT_ASYMMETRY_OVER_2MM"]
        if min_mrd1 is not None and min_mrd1 < settings.ABSOLUTE_SEVERE_MRD1_MM:  # min(MRD1) < 2.0 mm
            trigger_reasons.append("UNILATERAL_SEVERE_MRD1_UNDER_2MM")

        is_brow_compensation = delta_brow_px > settings.BROW_ASYMMETRY_THRESHOLD_PX
        if is_brow_compensation:
            trigger_reasons.append("BROW_COMPENSATION_SUSPECTED")

        flags = TriageFlags(
            is_bilateral_severe_triggered=False,
            is_brow_compensation_detected=is_brow_compensation,
            is_pupil_fallback_used=is_pupil_fallback_used,
        )
        return TriageResult(
            tier=ScreeningTier.TIER_C,
            trigger_reasons=trigger_reasons,
            description_vi=DESCRIPTION_TIER_C,
            recommendation_vi=RECOMMENDATION_TIER_C,
            safety_disclaimer_vi=SAFETY_DISCLAIMER_VI,
            is_bilateral_severe_triggered=False,
            flags=flags,
        )

    # -------------------------------------------------------------------------
    # Nhánh 1.2: BẤT ĐỐI XỨNG RANH GIỚI (1.5 mm < delta_MRD1 <= 2.0 mm) -> MỨC B
    # -------------------------------------------------------------------------
    if delta_mrd1_mm is not None and delta_mrd1_mm > settings.PHYSIOLOGICAL_ASYMMETRY_MAX_MM:
        trigger_reasons = ["BORDERLINE_ASYMMETRY"]
        is_brow_compensation = delta_brow_px > settings.BROW_ASYMMETRY_THRESHOLD_PX
        if is_brow_compensation:
            trigger_reasons.append("BROW_COMPENSATION_SUSPECTED")

        flags = TriageFlags(
            is_bilateral_severe_triggered=False,
            is_brow_compensation_detected=is_brow_compensation,
            is_pupil_fallback_used=is_pupil_fallback_used,
        )
        return TriageResult(
            tier=ScreeningTier.TIER_B,
            trigger_reasons=trigger_reasons,
            description_vi=DESCRIPTION_TIER_B,
            recommendation_vi=RECOMMENDATION_TIER_B,
            safety_disclaimer_vi=SAFETY_DISCLAIMER_VI,
            is_bilateral_severe_triggered=False,
            flags=flags,
        )

    # =========================================================================
    # GIAI ĐOẠN 2: ĐỐI XỨNG SINH LÝ (delta_MRD1 <= 1.5 mm VÀ min(MRD1) >= 1.0 mm)
    # Ưu tiên cao nhất: Khi hai mắt đối xứng sinh lý (như trường hợp delta=0.26mm, OD=1.49, OS=1.75 ở Ảnh 3) -> MỨC A.
    # Nếu có dấu hiệu sụp mi ranh giới (min(MRD1) < 2.0 mm) KÈM nhướng mày bù trừ rõ rệt (delta_brow > 8px)
    # -> Xếp vào MỨC B (BROW_COMPENSATION_SUSPECTED).
    # Ngược lại, nếu mi mở tốt (min >= 2.0 mm) hoặc không nhướng mày -> Luôn là MỨC A (PHYSIOLOGICAL_NORMAL).
    # =========================================================================
    # Sụp mi hai bên ranh giới (Borderline Bilateral Droop):
    # Cả hai mắt đều mở thấp (< 1.8 mm) với min(MRD1) < 1.6 mm dù đối xứng -> MỨC B (tránh bỏ sót)
    is_bilateral_borderline = bool(
        min_mrd1 is not None
        and min_mrd1 < 1.6
        and od_val is not None
        and od_val < 1.8
        and os_val is not None
        and os_val < 1.8
    )

    is_brow_comp_with_droop = (
        delta_brow_px > settings.BROW_ASYMMETRY_THRESHOLD_PX
        and min_mrd1 is not None
        and min_mrd1 < settings.ABSOLUTE_SEVERE_MRD1_MM
    )

    if is_bilateral_borderline or is_brow_comp_with_droop:
        reasons = []
        if is_bilateral_borderline:
            reasons.append("BORDERLINE_BILATERAL_LOW_MRD1")
        if is_brow_comp_with_droop:
            reasons.append("BROW_COMPENSATION_SUSPECTED")

        flags = TriageFlags(
            is_bilateral_severe_triggered=False,
            is_brow_compensation_detected=is_brow_comp_with_droop,
            is_pupil_fallback_used=is_pupil_fallback_used,
        )
        return TriageResult(
            tier=ScreeningTier.TIER_B,
            trigger_reasons=reasons,
            description_vi=DESCRIPTION_TIER_B,
            recommendation_vi=RECOMMENDATION_TIER_B,
            safety_disclaimer_vi=SAFETY_DISCLAIMER_VI,
            is_bilateral_severe_triggered=False,
            flags=flags,
        )

    flags = TriageFlags(
        is_bilateral_severe_triggered=False,
        is_brow_compensation_detected=False,
        is_pupil_fallback_used=is_pupil_fallback_used,
    )
    return TriageResult(
        tier=ScreeningTier.TIER_A,
        trigger_reasons=["PHYSIOLOGICAL_NORMAL"],
        description_vi=DESCRIPTION_TIER_A,
        recommendation_vi=RECOMMENDATION_TIER_A,
        safety_disclaimer_vi=SAFETY_DISCLAIMER_VI,
        is_bilateral_severe_triggered=False,
        flags=flags,
    )
