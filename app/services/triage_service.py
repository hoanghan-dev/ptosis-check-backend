"""
Dịch vụ Phân tầng Nguy cơ Ưu tiên Bất đối xứng (Module M7: Asymmetry-First Triage Engine).
Triển khai cây quyết định lâm sàng 3 mức (TIER_A, TIER_B, TIER_C) dựa trên:
- Chốt chặn an toàn tuyệt đối: min(MRD1) < 2.0 mm hoặc phản quang bị che lấp -> Mức C.
- Tiêu chí ưu tiên bất đối xứng: delta_MRD1 > 1.5 mm -> Mức C.
- Tiêu chí ranh giới: 1.0 < delta_MRD1 <= 1.5 mm HOẶC 2.0 <= min(MRD1) < 2.5 mm HOẶC delta_Brow > 8.0 px -> Mức B.
- Sinh lý bình thường: min(MRD1) >= 2.5 mm VÀ delta_MRD1 <= 1.0 mm VÀ delta_Brow <= 8.0 px -> Mức A.

Tuân thủ tuyệt đối:
- G3 (Asymmetry-First & Bilateral Fail-Safe)
- G4 (Non-Diagnostic Safety: Tuyệt đối không dùng từ ngữ thuộc danh mục cấm)
"""

from typing import List, Optional
from app.core.config import settings
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
    od_mrd1_mm: float,
    os_mrd1_mm: float,
    delta_mrd1_mm: Optional[float] = None,
    delta_brow_px: float = 0.0,
    is_reflex_occluded: bool = False,
    is_pupil_fallback_used: bool = False,
) -> TriageResult:
    """
    Đánh giá phân tầng nguy cơ sàng lọc hình thái mi mắt theo cây quyết định 3 mức:
    Mức A (Không phát hiện bất thường), Mức B (Dấu hiệu nghi ngờ), Mức C (Khuyến nghị khám chuyên khoa).

    Args:
        od_mrd1_mm: Chỉ số MRD1 mắt phải (milimét).
        os_mrd1_mm: Chỉ số MRD1 mắt trái (milimét).
        delta_mrd1_mm: Độ chênh lệch tuyệt đối |OS - OD| (mm). Tự tính nếu là None.
        delta_brow_px: Độ chênh lệch độ cao cung mày hai bên (pixels).
        is_reflex_occluded: Cờ báo bờ mi che lấp hoàn toàn phản quang/trục thị giác.
        is_pupil_fallback_used: Cờ báo hệ thống đã phải dự phòng sang tâm đồng tử.

    Returns:
        TriageResult: Đối tượng chứa mức phân tầng, mã lý do kích hoạt, mô tả, khuyến nghị và cờ an toàn.
    """
    if delta_mrd1_mm is None:
        delta_mrd1_mm = abs(float(os_mrd1_mm) - float(od_mrd1_mm))

    delta_mrd1_mm = round(float(delta_mrd1_mm), 2)
    od_val = float(od_mrd1_mm)
    os_val = float(os_mrd1_mm)
    min_mrd1 = min(od_val, os_val)

    trigger_reasons: List[str] = []
    is_bilateral_severe = False
    is_brow_compensation = False

    # =========================================================================
    # BƯỚC 1: KIỂM TRA CHỐT CHẶN AN TOÀN VÀ TIÊU CHÍ MỨC C (TIER_C)
    # =========================================================================
    # Chốt chặn 1.1: Phản quang bị che lấp hoàn toàn (bờ mi che lấp trục nhìn)
    if is_reflex_occluded:
        trigger_reasons.append("REFLEX_OCCLUDED_SEVERE")

    # Chốt chặn 1.2: min(MRD1) < ABSOLUTE_SEVERE_MRD1_MM (2.0 mm)
    if min_mrd1 < settings.ABSOLUTE_SEVERE_MRD1_MM:
        if od_val < settings.ABSOLUTE_SEVERE_MRD1_MM and os_val < settings.ABSOLUTE_SEVERE_MRD1_MM:
            is_bilateral_severe = True
            trigger_reasons.append("BILATERAL_SEVERE_MRD1_UNDER_2MM")
        else:
            trigger_reasons.append("UNILATERAL_SEVERE_MRD1_UNDER_2MM")

    # Tiêu chí 1.3: Bất đối xứng rõ rệt vượt ngưỡng ranh giới (> 1.5 mm)
    if delta_mrd1_mm > settings.BORDERLINE_ASYMMETRY_MAX_MM:
        trigger_reasons.append("SIGNIFICANT_ASYMMETRY_OVER_1_5MM")

    # Nếu thỏa mãn bất kỳ điều kiện nào ở trên -> MỨC C
    if trigger_reasons:
        flags = TriageFlags(
            is_bilateral_severe_triggered=is_bilateral_severe,
            is_brow_compensation_detected=False,
            is_pupil_fallback_used=is_pupil_fallback_used,
        )
        return TriageResult(
            tier=ScreeningTier.TIER_C,
            trigger_reasons=trigger_reasons,
            description_vi=DESCRIPTION_TIER_C,
            recommendation_vi=RECOMMENDATION_TIER_C,
            safety_disclaimer_vi=SAFETY_DISCLAIMER_VI,
            is_bilateral_severe_triggered=is_bilateral_severe,
            flags=flags,
        )

    # =========================================================================
    # BƯỚC 2: KIỂM TRA TIÊU CHÍ RANH GIỚI / NGHI NGỜ (TIER_B)
    # =========================================================================
    # Tiêu chí 2.1: Bất đối xứng ranh giới (1.0 mm < delta_mrd1 <= 1.5 mm)
    if delta_mrd1_mm > settings.PHYSIOLOGICAL_ASYMMETRY_MAX_MM:
        trigger_reasons.append("BORDERLINE_ASYMMETRY")

    # Tiêu chí 2.2: MRD1 một trong hai mắt nằm trong khoảng ranh giới [2.0 mm, 2.5 mm)
    if min_mrd1 < settings.TIER_A_MIN_MRD1_MM:
        trigger_reasons.append("BORDERLINE_LOW_MRD1")

    # Tiêu chí 2.3: Bất đối xứng cung mày vượt ngưỡng (chỉ báo co cơ trán nâng mi bù trừ)
    if delta_brow_px > settings.BROW_ASYMMETRY_THRESHOLD_PX:
        is_brow_compensation = True
        trigger_reasons.append("BROW_COMPENSATION_SUSPECTED")

    if trigger_reasons:
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
    # BƯỚC 3: SINH LÝ BÌNH THƯỜNG (TIER_A)
    # =========================================================================
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
