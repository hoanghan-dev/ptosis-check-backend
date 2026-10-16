"""
Dịch vụ Phân tầng Nguy cơ Ưu tiên Bất đối xứng (Module M7: Asymmetry-First Triage Engine).
Triển khai cây quyết định lâm sàng 3 mức (TIER_A, TIER_B, TIER_C) dựa trên:
- Ngoại lệ chốt chặn an toàn: min(MRD1) <= 0.0 mm (bất kể delta_MRD1) hoặc phản quang bị che lấp -> Mức C.
- Căn cứ ưu tiên cao nhất (Bất đối xứng hai bên delta_MRD1):
  + Tiêu chí ưu tiên: delta_MRD1 > 2.0 mm -> Mức C (lúc này mới xét chi tiết MRD1 2 mắt để phân định mức độ).
  + Tiêu chí ranh giới: 1.5 mm < delta_MRD1 <= 2.0 mm HOẶC delta_Brow > 8.0 px -> Mức B.
  + Sinh lý bình thường: delta_MRD1 <= 1.5 mm VÀ min(MRD1) > 0.0 mm VÀ delta_Brow <= 8.0 px -> Mức A.

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
    # GIAI ĐOẠN 0: NGOẠI LỆ CHỐT CHẶN AN TOÀN TUYỆT ĐỐI (FAIL-SAFE OVERRIDE)
    # =========================================================================
    # Ngoại lệ 0.1: Phản quang bị che lấp hoàn toàn (bờ mi che lấp trục nhìn)
    if is_reflex_occluded:
        trigger_reasons.append("REFLEX_OCCLUDED_SEVERE")

    # Ngoại lệ 0.2: min(MRD1) <= 0.0 mm (bất kể delta_MRD1)
    critical_threshold = getattr(settings, "CRITICAL_SEVERE_MRD1_MM", 0.0)
    if min_mrd1 <= critical_threshold:
        if od_val <= critical_threshold and os_val <= critical_threshold:
            is_bilateral_severe = True
            trigger_reasons.append("BILATERAL_SEVERE_MRD1_LE_0MM")
        else:
            trigger_reasons.append("UNILATERAL_SEVERE_MRD1_LE_0MM")

    # Nếu vi phạm bất kỳ ngoại lệ chốt chặn an toàn nào -> MỨC C ngay lập tức
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
    # GIAI ĐOẠN 1: ĐÁNH GIÁ CĂN CỨ ƯU TIÊN CAO NHẤT - BẤT ĐỐI XỨNG HAI BÊN (ΔMRD1)
    # =========================================================================
    # Nhánh 1.1: BẤT ĐỐI XỨNG RÕ RỆT (delta_MRD1 > 2.0 mm) -> MỨC C
    # Lúc này MỚI XÉT đến chỉ số MRD1 của 2 mắt để phân định mức độ
    if delta_mrd1_mm > settings.BORDERLINE_ASYMMETRY_MAX_MM:
        trigger_reasons.append("SIGNIFICANT_ASYMMETRY_OVER_2MM")
        if min_mrd1 < settings.ABSOLUTE_SEVERE_MRD1_MM:  # min(MRD1) < 2.0 mm
            trigger_reasons.append("UNILATERAL_SEVERE_MRD1_UNDER_2MM")

        flags = TriageFlags(
            is_bilateral_severe_triggered=False,
            is_brow_compensation_detected=False,
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

    # Nhánh 1.2: BẤT ĐỐI XỨNG RANH GIỚI (1.5 mm < delta_MRD1 <= 2.0 mm) -> MỨC B
    if delta_mrd1_mm > settings.PHYSIOLOGICAL_ASYMMETRY_MAX_MM:
        trigger_reasons.append("BORDERLINE_ASYMMETRY")
        if delta_brow_px > settings.BROW_ASYMMETRY_THRESHOLD_PX:
            is_brow_compensation = True
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
    # GIAI ĐOẠN 2: ĐỐI XỨNG SINH LÝ BÌNH THƯỜNG (delta_MRD1 <= 1.5 mm VÀ min(MRD1) > 0.0 mm) -> MỨC A
    # Ưu tiên cao nhất: Khi hai mắt đối xứng sinh lý (delta <= 1.5mm) thì luôn là Mức A,
    # độ chênh lệch chân mày tự nhiên không làm tăng bậc phân tầng.
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
