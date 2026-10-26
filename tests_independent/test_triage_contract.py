"""
Bộ kiểm thử độc lập cho Hợp đồng Phân tầng Nguy cơ Lâm sàng (Clinical Triage Contract).
Nguồn chân lý (Oracle):
- docs/02-requirements.md: FR-022, FR-023, FR-024, FR-025, FR-029
- docs/06-system-architecture.md: § 1 Nguyên tắc P3, § 8.6
- docs/04-measurement-specification.md: § 8.2, § 16

Mục tiêu chính:
Phát hiện các lỗi che giấu sụp mi hai bên (Bilateral Ptosis), phân tầng sai mức độ,
hoặc vi phạm danh mục từ ngữ cấm.
"""

import re
import pytest
from app.schemas.measurement import ReflexVisibilityEnum
from app.schemas.triage import ScreeningTier
from app.services.triage_service import evaluate_triage

BANNED_DIAGNOSTIC_TERMS = [
    "chẩn đoán", "bệnh nhân", "bệnh", "nhược cơ", "liệt",
    "horner", "bell", "myasthenia", "diagnosis", "patient", "diagnose"
]


class TestTriageClinicalContract:
    """Kiểm thử các quy tắc nghiệp vụ phân tầng lâm sàng theo FR-023 và P3."""

    def test_bilateral_severe_ptosis_under_2mm_must_not_be_tier_a(self):
        """
        [IT-001 / BR-006]
        Ca sụp mi hai bên đối xứng nghiêm trọng:
        OD = 1.2 mm, OS = 1.2 mm (Cả 2 mắt đều < 2.0 mm, đối xứng delta = 0.0 mm).
        Theo FR-023: 'sụp mi nặng cả hai bên (MRD1 < 2.0 mm cả hai mắt)' phải xếp vào MỨC C.
        Nếu hệ thống xếp loại TIER_A (Bình thường), đây là BUG NGUY HIỂM đe dọa an toàn người dùng.
        """
        res = evaluate_triage(
            od_mrd1_mm=1.2,
            os_mrd1_mm=1.2,
            delta_mrd1_mm=0.0,
            delta_brow_px=0.0,
        )

        assert res.tier != ScreeningTier.TIER_A, (
            f"VI PHẠM AN TOÀN LÂM SÀNG BR-006: Cả hai mắt đều sụp nặng (MRD1=1.2mm < 2.0mm) "
            f"nhưng hệ thống xếp vào {res.tier} (Bình thường)! "
            f"Bệnh nhân sụp mi hai bên bị kết luận là mắt bình thường."
        )
        assert res.tier == ScreeningTier.TIER_C, (
            f"Kỳ vọng TIER_C cho ca sụp mi 2 bên < 2.0mm theo FR-023, nhưng nhận {res.tier}"
        )
        assert res.is_bilateral_severe_triggered is True, (
            "Chốt chặn hai bên is_bilateral_severe_triggered phải bật True khi cả 2 mắt < 2.0mm"
        )

    def test_bilateral_borderline_droop_under_2mm_must_not_be_tier_a(self):
        """
        [IT-002 / BR-006]
        Ca sụp mi hai bên ranh giới:
        OD = 1.5 mm, OS = 1.6 mm (delta = 0.1 mm, cả 2 mắt < 2.0 mm, không nhướng mày).
        Theo FR-023: MRD1 < 2.0mm ở cả 2 mắt không được phép là TIER_A.
        """
        res = evaluate_triage(
            od_mrd1_mm=1.5,
            os_mrd1_mm=1.6,
            delta_mrd1_mm=0.1,
            delta_brow_px=0.0,
        )

        assert res.tier != ScreeningTier.TIER_A, (
            f"VI PHẠM BR-006: Hai mắt mở rất thấp (1.5mm và 1.6mm) nhưng xếp loại TIER_A "
            f"chỉ vì chênh lệch nhỏ (delta=0.1mm)."
        )

    def test_significant_asymmetry_over_2mm_triggers_tier_c(self):
        """
        [IT-003 / BR-007]
        Bất đối xứng rõ rệt (delta_MRD1 > 2.0 mm) bắt buộc xếp vào TIER_C.
        OD = 3.8 mm, OS = 1.5 mm -> delta = 2.3 mm.
        """
        res = evaluate_triage(
            od_mrd1_mm=3.8,
            os_mrd1_mm=1.5,
            delta_mrd1_mm=2.3,
            delta_brow_px=2.0,
        )

        assert res.tier == ScreeningTier.TIER_C
        assert any("SIGNIFICANT_ASYMMETRY" in reason or "2MM" in reason for reason in res.trigger_reasons), (
            f"Lý do kích hoạt phải phản ánh bất đối xứng > 2mm, thực tế: {res.trigger_reasons}"
        )

    def test_borderline_asymmetry_between_1_5_and_2_0_triggers_tier_b(self):
        """
        [IT-004 / BR-007]
        Bất đối xứng ranh giới (1.5 mm < delta_MRD1 <= 2.0 mm) bắt buộc xếp vào TIER_B.
        OD = 3.6 mm, OS = 1.9 mm -> delta = 1.7 mm.
        """
        res = evaluate_triage(
            od_mrd1_mm=3.6,
            os_mrd1_mm=1.9,
            delta_mrd1_mm=1.7,
            delta_brow_px=0.0,
        )

        assert res.tier == ScreeningTier.TIER_B
        assert any("BORDERLINE_ASYMMETRY" in reason for reason in res.trigger_reasons)

    def test_normal_physiological_symmetry_triggers_tier_a(self):
        """
        [IT-005 / BR-007]
        Mắt mở tốt, đối xứng sinh lý bình thường:
        OD = 3.2 mm, OS = 3.0 mm -> delta = 0.2 mm.
        Kỳ vọng: TIER_A (PHYSIOLOGICAL_NORMAL).
        """
        res = evaluate_triage(
            od_mrd1_mm=3.2,
            os_mrd1_mm=3.0,
            delta_mrd1_mm=0.2,
            delta_brow_px=0.0,
        )

        assert res.tier == ScreeningTier.TIER_A
        assert "PHYSIOLOGICAL_NORMAL" in res.trigger_reasons

    def test_brow_compensation_elevates_borderline_to_tier_b(self):
        """
        [IT-008 / BR-008]
        Chốt chặn nhướng mày bù trừ (Frontalis Brow Compensation):
        delta_MRD1 <= 1.5mm, min_MRD1 < 2.0mm KÈM delta_Brow > 8px -> TIER_B.
        OD = 1.8 mm, OS = 2.4 mm (delta = 0.6 mm), delta_brow = 10 px.
        """
        res = evaluate_triage(
            od_mrd1_mm=1.8,
            os_mrd1_mm=2.4,
            delta_mrd1_mm=0.6,
            delta_brow_px=10.0,
        )

        assert res.tier == ScreeningTier.TIER_B
        assert "BROW_COMPENSATION_SUSPECTED" in res.trigger_reasons
        assert res.flags.is_brow_compensation_detected is True

    def test_occluded_lid_triggers_tier_c_override(self):
        """
        [IT-006 / BR-009]
        Bờ mi che lấp hoàn toàn phản quang (OCCLUDED_BY_LID) phải kích hoạt TIER_C ngay lập tức.
        """
        res = evaluate_triage(
            od_mrd1_mm=None,
            os_mrd1_mm=3.2,
            delta_mrd1_mm=None,
            reflex_visibility_od=ReflexVisibilityEnum.OCCLUDED_BY_LID,
            reflex_visibility_os=ReflexVisibilityEnum.VISIBLE,
        )

        assert res.tier == ScreeningTier.TIER_C
        assert any("REFLEX_OCCLUDED" in r or "OCCLUDED" in r for r in res.trigger_reasons)

    @pytest.mark.parametrize("tier_eval", [
        (4.0, 4.0, 0.0, 0.0),    # TIER_A
        (3.5, 1.8, 1.7, 0.0),    # TIER_B
        (4.0, 1.2, 2.8, 0.0),    # TIER_C
    ])
    def test_zero_banned_diagnostic_words_in_triage_outputs(self, tier_eval):
        """
        [IT-013 / BR-010]
        Khẳng định 100% văn bản mô tả, khuyến nghị và miễn trừ không chứa từ ngữ y tế cấm (G4).
        """
        od, os_val, delta, brow = tier_eval
        res = evaluate_triage(od_mrd1_mm=od, os_mrd1_mm=os_val, delta_mrd1_mm=delta, delta_brow_px=brow)

        all_text = f"{res.description_vi} {res.recommendation_vi} {res.safety_disclaimer_vi}".lower()
        for banned in BANNED_DIAGNOSTIC_TERMS:
            pattern = r"\b" + re.escape(banned) + r"\b"
            assert not re.search(pattern, all_text), (
                f"VI PHẠM QUY TẮC PHI CHẨN ĐOÁN (BR-010): Phát hiện từ cấm '{banned}' "
                f"trong kết quả phân tầng: {all_text}"
            )
