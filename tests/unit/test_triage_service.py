"""
Kiểm thử đơn vị và tích hợp cho Module M7: Triage Service (Asymmetry-First Triage Engine).
Bao phủ 100% 20 kịch bản kiểm thử biên quy định trong plan_implement.md.
Tuyệt đối tuân thủ G3: Ưu tiên bất đối xứng & chốt chặn hai bên.
Tuyệt đối tuân thủ G4: Không dùng từ ngữ thuộc danh mục cấm.
"""

from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from app.core.config import settings
from app.main import app
from app.schemas.triage import ScreeningTier
from app.services.triage_service import evaluate_triage
from tests.test_helpers import get_sample_image_path

client = TestClient(app)
SAMPLE_IMAGE_PATH = get_sample_image_path()


# ==============================================================================
# BẢNG 20 TEST CASES BIÊN THEO ĐẶC TẢ PLAN_IMPLEMENT.MD (MỤC 7.2)
# ==============================================================================

@pytest.mark.parametrize(
    "case_id, od_mrd1, os_mrd1, delta_mrd1, delta_brow, is_occluded, expected_tier, expected_bilateral_severe",
    [
        # TC-7.1: MRD1_OD=4.0, MRD1_OS=4.0 -> Mức A
        ("TC-7.1", 4.0, 4.0, 0.0, 0.0, False, ScreeningTier.TIER_A, False),
        # TC-7.2: MRD1_OD=4.0, MRD1_OS=2.8, delta=1.2 -> Mức B (bất đối xứng ranh giới)
        ("TC-7.2", 4.0, 2.8, 1.2, 0.0, False, ScreeningTier.TIER_B, False),
        # TC-7.3: MRD1_OD=4.0, MRD1_OS=2.3, delta=1.7 -> Mức B (bất đối xứng ranh giới <= 2.0mm)
        ("TC-7.3", 4.0, 2.3, 1.7, 0.0, False, ScreeningTier.TIER_B, False),
        # TC-7.4: MRD1_OD=1.0, MRD1_OS=1.0, delta=0.0 -> Mức C (chốt chặn nghiêm trọng < 1.5mm)
        ("TC-7.4", 1.0, 1.0, 0.0, 0.0, False, ScreeningTier.TIER_C, True),
        # TC-7.5: MRD1_OD=1.5, MRD1_OS=1.5 -> Mức C (chốt chặn hai mắt < 1.8mm)
        ("TC-7.5", 1.5, 1.5, 0.0, 0.0, False, ScreeningTier.TIER_C, True),
        # TC-7.6: MRD1_OD=2.0, MRD1_OS=2.0, delta=0.0 -> Mức A (đối xứng cao delta <= 0.5, min >= 2.0mm)
        ("TC-7.6", 2.0, 2.0, 0.0, 0.0, False, ScreeningTier.TIER_A, False),
        # TC-7.7: MRD1_OD=3.0, MRD1_OS=3.0 -> Mức A
        ("TC-7.7", 3.0, 3.0, 0.0, 0.0, False, ScreeningTier.TIER_A, False),
        # TC-7.8: MRD1_OD=2.5, MRD1_OS=2.5 -> Mức A (ngưỡng sàn Mức A)
        ("TC-7.8", 2.5, 2.5, 0.0, 0.0, False, ScreeningTier.TIER_A, False),
        # TC-7.9: MRD1_OD=2.4, MRD1_OS=2.4, delta=0.0 -> Mức A (đối xứng cao delta <= 0.5, hình thái Á Đông không bị phạt)
        ("TC-7.9", 2.4, 2.4, 0.0, 0.0, False, ScreeningTier.TIER_A, False),
        # TC-7.10: Phản quang bị che lấp -> Mức C (bất kể MRD1 cao)
        ("TC-7.10", 3.5, 3.5, 0.0, 0.0, True, ScreeningTier.TIER_C, False),
        # TC-7.11: MRD1_OD=4.0, MRD1_OS=3.0, delta=1.0 -> Mức A (ngưỡng biên delta <= 1.0)
        ("TC-7.11", 4.0, 3.0, 1.0, 0.0, False, ScreeningTier.TIER_A, False),
        # TC-7.12: MRD1_OD=4.0, MRD1_OS=2.9, delta=1.1 -> Mức B (vượt delta 1.0)
        ("TC-7.12", 4.0, 2.9, 1.1, 0.0, False, ScreeningTier.TIER_B, False),
        # TC-7.13: MRD1_OD=4.0, MRD1_OS=2.0, delta=2.0 -> Mức B (ngưỡng biên delta <= 2.0)
        ("TC-7.13", 4.0, 2.0, 2.0, 0.0, False, ScreeningTier.TIER_B, False),
        # TC-7.14: MRD1_OD=4.5, MRD1_OS=2.4, delta=2.1 -> Mức C (vượt delta > 2.0mm cảnh báo cần thăm khám)
        ("TC-7.14", 4.5, 2.4, 2.1, 0.0, False, ScreeningTier.TIER_C, False),
        # TC-7.15: delta_Brow > threshold, MRD1 bình thường -> Mức B (co cơ trán bù trừ)
        ("TC-7.15", 3.0, 3.0, 0.0, 10.0, False, ScreeningTier.TIER_B, False),
        # TC-7.16: MRD1_OD=1.9, MRD1_OS=4.0 -> Mức C (chốt chặn 1 mắt < 2.0mm và delta > 1.0mm)
        ("TC-7.16", 1.9, 4.0, 2.1, 0.0, False, ScreeningTier.TIER_C, False),
        # TC-7.17: MRD1_OD=0.0, MRD1_OS=4.0 -> Mức C (sụp mi hoàn toàn)
        ("TC-7.17", 0.0, 4.0, 4.0, 0.0, False, ScreeningTier.TIER_C, False),
        # TC-7.18: MRD1_OD=-1.0, MRD1_OS=4.0 -> Mức C (MRD1 âm)
        ("TC-7.18", -1.0, 4.0, 5.0, 0.0, False, ScreeningTier.TIER_C, False),
        # TC-7.19: MRD1_OD=2.1, MRD1_OS=2.3, delta=0.2 -> Mức A (đối xứng cao delta=0.2 <= 0.5, min >= 2.0mm)
        ("TC-7.19", 2.1, 2.3, 0.2, 0.0, False, ScreeningTier.TIER_A, False),
        # TC-7.20: MRD1_OD=2.5, MRD1_OS=3.5, delta=1.0 -> Mức A (biên đối xứng)
        ("TC-7.20", 2.5, 3.5, 1.0, 0.0, False, ScreeningTier.TIER_A, False),
    ],
)
def test_all_20_triage_decision_tree_cases(
    case_id: str,
    od_mrd1: float,
    os_mrd1: float,
    delta_mrd1: float,
    delta_brow: float,
    is_occluded: bool,
    expected_tier: ScreeningTier,
    expected_bilateral_severe: bool,
):
    """Xác thực 100% 20 ca kiểm thử biên phân tầng theo đặc tả FR-023."""
    res = evaluate_triage(
        od_mrd1_mm=od_mrd1,
        os_mrd1_mm=os_mrd1,
        delta_mrd1_mm=delta_mrd1,
        delta_brow_px=delta_brow,
        is_reflex_occluded=is_occluded,
    )

    assert res.tier == expected_tier, f"[{case_id}] Sai mức phân tầng: nhận {res.tier}, kỳ vọng {expected_tier}"
    assert res.is_bilateral_severe_triggered == expected_bilateral_severe
    assert len(res.description_vi) > 0
    assert len(res.recommendation_vi) > 0
    assert len(res.safety_disclaimer_vi) > 0


# ==============================================================================
# KIỂM THỬ TÍNH HIỆN DIỆN CỦA THÔNG CÁO AN TOÀN TRÊN 100% KẾT QUẢ (FR-029)
# ==============================================================================

def test_safety_disclaimer_presence_in_all_tiers():
    """Kiểm tra thông cáo an toàn bắt buộc xuất hiện trong mọi mức A, B, C."""
    res_a = evaluate_triage(3.5, 3.5, 0.0)
    res_b = evaluate_triage(4.0, 2.8, 1.2)
    res_c = evaluate_triage(1.2, 1.2, 0.0)

    for res in [res_a, res_b, res_c]:
        assert "chỉ mang tính sàng lọc" in res.safety_disclaimer_vi
        assert "không thay thế thăm khám" in res.safety_disclaimer_vi
        assert "bác sĩ" in res.safety_disclaimer_vi


def test_brow_compensation_flag():
    """Kiểm tra cờ co cơ trán nâng mi bù trừ được kích hoạt chính xác."""
    res = evaluate_triage(3.0, 3.0, 0.0, delta_brow_px=12.0)
    assert res.tier == ScreeningTier.TIER_B
    assert res.flags is not None
    assert res.flags.is_brow_compensation_detected is True
    assert "BROW_COMPENSATION_SUSPECTED" in res.trigger_reasons


def test_triage_integration_with_screening_endpoint():
    """Kiểm thử tích hợp endpoint /screening/analyze trả về TriageResult động chính xác."""
    if not SAMPLE_IMAGE_PATH.exists():
        pytest.skip("Không tìm thấy tệp ảnh mẫu thực tế.")

    with open(SAMPLE_IMAGE_PATH, "rb") as f:
        file_bytes = f.read()

    response = client.post(
        "/api/v1/screening/analyze",
        files={"file": ("test_benchmark.jpg", file_bytes, "image/jpeg")},
    )

    assert response.status_code == 200
    data = response.json()
    triage = data["triage_result"]

    assert triage["tier"] in ["TIER_A", "TIER_B", "TIER_C"]
    assert "description_vi" in triage
    assert "recommendation_vi" in triage
    assert "safety_disclaimer_vi" in triage
    assert len(triage["trigger_reasons"]) > 0
    assert "flags" in triage


def test_significant_asymmetry_over_2mm_trigger():
    """Kiểm tra bất đối xứng delta_MRD1 > 2.0mm mới kích hoạt Mức C với lý do SIGNIFICANT_ASYMMETRY_OVER_2MM."""
    # delta = 1.8 mm (<= 2.0 mm) -> Mức B
    res_b = evaluate_triage(4.0, 2.2, 1.8)
    assert res_b.tier == ScreeningTier.TIER_B
    assert "BORDERLINE_ASYMMETRY" in res_b.trigger_reasons
    assert "SIGNIFICANT_ASYMMETRY_OVER_2MM" not in res_b.trigger_reasons

    # delta = 2.0 mm (ngưỡng biên <= 2.0 mm) -> Mức B
    res_b_border = evaluate_triage(4.0, 2.0, 2.0)
    assert res_b_border.tier == ScreeningTier.TIER_B
    assert "BORDERLINE_ASYMMETRY" in res_b_border.trigger_reasons
    assert "SIGNIFICANT_ASYMMETRY_OVER_2MM" not in res_b_border.trigger_reasons

    # delta = 2.1 mm (> 2.0 mm) -> Mức C
    res_c = evaluate_triage(4.5, 2.4, 2.1)
    assert res_c.tier == ScreeningTier.TIER_C
    assert "SIGNIFICANT_ASYMMETRY_OVER_2MM" in res_c.trigger_reasons
