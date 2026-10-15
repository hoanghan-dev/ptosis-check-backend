"""
Định nghĩa cấu trúc dữ liệu phân tầng nguy cơ sàng lọc hình thái mi mắt.
Tuyệt đối tuân thủ G4: Không dùng từ ngữ thuộc danh mục cấm.
"""

from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, ConfigDict, Field


class ScreeningTier(str, Enum):
    TIER_A = "TIER_A"  # Không phát hiện dấu hiệu bất thường
    TIER_B = "TIER_B"  # Dấu hiệu nghi ngờ hoặc ranh giới
    TIER_C = "TIER_C"  # Khuyến nghị thăm khám chuyên khoa mắt


class TriageFlags(BaseModel):
    is_bilateral_severe_triggered: bool = False
    is_brow_compensation_detected: bool = False
    is_pupil_fallback_used: bool = False

    model_config = ConfigDict(populate_by_name=True)


class TriageResult(BaseModel):
    tier: ScreeningTier = Field(..., description="Phân tầng kết quả sàng lọc: TIER_A, TIER_B, TIER_C")
    trigger_reasons: List[str] = Field(default_factory=list, description="Danh sách mã lý do kích hoạt phân tầng")
    description_vi: str = Field(..., description="Mô tả kết quả bằng ngôn ngữ đại chúng")
    recommendation_vi: str = Field(..., description="Khuyến nghị hành vi tiếp theo")
    safety_disclaimer_vi: str = Field(
        default=(
            "Kết quả này chỉ mang tính sàng lọc hình thái ban đầu, không thay thế "
            "thăm khám y khoa chuyên sâu. Vui lòng tham vấn chuyên gia hoặc bác sĩ "
            "chuyên khoa mắt khi có băn khoăn."
        ),
        description="Thông cáo an toàn phi can thiệp y tế bắt buộc",
    )
    is_bilateral_severe_triggered: bool = Field(default=False, description="Cờ chốt chặn an toàn hai mắt")
    flags: Optional[TriageFlags] = Field(default=None, description="Các cờ trạng thái chi tiết")

    model_config = ConfigDict(populate_by_name=True)
