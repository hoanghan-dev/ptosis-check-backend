"""
Định nghĩa cấu trúc dữ liệu cho Cổng Sàng lọc Chất lượng hình ảnh.
Tuyệt đối tuân thủ G4: Không dùng từ ngữ thuộc danh mục cấm.
"""

from typing import Any, Dict, Optional
from pydantic import BaseModel, ConfigDict, Field


class PoseAudit(BaseModel):
    yaw: float = Field(..., description="Góc quay đầu quanh trục đứng (độ)")
    pitch: float = Field(..., description="Góc gật/ngửa đầu quanh trục ngang (độ)")
    roll: float = Field(..., description="Góc nghiêng đầu sang vai (độ)")

    model_config = ConfigDict(populate_by_name=True)


class QualityGateResult(BaseModel):
    passed: bool = Field(..., description="Trạng thái vượt qua cổng chất lượng")
    rejection_reason: Optional[str] = Field(None, description="Mã nguyên nhân từ chối nếu không đạt")
    rejection_message_vi: Optional[str] = Field(None, description="Hướng dẫn chụp lại bằng tiếng Việt")
    pose: Optional[PoseAudit] = Field(None, description="Kết quả ước lượng tư thế 3D")
    sharpness_score: Optional[float] = Field(None, description="Phương sai Laplacian đo độ sắc nét")
    mean_illumination: Optional[float] = Field(None, description="Độ sáng trung bình vùng mặt")
    glare_ratio: Optional[float] = Field(None, description="Tỷ lệ vùng lóa sáng trên mống mắt")
    aperture_ratio_od: Optional[float] = Field(None, description="Tỷ lệ mở khe mi mắt phải")
    aperture_ratio_os: Optional[float] = Field(None, description="Tỷ lệ mở khe mi mắt trái")
    audit_details: Optional[Dict[str, Any]] = Field(None, description="Chi tiết số đo kiểm định")

    model_config = ConfigDict(populate_by_name=True)


class QualityRejection(BaseModel):
    error_code: str = Field(default="QUALITY_GATE_REJECTED", description="Mã lỗi hệ thống")
    rejection_reason: str = Field(..., description="Mã định danh lý do từ chối")
    rejection_message_vi: str = Field(..., description="Thông điệp hướng dẫn người dùng")
    audit_details: Optional[Dict[str, Any]] = Field(default=None, description="Thông số kỹ thuật liên quan")

    model_config = ConfigDict(populate_by_name=True)
