"""
Cấu hình hệ thống và các ngưỡng tham số lâm sàng sàng lọc mi mắt.
Tuân thủ tuyệt đối quy tắc không dùng từ ngữ thuộc danh mục cấm (G4).
"""

from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    PROJECT_NAME: str = "PtosisCheck API"
    VERSION: str = "1.0.0"
    API_V1_PREFIX: str = "/api/v1"

    # CORS cấu hình cho phép Frontend truy cập
    CORS_ORIGINS: List[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "https://*.vercel.app",
    ]

    # Ngưỡng lâm sàng sàng lọc mi mắt (Đông Á / Dân số học)
    ABSOLUTE_SEVERE_MRD1_MM: float = 2.0
    TIER_A_MIN_MRD1_MM: float = 2.5
    TIER_B_MIN_MRD1_MM: float = 2.0
    PHYSIOLOGICAL_ASYMMETRY_MAX_MM: float = 1.0
    BORDERLINE_ASYMMETRY_MAX_MM: float = 1.5
    BROW_ASYMMETRY_THRESHOLD_PX: float = 8.0

    # Ngưỡng cổng chất lượng tư thế và hình ảnh
    MAX_YAW_DEG: float = 10.0
    MAX_PITCH_DEG: float = 10.0
    MAX_ROLL_DEG: float = 5.0
    MIN_EYE_WIDTH_PX: int = 80
    MIN_IRIS_DIAMETER_PX: int = 30
    MIN_BRIGHTNESS: float = 40.0
    MAX_BRIGHTNESS: float = 220.0
    MAX_GLARE_RATIO: float = 0.15
    MIN_EYE_APERTURE_RATIO: float = 0.15
    LAPLACIAN_SHARP_THRESHOLD: float = 50.0

    # Thước đo giải phẫu quy chuẩn
    ASSUMED_CORNEAL_DIAMETER_MM: float = 11.71
    NOMINAL_TOLERANCE_MM: float = 0.5

    # Ngưỡng phát hiện phản quang giác mạc Purkinje và dự phòng (Module M4)
    PURKINJE_MIN_INTENSITY: int = 230
    PURKINJE_PERCENTILE: float = 95.0
    PURKINJE_CENTRAL_DISC_RATIO: float = 0.60
    PURKINJE_FALLBACK_CONFIDENCE: float = 0.65
    PURKINJE_NORMAL_CONFIDENCE: float = 1.0

    # Ngưỡng phân đoạn bờ sụn mi trên (Module M5)
    EYELID_GRADIENT_WINDOW_PX: int = 6
    EYELID_SPLINE_DEGREE: int = 3
    EYELID_MAX_EDGE_DEVIATION_PX: float = 6.0
    EYELID_MIN_GRADIENT_THRESHOLD: float = 8.0
    EYELID_DENSE_CURVE_POINTS: int = 30

    # Ngưỡng độ tin cậy đo lường (Module M6)
    CONFIDENCE_VALID_THRESHOLD: float = 0.70
    CONFIDENCE_LOW_THRESHOLD: float = 0.50

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )


settings = Settings()
