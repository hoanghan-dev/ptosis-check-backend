"""
Cấu hình hệ thống và các ngưỡng tham số lâm sàng sàng lọc mi mắt.
Tuân thủ tuyệt đối quy tắc không dùng từ ngữ thuộc danh mục cấm (G4).
"""

from pathlib import Path
from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    PROJECT_NAME: str = "PtosisCheck API"
    VERSION: str = "1.0.0"
    API_V1_PREFIX: str = "/api/v1"

    # CORS: chỉ cho phép các frontend origin đã xác định
    CORS_ORIGINS: List[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "https://ptosis-check.remicare.id.vn",
        "https://remicare.id.vn",
    ]

    # Ngưỡng phân tầng sàng lọc mi mắt (Đông Á / Dân số học - Asymmetry-First)
    CRITICAL_SEVERE_MRD1_MM: float = 2.0
    ABSOLUTE_SEVERE_MRD1_MM: float = 2.0
    HIGH_SYMMETRY_DELTA_MAX_MM: float = 0.5
    SYMMETRIC_MIN_MRD1_MM: float = 2.0
    TIER_A_MIN_MRD1_MM: float = 2.5
    TIER_B_MIN_MRD1_MM: float = 2.0
    PHYSIOLOGICAL_ASYMMETRY_MAX_MM: float = 1.5
    BORDERLINE_ASYMMETRY_MAX_MM: float = 2.0
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

    # Ngưỡng chuyên biệt cho chế độ tải ảnh crop mắt (eye_crop)
    EYE_CROP_MIN_RESOLUTION_PX: int = 150
    EYE_CROP_EDGE_MARGIN_PX: int = 2

    # Ngưỡng dung lượng tệp tải lên (10KB - 15MB cho eye_crop; 50KB - 15MB cho fullface)
    MIN_FILE_SIZE_FULLFACE_BYTES: int = 50 * 1024  # 50 KB
    MIN_FILE_SIZE_EYE_CROP_BYTES: int = 10 * 1024  # 10 KB
    MAX_FILE_SIZE_BYTES: int = 15 * 1024 * 1024  # 15 MB

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

    # Ngưỡng phát hiện phản quang giác mạc nâng cao từ V2 (Module M4)
    REFLEX_MAX_ASPECT_RATIO: float = 2.8  # TODO_PILOT: Giới hạn tỷ lệ dài/rộng vệt lóa
    REFLEX_MAX_AREA_PX: int = 60  # TODO_PILOT: Diện tích tuyệt đối tối đa đốm phản quang
    REFLEX_MAX_AREA_IRIS_RATIO: float = 0.05  # Diện tích tương đối tối đa so với pi * R_iris^2
    REFLEX_CLUSTER_DISTANCE_PX: float = 8.0  # Khoảng cách tối thiểu giữa 2 ứng viên cạnh tranh
    REFLEX_CLUSTER_DISTANCE_IRIS_RATIO: float = 0.20  # Hệ số khoảng cách cụm theo R_iris
    REFLEX_CLUSTER_SCORE_GAP: float = 3.0  # Độ chênh điểm số tối đa xác định cụm cạnh tranh

    # Ngưỡng định vị đồng tử độc lập (Module M4.5 - pupil_service)
    PUPIL_SEARCH_RADIUS_FACTOR: float = 0.85  # Bán kính tìm kiếm đồng tử trong mống mắt
    PUPIL_MIN_CONTRAST_DIFF: float = 12.0  # Độ tương phản tối thiểu (P85 - P15) trong ROI
    PUPIL_MIN_RADIUS_RATIO: float = 0.15  # Bán kính đồng tử tối thiểu / R_iris
    PUPIL_MAX_RADIUS_RATIO: float = 0.75  # Bán kính đồng tử tối đa / R_iris
    PUPIL_MAX_OFFSET_FROM_IRIS: float = 0.25  # Độ lệch tâm tối đa cho phép / Đường kính mống mắt
    PUPIL_MIN_CIRCULARITY: float = 0.35  # Độ tròn tối thiểu (4 * pi * Area / Perimeter^2)
    PUPIL_FALLBACK_CONFIDENCE: float = 0.75  # Điểm tin cậy khi dự phòng sang đồng tử đo được
    IRIS_CENTER_FALLBACK_CONFIDENCE: float = 0.60  # Điểm tin cậy khi dự phòng sang tâm mống mắt MediaPipe

    # Ngưỡng kiểm tra tính toàn vẹn hình thái học mống mắt hai bên
    IRIS_MAX_BILATERAL_DIFF_RATIO: float = 0.15  # Công thức V2: chênh lệch tối đa 15% theo mẫu số max(OD, OS)

    # Ngưỡng an toàn biên khung hình (Module M2 - Quality Gate)
    OCULAR_EDGE_MARGIN_PX: int = 2  # Khoảng cách an toàn tối thiểu từ mốc mắt đến mép ảnh

    # Cấu hình tự động định hướng ảnh (Phase B: EXIF & 4-Direction Auto-Orientation)
    AUTO_ORIENTATION_ENABLED: bool = True
    MAX_ORIENTATION_PROBE_ROLL_DEG: float = 15.0

    # Cấu hình Nhánh Xử lý Dual-Eye Crop (Phase D: Periocular Pipeline)
    PERIOCULAR_PIPELINE_ENABLED: bool = True
    PERIOCULAR_SHARP_THRESHOLD: float = 20.0
    PERIOCULAR_MAX_ROLL_DEG: float = 7.0
    MIN_PERIOCULAR_CROP_HEIGHT: int = 180
    MIN_PERIOCULAR_CROP_WIDTH: int = 300
    PERIOCULAR_MIN_SCALE_FACTOR: float = 0.015
    PERIOCULAR_MAX_SCALE_FACTOR: float = 0.60

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )


settings = Settings()
