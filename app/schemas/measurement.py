"""
Định nghĩa cấu trúc dữ liệu MeasurementPayload chuẩn hóa cho Khối Đo lường PtosisCheck.
Khớp 100% với giao ước tại Mục 15 của 04-measurement-specification.md
và Mục 9.1 của 06-system-architecture.md.
Tuyệt đối tuân thủ G4: Phi can thiệp y tế chuyên sâu.
"""

from typing import List, Optional, Tuple, Dict, Any, Union
from pydantic import BaseModel, ConfigDict, Field
from app.schemas.triage import TriageResult
from app.schemas.quality import PoseAudit


class PointCoordinates(BaseModel):
    x: float
    y: float

    model_config = ConfigDict(populate_by_name=True)


class ImageQualityAudit(BaseModel):
    passed: bool
    rejection_reason: Optional[str] = None
    pose: PoseAudit
    sharpness_score: float
    mean_illumination: float

    model_config = ConfigDict(populate_by_name=True)


class EyeFloatPair(BaseModel):
    right_eye: float
    left_eye: float

    model_config = ConfigDict(populate_by_name=True)


class ScaleReference(BaseModel):
    method: str = "HVID_WTW_POPULATION_AVERAGE"
    assumed_corneal_diameter_mm: float = 11.71
    hvid_pixels: EyeFloatPair
    scale_factor_mm_per_px: EyeFloatPair

    model_config = ConfigDict(populate_by_name=True)


class SingleEyeMeasurement(BaseModel):
    mrd1_px: float
    mrd1_mm: float
    mrd1_estimated_range: str
    pfh_px: Optional[float] = None
    pfh_mm: Optional[float] = None
    reference_point_used: str = "CORNEAL_LIGHT_REFLEX"
    reference_coordinates: Optional[PointCoordinates] = None
    eyelid_margin_point: Optional[PointCoordinates] = None
    status: str = "VALID"
    confidence_score: float = 1.0
    warning_flags: List[str] = Field(default_factory=list)

    model_config = ConfigDict(populate_by_name=True)


class MeasurementsContainer(BaseModel):
    right_eye_od: SingleEyeMeasurement
    left_eye_os: SingleEyeMeasurement

    model_config = ConfigDict(populate_by_name=True)


class BilateralComparison(BaseModel):
    delta_mrd1_mm: float
    signed_delta_mrd1_mm: float
    asymmetry_classification: str
    delta_pfh_mm: Optional[float] = None
    delta_brow_px: Optional[float] = None
    eyebrow_height_discrepancy_px: Optional[float] = None

    model_config = ConfigDict(populate_by_name=True)


class EyeAnnotationPoints(BaseModel):
    corneal_reflex: List[float] = Field(..., description="[x, y] phản quang")
    pupil_center: Optional[List[float]] = Field(default=None, description="[x, y] tâm mống mắt/đồng tử")
    iris_limbus_contour: List[List[float]] = Field(default_factory=list, description="Danh sách các điểm viền mống mắt")
    upper_eyelid_contour: List[List[float]] = Field(default_factory=list, description="Danh sách các điểm bờ mi trên")
    lower_eyelid_contour: Optional[List[List[float]]] = Field(default=None, description="Danh sách các điểm bờ mi dưới")

    model_config = ConfigDict(populate_by_name=True)


class AnnotatedOverlayContainer(BaseModel):
    right_eye: EyeAnnotationPoints
    left_eye: EyeAnnotationPoints

    model_config = ConfigDict(populate_by_name=True)


class MeasurementPayload(BaseModel):
    session_id: str
    timestamp: str
    image_quality_audit: ImageQualityAudit
    scale_reference: ScaleReference
    measurements: MeasurementsContainer
    bilateral_comparison: BilateralComparison
    annotated_overlay_coordinates: AnnotatedOverlayContainer
    triage_result: Optional[TriageResult] = None

    model_config = ConfigDict(populate_by_name=True)
