"""
Endpoint tiếp nhận ảnh và thực thi quy trình phân tích sàng lọc mi mắt.
Tuân thủ G2: Xử lý hoàn toàn trong RAM (In-Memory Zero-Retention).
Tuân thủ G4: Tuyệt đối không sử dụng từ ngữ thuộc danh mục cấm.
"""

from datetime import datetime, timezone
import uuid
import cv2
import numpy as np
from fastapi import APIRouter, File, HTTPException, UploadFile, status

from app.core.config import settings
from app.core.memory_guard import purge_image_data
from app.schemas.measurement import (
    AnnotatedOverlayContainer,
    BilateralComparison,
    EyeAnnotationPoints,
    EyeFloatPair,
    ImageQualityAudit,
    MeasurementPayload,
    MeasurementsContainer,
    PointCoordinates,
    ScaleReference,
    SingleEyeMeasurement,
)
from app.schemas.quality import PoseAudit, QualityRejection
from app.schemas.triage import ScreeningTier, TriageFlags, TriageResult
from app.services.quality_gate_service import evaluate_quality
from app.services.landmark_service import detect_landmarks
from app.services.reflex_service import detect_corneal_reflex
from app.services.eyelid_margin_service import extract_upper_margin
from app.services.measurement_service import compute_measurements
from app.services.triage_service import evaluate_triage

router = APIRouter()


@router.post(
    "/analyze",
    response_model=MeasurementPayload,
    status_code=status.HTTP_200_OK,
    summary="Phân tích ảnh khuôn mặt phục vụ sàng lọc hình thái mi mắt",
)
async def analyze_eye_image(
    file: UploadFile = File(..., description="Tệp ảnh nhị phân (JPEG, PNG, WebP)"),
) -> MeasurementPayload:
    """
    Tiếp nhận ảnh, nạp vào RAM qua buffer nhị phân, giải mã bằng OpenCV và kiểm tra
    Cổng Sàng lọc Chất lượng Hình ảnh Lâm sàng (Module M2).
    Sau khi xử lý, toàn bộ dữ liệu ảnh trên RAM được giải phóng triệt để.
    """
    # 1. Đọc luồng bytes trực tiếp vào RAM
    contents = await file.read()
    if not contents:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Tệp ảnh tải lên trống hoặc không hợp lệ.",
        )

    # 2. Giải mã bằng buffer trong RAM (Tuyệt đối không ghi đĩa)
    nparr = np.frombuffer(contents, np.uint8)
    image_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image_bgr is None:
        purge_image_data(contents, nparr)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Không thể giải mã dữ liệu ảnh.",
        )

    try:
        # 3. Đánh giá Cổng Sàng lọc Chất lượng Chuyên sâu (Module M2)
        quality_gate = evaluate_quality(image_bgr)
        if not quality_gate.passed:
            rejection = QualityRejection(
                error_code="QUALITY_GATE_REJECTED",
                rejection_reason=quality_gate.rejection_reason or "QUALITY_CRITERIA_FAILED",
                rejection_message_vi=quality_gate.rejection_message_vi or "Hình ảnh không đạt tiêu chuẩn sàng lọc.",
                audit_details=quality_gate.audit_details,
            )
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=rejection.model_dump(),
            )

        # 4. Định vị Mốc Mống mắt & Xác lập Thước đo HVID (Module M3)
        landmarks = detect_landmarks(image_bgr)

        # 5. Phát hiện Phản quang Giác mạc Purkinje & Cơ chế Dự phòng (Module M4)
        image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        r_iris_od = landmarks.hvid_od_px / 2.0
        r_iris_os = landmarks.hvid_os_px / 2.0
        reflex_od = detect_corneal_reflex(image_gray, landmarks.iris_center_od, r_iris_od)
        reflex_os = detect_corneal_reflex(image_gray, landmarks.iris_center_os, r_iris_os)

        # 6. Phân đoạn Bờ Sụn mi Trên (Module M5)
        margin_od = extract_upper_margin(
            landmarks_upper=landmarks.upper_eyelid_od,
            reflex_point=reflex_od.reflex_point,
            image_gray=image_gray,
        )
        margin_os = extract_upper_margin(
            landmarks_upper=landmarks.upper_eyelid_os,
            reflex_point=reflex_os.reflex_point,
            image_gray=image_gray,
        )

        # 7. Khối Đo lường MRD1, Độ Bất đối xứng & Độ Tin cậy (Module M6)
        measurements_container, bilateral_comparison = compute_measurements(
            landmarks=landmarks,
            reflex_od=reflex_od,
            reflex_os=reflex_os,
            margin_od=margin_od,
            margin_os=margin_os,
            sharpness_score=quality_gate.sharpness_score,
        )

        # 8. Phân tầng Nguy cơ Ưu tiên Bất đối xứng (Module M7)
        is_reflex_occluded = (
            measurements_container.right_eye_od.mrd1_mm < 0
            or measurements_container.left_eye_os.mrd1_mm < 0
        )
        is_fallback_used = reflex_od.has_fallback or reflex_os.has_fallback

        triage_result = evaluate_triage(
            od_mrd1_mm=measurements_container.right_eye_od.mrd1_mm,
            os_mrd1_mm=measurements_container.left_eye_os.mrd1_mm,
            delta_mrd1_mm=bilateral_comparison.delta_mrd1_mm,
            delta_brow_px=bilateral_comparison.delta_brow_px or 0.0,
            is_reflex_occluded=is_reflex_occluded,
            is_pupil_fallback_used=is_fallback_used,
        )

        # Cấu trúc payload chuẩn theo Mục 15 của 04-measurement-specification.md
        session_id = str(uuid.uuid4())
        current_time = datetime.now(timezone.utc).isoformat()

        payload = MeasurementPayload(
            session_id=session_id,
            timestamp=current_time,
            image_quality_audit=ImageQualityAudit(
                passed=True,
                rejection_reason=None,
                pose=quality_gate.pose or PoseAudit(yaw=0.0, pitch=0.0, roll=0.0),
                sharpness_score=quality_gate.sharpness_score or 100.0,
                mean_illumination=quality_gate.mean_illumination or 120.0,
            ),
            scale_reference=ScaleReference(
                method="HVID_WTW_POPULATION_AVERAGE",
                assumed_corneal_diameter_mm=settings.ASSUMED_CORNEAL_DIAMETER_MM,
                hvid_pixels=EyeFloatPair(
                    right_eye=landmarks.hvid_od_px,
                    left_eye=landmarks.hvid_os_px,
                ),
                scale_factor_mm_per_px=EyeFloatPair(
                    right_eye=landmarks.scale_factor_od,
                    left_eye=landmarks.scale_factor_os,
                ),
            ),
            measurements=measurements_container,
            bilateral_comparison=bilateral_comparison,
            triage_result=triage_result,
            annotated_overlay_coordinates=AnnotatedOverlayContainer(
                right_eye=EyeAnnotationPoints(
                    corneal_reflex=[reflex_od.reflex_point[0], reflex_od.reflex_point[1]],
                    pupil_center=[landmarks.iris_center_od[0], landmarks.iris_center_od[1]],
                    iris_limbus_contour=[
                        [landmarks.limbus_horizontal_od[0][0], landmarks.limbus_horizontal_od[0][1]],
                        [landmarks.limbus_horizontal_od[1][0], landmarks.limbus_horizontal_od[1][1]],
                    ],
                    upper_eyelid_contour=[[pt[0], pt[1]] for pt in margin_od.spline_curve_points],
                    lower_eyelid_contour=[[pt[0], pt[1]] for pt in landmarks.lower_eyelid_od],
                ),
                left_eye=EyeAnnotationPoints(
                    corneal_reflex=[reflex_os.reflex_point[0], reflex_os.reflex_point[1]],
                    pupil_center=[landmarks.iris_center_os[0], landmarks.iris_center_os[1]],
                    iris_limbus_contour=[
                        [landmarks.limbus_horizontal_os[0][0], landmarks.limbus_horizontal_os[0][1]],
                        [landmarks.limbus_horizontal_os[1][0], landmarks.limbus_horizontal_os[1][1]],
                    ],
                    upper_eyelid_contour=[[pt[0], pt[1]] for pt in margin_os.spline_curve_points],
                    lower_eyelid_contour=[[pt[0], pt[1]] for pt in landmarks.lower_eyelid_os],
                ),
            ),
        )

        return payload
    finally:
        # G2: Giải phóng triệt để các biến ảnh khỏi RAM
        purge_image_data(contents, nparr, image_bgr, locals().get("image_gray", None))
