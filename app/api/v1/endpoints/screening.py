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
from app.utils.image_orientation import decode_and_orient_image
from app.schemas.measurement import (
    AnnotatedOverlayContainer,
    BilateralComparison,
    EyeAnnotationPoints,
    EyeFloatPair,
    ImageQualityAudit,
    MeasurementPayload,
    MeasurementStatusEnum,
    MeasurementsContainer,
    PointCoordinates,
    ReflexVisibilityEnum,
    ScaleReference,
    SingleEyeMeasurement,
)
from app.schemas.quality import PoseAudit, QualityRejection
from app.schemas.triage import ScreeningTier, TriageFlags, TriageResult
from app.services.input_router_service import InputType, classify_input_image
from app.services.periocular_adapter_service import process_periocular_crop
from app.services.quality_gate_service import evaluate_periocular_quality, evaluate_quality
from app.services.landmark_service import detect_landmarks
from app.services.reflex_service import ReflexDetectionStatus, detect_corneal_reflex
from app.services.pupil_service import detect_pupil_in_iris_roi
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

    # 2. Giải mã bằng buffer trong RAM và tự động định hướng (Module M1 / Phase B: EXIF & 4-Direction Auto-Orientation)
    image_bgr, orientation_meta = decode_and_orient_image(contents)

    if image_bgr is None:
        purge_image_data(contents)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Không thể giải mã dữ liệu ảnh.",
        )

    try:
        # 3. Tiêu chí chất lượng toàn cục tiên quyết (Global Illumination Pre-Check)
        gray_initial = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        mean_brightness = float(np.mean(gray_initial))

        if mean_brightness < settings.MIN_BRIGHTNESS:
            rejection = QualityRejection(
                error_code="QUALITY_GATE_REJECTED",
                rejection_reason="TOO_DARK",
                rejection_message_vi="Ảnh quá tối, thiếu ánh sáng ở vùng mặt và mắt. Vui lòng bật thêm đèn hoặc chuyển tới nơi đủ sáng và chụp lại.",
                audit_details={
                    "mean_brightness": round(mean_brightness, 2),
                    "min_allowed_brightness": settings.MIN_BRIGHTNESS,
                },
            )
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=rejection.model_dump(),
            )

        if mean_brightness > settings.MAX_BRIGHTNESS:
            rejection = QualityRejection(
                error_code="QUALITY_GATE_REJECTED",
                rejection_reason="TOO_BRIGHT",
                rejection_message_vi="Ảnh quá sáng hoặc bị cháy sáng. Vui lòng giảm bớt ánh sáng chiếu trực tiếp vào mặt và chụp lại.",
                audit_details={
                    "mean_brightness": round(mean_brightness, 2),
                    "max_allowed_brightness": settings.MAX_BRIGHTNESS,
                },
            )
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=rejection.model_dump(),
            )

        # 4. Điều tuyến dạng ảnh đầu vào: Toàn Mặt (Full-Face) hay Cận cảnh Hai Mắt (Dual-Eye Crop)
        input_type, router_meta = classify_input_image(image_bgr)

        if input_type == InputType.UNSUPPORTED:
            rejection = QualityRejection(
                error_code="QUALITY_GATE_REJECTED",
                rejection_reason="UNSUPPORTED_INPUT_FORMAT",
                rejection_message_vi="Không phát hiện khuôn mặt hoặc vùng hai mắt rõ ràng. Vui lòng chụp trực diện rõ hai mắt hoặc toàn khuôn mặt.",
                audit_details=router_meta,
            )
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=rejection.model_dump(),
            )

        if input_type == InputType.DUAL_EYE_CROP and settings.PERIOCULAR_PIPELINE_ENABLED:
            # Nhánh 2: Xử lý qua Periocular Adapter & Periocular Quality Gate
            landmarks = process_periocular_crop(image_bgr, router_meta)
            quality_gate = evaluate_periocular_quality(image_bgr, landmarks)
            if quality_gate.audit_details is None:
                quality_gate.audit_details = {}
            quality_gate.audit_details["applied_rotation_deg"] = orientation_meta.get("applied_rotation_deg", 0)
            quality_gate.audit_details["orientation_method"] = orientation_meta.get("method", "NONE")
            quality_gate.audit_details["input_type"] = "DUAL_EYE_CROP"
            quality_gate.audit_details["router_meta"] = router_meta

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
        else:
            # Nhánh 1: Xử lý ảnh Toàn Khuôn Mặt chuẩn (Full-Face Pipeline)
            quality_gate = evaluate_quality(image_bgr)
            if quality_gate.audit_details is None:
                quality_gate.audit_details = {}
            quality_gate.audit_details["applied_rotation_deg"] = orientation_meta.get("applied_rotation_deg", 0)
            quality_gate.audit_details["orientation_method"] = orientation_meta.get("method", "NONE")
            quality_gate.audit_details["input_type"] = "FULL_FACE"

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

        # Chốt chặn Cổng Chất lượng từ chối ảnh khi có cụm đốm sáng cạnh tranh (HTTP 422)
        # Lưu ý (Phase A - A4): Vệt lóa dài (LARGE_OR_ELONGATED_GLARE) được chuyển tiếp sang chuỗi dự phòng (Fallback) thay vì từ chối cứng HTTP 422

        if (
            reflex_od.status == ReflexDetectionStatus.CLUSTERED_REFLEX_CANDIDATES
            or reflex_os.status == ReflexDetectionStatus.CLUSTERED_REFLEX_CANDIDATES
        ):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=QualityRejection(
                    error_code="QUALITY_GATE_REJECTED",
                    rejection_reason="REFLEX_AMBIGUOUS_CLUSTER",
                    rejection_message_vi="Xuất hiện nhiều đốm phản xạ ánh sáng cạnh tranh trên giác mạc. Vui lòng tránh chụp dưới nguồn sáng chùm kép hoặc bóng đèn đôi.",
                ).model_dump(),
            )

        # Thiết lập Chuỗi Dự phòng 3 Cấp (Three-Tier Fallback Chain)
        # ---------------------------------------------------------------------
        # Mắt phải (OD)
        fallback_type_od = "NONE"
        glint_aspect_od = reflex_od.glint_aspect_ratio
        pupil_diam_od = None

        if reflex_od.has_fallback:
            pupil_res_od = detect_pupil_in_iris_roi(image_bgr, landmarks.iris_center_od, r_iris_od)
            if pupil_res_od.status == "DETECTED" and pupil_res_od.pupil_point is not None:
                reflex_od.reflex_point = pupil_res_od.pupil_point
                reflex_od.confidence_score = settings.PUPIL_FALLBACK_CONFIDENCE
                reflex_od.reference_point_used = "PUPIL_CENTER_FALLBACK"
                if "REFLEX_FALLBACK_TO_PUPIL" not in reflex_od.warning_flags: reflex_od.warning_flags.append("REFLEX_FALLBACK_TO_PUPIL")
                fallback_type_od = "PUPIL_DETECTOR"
                pupil_diam_od = pupil_res_od.pupil_diameter_px
            else:
                reflex_od.reflex_point = landmarks.iris_center_od
                reflex_od.confidence_score = settings.IRIS_CENTER_FALLBACK_CONFIDENCE
                reflex_od.reference_point_used = "PUPIL_CENTER_FALLBACK"
                if "REFLEX_FALLBACK_TO_IRIS_CENTER" not in reflex_od.warning_flags: reflex_od.warning_flags.append("REFLEX_FALLBACK_TO_IRIS_CENTER")
                fallback_type_od = "IRIS_CENTER"

        # Mắt trái (OS)
        fallback_type_os = "NONE"
        glint_aspect_os = reflex_os.glint_aspect_ratio
        pupil_diam_os = None

        if reflex_os.has_fallback:
            pupil_res_os = detect_pupil_in_iris_roi(image_bgr, landmarks.iris_center_os, r_iris_os)
            if pupil_res_os.status == "DETECTED" and pupil_res_os.pupil_point is not None:
                reflex_os.reflex_point = pupil_res_os.pupil_point
                reflex_os.confidence_score = settings.PUPIL_FALLBACK_CONFIDENCE
                reflex_os.reference_point_used = "PUPIL_CENTER_FALLBACK"
                if "REFLEX_FALLBACK_TO_PUPIL" not in reflex_os.warning_flags: reflex_os.warning_flags.append("REFLEX_FALLBACK_TO_PUPIL")
                fallback_type_os = "PUPIL_DETECTOR"
                pupil_diam_os = pupil_res_os.pupil_diameter_px
            else:
                reflex_os.reflex_point = landmarks.iris_center_os
                reflex_os.confidence_score = settings.IRIS_CENTER_FALLBACK_CONFIDENCE
                reflex_os.reference_point_used = "PUPIL_CENTER_FALLBACK"
                if "REFLEX_FALLBACK_TO_IRIS_CENTER" not in reflex_os.warning_flags: reflex_os.warning_flags.append("REFLEX_FALLBACK_TO_IRIS_CENTER")
                fallback_type_os = "IRIS_CENTER"

        # 6. Phân đoạn Bờ Sụn mi Trên (Module M5)
        margin_od = extract_upper_margin(
            landmarks_upper=landmarks.upper_eyelid_od,
            reflex_point=reflex_od.reflex_point,
            image_gray=image_gray,
            r_iris=r_iris_od,
        )
        margin_os = extract_upper_margin(
            landmarks_upper=landmarks.upper_eyelid_os,
            reflex_point=reflex_os.reflex_point,
            image_gray=image_gray,
            r_iris=r_iris_os,
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

        # Bổ sung các trường hình thái học V2 vào kết quả đo từng mắt
        measurements_container.right_eye_od.fallback_type = fallback_type_od
        measurements_container.right_eye_od.glint_aspect_ratio = glint_aspect_od
        measurements_container.right_eye_od.pupil_diameter_px = pupil_diam_od

        measurements_container.left_eye_os.fallback_type = fallback_type_os
        measurements_container.left_eye_os.glint_aspect_ratio = glint_aspect_os
        measurements_container.left_eye_os.pupil_diameter_px = pupil_diam_os

        if input_type == InputType.DUAL_EYE_CROP:
            if "PROCESSED_VIA_PERIOCULAR_PIPELINE" not in measurements_container.right_eye_od.warning_flags:
                measurements_container.right_eye_od.warning_flags.append("PROCESSED_VIA_PERIOCULAR_PIPELINE")
            if "PROCESSED_VIA_PERIOCULAR_PIPELINE" not in measurements_container.left_eye_os.warning_flags:
                measurements_container.left_eye_os.warning_flags.append("PROCESSED_VIA_PERIOCULAR_PIPELINE")

        # 8. Phân tầng Nguy cơ Ưu tiên Bất đối xứng (Module M7)
        # Bằng chứng giải phẫu che lấp phản quang/trục thị giác (Phase A - A3 & Phase C):
        # 1. Bờ mi trên che xuống dưới điểm mốc tham chiếu (ANATOMICAL_NEGATIVE_SUSPECTED)
        # 2. Hoặc bờ mi trên che xuống qua tâm giải phẫu mống mắt (y_margin >= y_iris_center)
        is_occluded_od = (
            "ANATOMICAL_NEGATIVE_SUSPECTED" in measurements_container.right_eye_od.warning_flags
            or margin_od.margin_point[1] >= landmarks.iris_center_od[1]
        )
        is_occluded_os = (
            "ANATOMICAL_NEGATIVE_SUSPECTED" in measurements_container.left_eye_os.warning_flags
            or margin_os.margin_point[1] >= landmarks.iris_center_os[1]
        )

        if is_occluded_od:
            measurements_container.right_eye_od.reflex_visibility = ReflexVisibilityEnum.OCCLUDED_BY_LID
            measurements_container.right_eye_od.mrd1_mm = None
            measurements_container.right_eye_od.mrd1_px = None
            measurements_container.right_eye_od.mrd1_estimated_range = None
            measurements_container.right_eye_od.status = MeasurementStatusEnum.MEASUREMENT_UNAVAILABLE
            if "REFLEX_OCCLUDED_BY_UPPER_LID" not in measurements_container.right_eye_od.warning_flags:
                measurements_container.right_eye_od.warning_flags.append("REFLEX_OCCLUDED_BY_UPPER_LID")
        elif reflex_od.status == ReflexDetectionStatus.NO_REFLEX_FOUND:
            measurements_container.right_eye_od.reflex_visibility = ReflexVisibilityEnum.NOT_VISIBLE_QUALITY

        if is_occluded_os:
            measurements_container.left_eye_os.reflex_visibility = ReflexVisibilityEnum.OCCLUDED_BY_LID
            measurements_container.left_eye_os.mrd1_mm = None
            measurements_container.left_eye_os.mrd1_px = None
            measurements_container.left_eye_os.mrd1_estimated_range = None
            measurements_container.left_eye_os.status = MeasurementStatusEnum.MEASUREMENT_UNAVAILABLE
            if "REFLEX_OCCLUDED_BY_UPPER_LID" not in measurements_container.left_eye_os.warning_flags:
                measurements_container.left_eye_os.warning_flags.append("REFLEX_OCCLUDED_BY_UPPER_LID")
        elif reflex_os.status == ReflexDetectionStatus.NO_REFLEX_FOUND:
            measurements_container.left_eye_os.reflex_visibility = ReflexVisibilityEnum.NOT_VISIBLE_QUALITY

        is_reflex_occluded = is_occluded_od or is_occluded_os
        is_fallback_used = reflex_od.has_fallback or reflex_os.has_fallback

        triage_result = evaluate_triage(
            od_mrd1_mm=measurements_container.right_eye_od.mrd1_mm,
            os_mrd1_mm=measurements_container.left_eye_os.mrd1_mm,
            delta_mrd1_mm=bilateral_comparison.delta_mrd1_mm,
            delta_brow_px=bilateral_comparison.delta_brow_px or 0.0,
            reflex_visibility_od=measurements_container.right_eye_od.reflex_visibility,
            reflex_visibility_os=measurements_container.left_eye_os.reflex_visibility,
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
                audit_details=quality_gate.audit_details,
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
                    corneal_reflex=None if is_occluded_od else [reflex_od.reflex_point[0], reflex_od.reflex_point[1]],
                    pupil_center=None if is_occluded_od else [
                        reflex_od.reflex_point[0] if fallback_type_od == "PUPIL_DETECTOR" else landmarks.iris_center_od[0],
                        reflex_od.reflex_point[1] if fallback_type_od == "PUPIL_DETECTOR" else landmarks.iris_center_od[1],
                    ],
                    iris_limbus_contour=[
                        [landmarks.limbus_horizontal_od[0][0], landmarks.limbus_horizontal_od[0][1]],
                        [landmarks.limbus_horizontal_od[1][0], landmarks.limbus_horizontal_od[1][1]],
                    ],
                    upper_eyelid_contour=[[pt[0], pt[1]] for pt in margin_od.spline_curve_points],
                    lower_eyelid_contour=[[pt[0], pt[1]] for pt in landmarks.lower_eyelid_od],
                ),
                left_eye=EyeAnnotationPoints(
                    corneal_reflex=None if is_occluded_os else [reflex_os.reflex_point[0], reflex_os.reflex_point[1]],
                    pupil_center=None if is_occluded_os else [
                        reflex_os.reflex_point[0] if fallback_type_os == "PUPIL_DETECTOR" else landmarks.iris_center_os[0],
                        reflex_os.reflex_point[1] if fallback_type_os == "PUPIL_DETECTOR" else landmarks.iris_center_os[1],
                    ],
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
        purge_image_data(contents, image_bgr, locals().get("image_gray", None))
