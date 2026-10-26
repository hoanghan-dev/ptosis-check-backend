from app.services.landmark_service import LandmarkResult
"""
Dịch vụ Cổng Sàng lọc Chất lượng Hình ảnh Lâm sàng (Module M2).
Đánh giá ảnh trực tiếp trong bộ nhớ RAM, tuyệt đối không ghi đĩa (G2).
Tuân thủ nghiêm ngặt quy tắc không dùng từ ngữ thuộc danh mục cấm (G4).
"""

import math
from typing import Any, Dict, Optional, Tuple
import cv2
import numpy as np

_FACE_MESH_INIT_ERROR: Optional[str] = None
try:
    import mediapipe as mp
    mp_face_mesh = mp.solutions.face_mesh
    _FACE_MESH = mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
    )
except Exception as e:
    _FACE_MESH = None
    _FACE_MESH_INIT_ERROR = str(e)

from app.core.config import settings
from app.schemas.quality import PoseAudit, QualityGateResult
from app.utils.mediapipe_lock import mediapipe_lock

# Mô hình 3D chuẩn của khuôn mặt (OpenCV camera coordinates: x sang phải, y hướng xuống, z hướng ra xa camera)
# Các điểm mốc tương ứng: 1 (đỉnh mũi), 152 (cằm), 33 (đuôi mắt phải), 263 (đuôi mắt trái), 61 (khóe miệng phải), 291 (khóe miệng trái)
FACE_MODEL_3D = np.array(
    [
        (0.0, 0.0, 0.0),         # Đỉnh mũi (Mốc 1)
        (0.0, 330.0, 65.0),      # Cằm (Mốc 152)
        (-225.0, -170.0, 135.0), # Đuôi mắt phải quan sát (Mốc 33)
        (225.0, -170.0, 135.0),  # Đuôi mắt trái quan sát (Mốc 263)
        (-150.0, 150.0, 125.0),  # Khóe miệng phải quan sát (Mốc 61)
        (150.0, 150.0, 125.0),   # Khóe miệng trái quan sát (Mốc 291)
    ],
    dtype=np.float64,
)

# Chỉ số các mốc hình học MediaPipe
LANDMARK_NOSE_TIP = 1
LANDMARK_CHIN = 152
LANDMARK_RIGHT_EYE_OUTER = 33   # Mắt phải người quan sát (OD)
LANDMARK_RIGHT_EYE_INNER = 133
LANDMARK_RIGHT_EYE_TOP = 159
LANDMARK_RIGHT_EYE_BOTTOM = 145

LANDMARK_LEFT_EYE_OUTER = 263   # Mắt trái người quan sát (OS)
LANDMARK_LEFT_EYE_INNER = 362
LANDMARK_LEFT_EYE_TOP = 386
LANDMARK_LEFT_EYE_BOTTOM = 374

LANDMARK_MOUTH_LEFT = 61
LANDMARK_MOUTH_RIGHT = 291

# Mốc mống mắt (yêu cầu refine_landmarks=True)
LANDMARK_IRIS_CENTER_OD = 468
LANDMARK_IRIS_EDGE_OD_1 = 469
LANDMARK_IRIS_EDGE_OD_2 = 471

LANDMARK_IRIS_CENTER_OS = 473
LANDMARK_IRIS_EDGE_OS_1 = 474
LANDMARK_IRIS_EDGE_OS_2 = 476


def estimate_head_pose_3d(
    landmarks_px: np.ndarray,
    image_width: int,
    image_height: int,
) -> Tuple[float, float, float]:
    """
    Ước lượng các góc tư thế đầu 3D (pitch, yaw, roll) bằng solvePnP.
    Trả về giá trị tính theo độ (degrees).
    """
    focal_length = float(image_width)
    center_x = float(image_width) / 2.0
    center_y = float(image_height) / 2.0
    camera_matrix = np.array(
        [
            [focal_length, 0.0, center_x],
            [0.0, focal_length, center_y],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    image_points = np.array(
        [
            landmarks_px[LANDMARK_NOSE_TIP],
            landmarks_px[LANDMARK_CHIN],
            landmarks_px[LANDMARK_RIGHT_EYE_OUTER],
            landmarks_px[LANDMARK_LEFT_EYE_OUTER],
            landmarks_px[LANDMARK_MOUTH_LEFT],
            landmarks_px[LANDMARK_MOUTH_RIGHT],
        ],
        dtype=np.float64,
    )

    success, rotation_vector, translation_vector = cv2.solvePnP(
        FACE_MODEL_3D,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_EPNP,
    )

    if success and translation_vector[2, 0] > 0:
        # Tinh chỉnh lại nghiệm bằng solver lặp với điểm khởi tạo tối ưu
        success, rotation_vector, translation_vector = cv2.solvePnP(
            FACE_MODEL_3D,
            image_points,
            camera_matrix,
            dist_coeffs,
            rvec=rotation_vector,
            tvec=translation_vector,
            useExtrinsicGuess=True,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

    if not success or translation_vector[2, 0] <= 0:
        raise ValueError("Không thể ước lượng tư thế 3D từ mốc khuôn mặt.")

    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rotation_matrix)

    pitch = float(angles[0])
    yaw = float(angles[1])
    roll = float(angles[2])

    return pitch, yaw, roll


def compute_eye_sharpness(
    gray_image: np.ndarray,
    landmarks_px: np.ndarray,
    image_width: int,
    image_height: int,
) -> float:
    """
    Tính phương sai toán tử Laplacian trên vùng quan tâm (ROI) của hai mắt để đo độ sắc nét.
    """
    # Vùng mắt phải (OD)
    od_pts = landmarks_px[[LANDMARK_RIGHT_EYE_OUTER, LANDMARK_RIGHT_EYE_INNER, LANDMARK_RIGHT_EYE_TOP, LANDMARK_RIGHT_EYE_BOTTOM]]
    od_min_x = max(0, int(np.min(od_pts[:, 0])) - 10)
    od_max_x = min(image_width, int(np.max(od_pts[:, 0])) + 10)
    od_min_y = max(0, int(np.min(od_pts[:, 1])) - 10)
    od_max_y = min(image_height, int(np.max(od_pts[:, 1])) + 10)

    # Vùng mắt trái (OS)
    os_pts = landmarks_px[[LANDMARK_LEFT_EYE_OUTER, LANDMARK_LEFT_EYE_INNER, LANDMARK_LEFT_EYE_TOP, LANDMARK_LEFT_EYE_BOTTOM]]
    os_min_x = max(0, int(np.min(os_pts[:, 0])) - 10)
    os_max_x = min(image_width, int(np.max(os_pts[:, 0])) + 10)
    os_min_y = max(0, int(np.min(os_pts[:, 1])) - 10)
    os_max_y = min(image_height, int(np.max(os_pts[:, 1])) + 10)

    sharpness_scores = []

    if od_max_x > od_min_x and od_max_y > od_min_y:
        od_crop = gray_image[od_min_y:od_max_y, od_min_x:od_max_x]
        if od_crop.size > 0:
            sharpness_scores.append(float(cv2.Laplacian(od_crop, cv2.CV_64F).var()))

    if os_max_x > os_min_x and os_max_y > os_min_y:
        os_crop = gray_image[os_min_y:os_max_y, os_min_x:os_max_x]
        if os_crop.size > 0:
            sharpness_scores.append(float(cv2.Laplacian(os_crop, cv2.CV_64F).var()))

    if not sharpness_scores:
        return float(cv2.Laplacian(gray_image, cv2.CV_64F).var())

    return float(min(sharpness_scores))


def compute_iris_glare_ratio(
    gray_image: np.ndarray,
    center_px: np.ndarray,
    radius_px: float,
    glare_intensity_threshold: int = 245,
) -> float:
    """
    Tính tỷ lệ diện tích lóa sáng (pixel >= 245) trong đĩa mống mắt.
    """
    radius = max(5, int(round(radius_px)))
    cx, cy = int(round(center_px[0])), int(round(center_px[1]))
    h, w = gray_image.shape

    min_x = max(0, cx - radius)
    max_x = min(w, cx + radius + 1)
    min_y = max(0, cy - radius)
    max_y = min(h, cy + radius + 1)

    if max_x <= min_x or max_y <= min_y:
        return 0.0

    crop = gray_image[min_y:max_y, min_x:max_x]
    y_coords, x_coords = np.ogrid[: crop.shape[0], : crop.shape[1]]
    mask = ((x_coords + min_x - cx) ** 2 + (y_coords + min_y - cy) ** 2) <= radius ** 2

    iris_pixels = crop[mask]
    if iris_pixels.size == 0:
        return 0.0

    glare_pixels = np.sum(iris_pixels >= glare_intensity_threshold)
    return float(glare_pixels) / float(iris_pixels.size)


def evaluate_quality(image_bgr: np.ndarray) -> QualityGateResult:
    """
    Đánh giá chất lượng hình ảnh lâm sàng (Module M2):
    1. Kiểm tra tính hợp lệ và độ phân giải tối thiểu của mảng ảnh.
    2. Kiểm tra độ sáng xám trung bình toàn khung ảnh (40 <= mean_I <= 220).
    3. Trích xuất mốc hình học khuôn mặt qua MediaPipe FaceMesh.
    4. Ước tính tư thế đầu 3D qua solvePnP: Yaw <= +-10 deg, Pitch <= +-10 deg, Roll <= +-5 deg.
    5. Kiểm tra tỷ lệ mở khe mi >= 0.15 (phát hiện nhắm mắt / chớp mắt).
    6. Kiểm tra độ phân giải vùng mắt: bề rộng mắt >= 80 px, đường kính mống mắt >= 30 px.
    7. Kiểm tra độ lóa sáng trên mống mắt < 15%.
    8. Kiểm tra độ sắc nét bằng phương sai Laplacian trên ROI mắt >= tau_sharp.
    """
    if image_bgr is None or not isinstance(image_bgr, np.ndarray) or image_bgr.size == 0:
        return QualityGateResult(
            passed=False,
            rejection_reason="IMAGE_UNREADABLE",
            rejection_message_vi="Không thể đọc dữ liệu ảnh. Vui lòng tải lại ảnh hợp lệ.",
            audit_details={"error": "empty_or_invalid_array"},
        )

    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        return QualityGateResult(
            passed=False,
            rejection_reason="IMAGE_UNREADABLE",
            rejection_message_vi="Định dạng màu của ảnh không đúng chuẩn 3 kênh màu (BGR).",
            audit_details={"channels": image_bgr.ndim},
        )

    height, width = image_bgr.shape[:2]
    if height < 200 or width < 200:
        return QualityGateResult(
            passed=False,
            rejection_reason="LOW_RESOLUTION",
            rejection_message_vi="Độ phân giải ảnh quá nhỏ. Vui lòng chụp ảnh rõ nét hơn với độ phân giải cao hơn.",
            audit_details={"height": height, "width": width, "min_required": 200},
        )

    # Chuyển đổi sang ảnh xám
    gray_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    mean_brightness = float(np.mean(gray_image))

    # 1. Kiểm tra độ sáng xám trung bình
    if mean_brightness < settings.MIN_BRIGHTNESS:
        return QualityGateResult(
            passed=False,
            rejection_reason="TOO_DARK",
            rejection_message_vi="Ảnh quá tối, thiếu ánh sáng ở vùng mặt và mắt. Vui lòng bật thêm đèn hoặc chuyển tới nơi đủ sáng và chụp lại.",
            mean_illumination=round(mean_brightness, 2),
            audit_details={
                "mean_brightness": round(mean_brightness, 2),
                "min_allowed_brightness": settings.MIN_BRIGHTNESS,
            },
        )

    if mean_brightness > settings.MAX_BRIGHTNESS:
        return QualityGateResult(
            passed=False,
            rejection_reason="TOO_BRIGHT",
            rejection_message_vi="Ảnh quá sáng hoặc bị cháy sáng. Vui lòng giảm bớt ánh sáng chiếu trực tiếp vào mặt và chụp lại.",
            mean_illumination=round(mean_brightness, 2),
            audit_details={
                "mean_brightness": round(mean_brightness, 2),
                "max_allowed_brightness": settings.MAX_BRIGHTNESS,
            },
        )

    # 2. Nhận diện khuôn mặt và trích xuất mốc hình học
    if _FACE_MESH is None:
        return QualityGateResult(
            passed=False,
            rejection_reason="MODEL_UNAVAILABLE",
            rejection_message_vi="Module trích xuất mốc hình học chưa sẵn sàng trên máy chủ.",
            audit_details={
                "error": "mediapipe_facemesh_missing",
                "details": _FACE_MESH_INIT_ERROR or "FaceMesh initialization failed",
            },
        )

    rgb_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    with mediapipe_lock:
        results = _FACE_MESH.process(rgb_image)

    if not results.multi_face_landmarks or len(results.multi_face_landmarks) == 0:
        return QualityGateResult(
            passed=False,
            rejection_reason="NO_FACE_DETECTED",
            rejection_message_vi="Không phát hiện khuôn mặt. Vui lòng chụp ảnh chính diện, rõ nét và đủ ánh sáng.",
            mean_illumination=round(mean_brightness, 2),
            audit_details={"detected_faces": 0},
        )

    face_landmarks = results.multi_face_landmarks[0].landmark
    if len(face_landmarks) < 478:
        return QualityGateResult(
            passed=False,
            rejection_reason="LOW_RESOLUTION",
            rejection_message_vi="Không nhận diện đủ mốc mống mắt. Vui lòng chụp lại gần hơn.",
            audit_details={"landmarks_count": len(face_landmarks), "expected": 478},
        )

    # Tọa độ pixel các mốc hình học
    landmarks_px = np.array(
        [[lm.x * width, lm.y * height] for lm in face_landmarks],
        dtype=np.float32,
    )

    # 3. Ước tính tư thế đầu 3D (solvePnP)
    try:
        pitch, yaw, roll = estimate_head_pose_3d(landmarks_px, width, height)
    except Exception as e:
        return QualityGateResult(
            passed=False,
            rejection_reason="HEAD_POSE_ERROR",
            rejection_message_vi="Không thể xác định góc tư thế đầu. Vui lòng giữ thẳng đầu nhìn trực diện.",
            audit_details={"error": str(e)},
        )

    pose_audit = PoseAudit(
        pitch=round(pitch, 2),
        yaw=round(yaw, 2),
        roll=round(roll, 2),
    )

    # Kiểm tra góc nghiêng đầu sang bên (Roll <= +-5 deg)
    if abs(roll) > settings.MAX_ROLL_DEG:
        return QualityGateResult(
            passed=False,
            rejection_reason="HEAD_ROLL_EXCEEDED",
            rejection_message_vi="Đầu đang bị nghiêng sang một bên. Vui lòng giữ thẳng đầu thăng bằng và chụp lại.",
            pose=pose_audit,
            audit_details={
                "roll_deg": round(roll, 2),
                "max_allowed_roll_deg": settings.MAX_ROLL_DEG,
            },
        )

    # Kiểm tra góc quay đầu sang trái/phải (Yaw <= +-10 deg)
    if abs(yaw) > settings.MAX_YAW_DEG:
        return QualityGateResult(
            passed=False,
            rejection_reason="HEAD_YAW_EXCEEDED",
            rejection_message_vi="Đầu đang bị quay sang bên. Vui lòng nhìn thẳng trực diện vào camera và chụp lại.",
            pose=pose_audit,
            audit_details={
                "yaw_deg": round(yaw, 2),
                "max_allowed_yaw_deg": settings.MAX_YAW_DEG,
            },
        )

    # Kiểm tra góc ngửa/cúi đầu (Pitch <= +-10 deg)
    if abs(pitch) > settings.MAX_PITCH_DEG:
        return QualityGateResult(
            passed=False,
            rejection_reason="HEAD_PITCH_EXCEEDED",
            rejection_message_vi="Cằm đang ngửa hoặc cúi đầu. Vui lòng giữ điện thoại ngang tầm mắt, giữ thẳng đầu và chụp lại.",
            pose=pose_audit,
            audit_details={
                "pitch_deg": round(pitch, 2),
                "max_allowed_pitch_deg": settings.MAX_PITCH_DEG,
            },
        )

    # 3.5. Kiểm tra vùng mắt chạm mép viền ảnh (touches_crop_edge)
    margin = settings.OCULAR_EDGE_MARGIN_PX
    ocular_indices = [
        LANDMARK_RIGHT_EYE_OUTER, LANDMARK_RIGHT_EYE_INNER,
        LANDMARK_RIGHT_EYE_TOP, LANDMARK_RIGHT_EYE_BOTTOM,
        LANDMARK_LEFT_EYE_OUTER, LANDMARK_LEFT_EYE_INNER,
        LANDMARK_LEFT_EYE_TOP, LANDMARK_LEFT_EYE_BOTTOM,
        LANDMARK_IRIS_CENTER_OD, LANDMARK_IRIS_CENTER_OS,
    ]
    touches_edge = False
    for idx in ocular_indices:
        x, y = landmarks_px[idx]
        if x <= margin or x >= (width - margin) or y <= margin or y >= (height - margin):
            touches_edge = True
            break

    if touches_edge:
        return QualityGateResult(
            passed=False,
            rejection_reason="OCULAR_REGION_TOUCHES_IMAGE_EDGE",
            rejection_message_vi="Vùng mắt bị sát mép ảnh hoặc mất một phần. Vui lòng căn khuôn mặt vào giữa khung hình và chụp lại.",
            pose=pose_audit,
            audit_details={"edge_margin_px": margin},
        )

    # 4. Kiểm tra độ mở của mắt (Palpebral aperture ratio >= 0.15)
    width_od = float(np.linalg.norm(landmarks_px[LANDMARK_RIGHT_EYE_OUTER] - landmarks_px[LANDMARK_RIGHT_EYE_INNER]))
    height_od = float(np.linalg.norm(landmarks_px[LANDMARK_RIGHT_EYE_TOP] - landmarks_px[LANDMARK_RIGHT_EYE_BOTTOM]))
    ratio_od = height_od / max(width_od, 1.0)

    width_os = float(np.linalg.norm(landmarks_px[LANDMARK_LEFT_EYE_OUTER] - landmarks_px[LANDMARK_LEFT_EYE_INNER]))
    height_os = float(np.linalg.norm(landmarks_px[LANDMARK_LEFT_EYE_TOP] - landmarks_px[LANDMARK_LEFT_EYE_BOTTOM]))
    ratio_os = height_os / max(width_os, 1.0)

    if min(ratio_od, ratio_os) < settings.MIN_EYE_APERTURE_RATIO:
        return QualityGateResult(
            passed=False,
            rejection_reason="EYES_CLOSED",
            rejection_message_vi="Mắt đang nhắm hoặc nheo mi. Vui lòng mở mắt to tự nhiên khi chụp lại.",
            pose=pose_audit,
            aperture_ratio_od=round(ratio_od, 3),
            aperture_ratio_os=round(ratio_os, 3),
            audit_details={
                "aperture_ratio_od": round(ratio_od, 3),
                "aperture_ratio_os": round(ratio_os, 3),
                "min_allowed_ratio": settings.MIN_EYE_APERTURE_RATIO,
            },
        )

    # 5. Kiểm tra độ phân giải vùng mắt và mống mắt
    if min(width_od, width_os) < settings.MIN_EYE_WIDTH_PX:
        return QualityGateResult(
            passed=False,
            rejection_reason="LOW_RESOLUTION",
            rejection_message_vi="Khuôn mặt hoặc mắt quá nhỏ trong khung hình. Vui lòng đưa camera lại gần hơn (khoảng một cánh tay) và chụp lại.",
            pose=pose_audit,
            audit_details={
                "eye_width_od_px": round(width_od, 1),
                "eye_width_os_px": round(width_os, 1),
                "min_allowed_eye_width_px": settings.MIN_EYE_WIDTH_PX,
            },
        )

    iris_diameter_od = float(np.linalg.norm(landmarks_px[LANDMARK_IRIS_EDGE_OD_1] - landmarks_px[LANDMARK_IRIS_EDGE_OD_2]))
    iris_diameter_os = float(np.linalg.norm(landmarks_px[LANDMARK_IRIS_EDGE_OS_1] - landmarks_px[LANDMARK_IRIS_EDGE_OS_2]))

    if min(iris_diameter_od, iris_diameter_os) < settings.MIN_IRIS_DIAMETER_PX:
        return QualityGateResult(
            passed=False,
            rejection_reason="LOW_RESOLUTION",
            rejection_message_vi="Khuôn mặt ở quá xa hoặc vùng mắt không rõ. Vui lòng đưa máy lại gần hơn và chụp lại.",
            pose=pose_audit,
            audit_details={
                "iris_diameter_od_px": round(iris_diameter_od, 1),
                "iris_diameter_os_px": round(iris_diameter_os, 1),
                "min_allowed_iris_diameter_px": settings.MIN_IRIS_DIAMETER_PX,
            },
        )

    # 6. Kiểm tra lóa sáng trên mống mắt
    glare_od = compute_iris_glare_ratio(gray_image, landmarks_px[LANDMARK_IRIS_CENTER_OD], iris_diameter_od / 2.0)
    glare_os = compute_iris_glare_ratio(gray_image, landmarks_px[LANDMARK_IRIS_CENTER_OS], iris_diameter_os / 2.0)
    max_glare = max(glare_od, glare_os)

    if max_glare > settings.MAX_GLARE_RATIO:
        return QualityGateResult(
            passed=False,
            rejection_reason="GLARE_DETECTED",
            rejection_message_vi="Mống mắt bị lóa sáng do ánh đèn hoặc phản quang gắt. Vui lòng tránh nguồn sáng chiếu trực diện và chụp lại.",
            pose=pose_audit,
            glare_ratio=round(max_glare, 3),
            audit_details={
                "glare_ratio_od": round(glare_od, 3),
                "glare_ratio_os": round(glare_os, 3),
                "max_allowed_glare_ratio": settings.MAX_GLARE_RATIO,
            },
        )

    # 7. Kiểm tra độ sắc nét bằng phương sai Laplacian
    sharpness = compute_eye_sharpness(gray_image, landmarks_px, width, height)
    if sharpness < settings.LAPLACIAN_SHARP_THRESHOLD:
        return QualityGateResult(
            passed=False,
            rejection_reason="BLUR_DETECTED",
            rejection_message_vi="Ảnh bị mờ nhòe do rung tay hoặc lấy nét sai. Vui lòng giữ chắc máy và chụp lại.",
            pose=pose_audit,
            sharpness_score=round(sharpness, 2),
            audit_details={
                "sharpness_score": round(sharpness, 2),
                "min_allowed_sharpness": settings.LAPLACIAN_SHARP_THRESHOLD,
            },
        )

    # Đạt toàn bộ tiêu chí chất lượng lâm sàng
    return QualityGateResult(
        passed=True,
        pose=pose_audit,
        sharpness_score=round(sharpness, 2),
        mean_illumination=round(mean_brightness, 2),
        glare_ratio=round(max_glare, 3),
        aperture_ratio_od=round(ratio_od, 3),
        aperture_ratio_os=round(ratio_os, 3),
        audit_details={
            "pose": pose_audit.model_dump(),
            "sharpness_score": round(sharpness, 2),
            "mean_illumination": round(mean_brightness, 2),
            "glare_ratio_od": round(glare_od, 3),
            "glare_ratio_os": round(glare_os, 3),
            "aperture_ratio_od": round(ratio_od, 3),
            "aperture_ratio_os": round(ratio_os, 3),
            "eye_width_od_px": round(width_od, 1),
            "eye_width_os_px": round(width_os, 1),
            "iris_diameter_od_px": round(iris_diameter_od, 1),
            "iris_diameter_os_px": round(iris_diameter_os, 1),
        },
    )


def evaluate_periocular_quality(
    image_bgr: np.ndarray,
    landmarks: LandmarkResult,
) -> QualityGateResult:
    """
    Cổng Sàng lọc Chất lượng cho Ảnh Crop Hai Mắt (Module M2-Crop / Phase D):
    1. Kiểm tra độ phân giải tối thiểu ảnh crop: Chiều cao >= 180 px, Chiều rộng >= 300 px.
    2. Kiểm tra độ sáng xám trung bình toàn khung ảnh (40 <= mean_I <= 220).
    3. Nhận diện mốc giải phẫu vùng mắt.
    4. Ước lượng góc nghiêng 2D (Head Roll) qua vector nối khóe mắt: |Roll| <= 7.0 deg (Risk 3).
    5. Kiểm tra tỷ lệ mở khe mi (Palpebral aperture ratio >= 0.15) phát hiện nhắm mắt.
    6. Kiểm tra độ sắc nét bằng phương sai Laplacian (ngưỡng thích ứng >= 20.0).
    7. Kiểm tra lóa sáng trên đĩa mống mắt.
    8. Nới lỏng kiểm tra biên ảnh (Risk 2).
    """
    if image_bgr is None or not isinstance(image_bgr, np.ndarray) or image_bgr.size == 0:
        return QualityGateResult(
            passed=False,
            rejection_reason="IMAGE_UNREADABLE",
            rejection_message_vi="Không thể đọc dữ liệu ảnh crop. Vui lòng tải lại ảnh hợp lệ.",
            audit_details={"error": "empty_or_invalid_array"},
        )

    height, width = image_bgr.shape[:2]
    min_h = getattr(settings, "MIN_PERIOCULAR_CROP_HEIGHT", 180)
    min_w = getattr(settings, "MIN_PERIOCULAR_CROP_WIDTH", 300)

    if height < min_h or width < min_w:
        return QualityGateResult(
            passed=False,
            rejection_reason="LOW_RESOLUTION",
            rejection_message_vi="Độ phân giải ảnh crop quá nhỏ. Vui lòng chụp hoặc cắt ảnh rõ nét hơn với độ phân giải cao hơn.",
            audit_details={"height": height, "width": width, "min_height_required": min_h, "min_width_required": min_w},
        )

    gray_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    mean_brightness = float(np.mean(gray_image))

    # 1. Kiểm tra độ sáng
    if mean_brightness < settings.MIN_BRIGHTNESS:
        return QualityGateResult(
            passed=False,
            rejection_reason="TOO_DARK",
            rejection_message_vi="Ảnh quá tối, thiếu ánh sáng ở vùng mặt và mắt. Vui lòng bật thêm đèn hoặc chuyển tới nơi đủ sáng và chụp lại.",
            mean_illumination=round(mean_brightness, 2),
            audit_details={"mean_brightness": round(mean_brightness, 2), "min_allowed_brightness": settings.MIN_BRIGHTNESS},
        )

    if mean_brightness > settings.MAX_BRIGHTNESS:
        return QualityGateResult(
            passed=False,
            rejection_reason="TOO_BRIGHT",
            rejection_message_vi="Ảnh quá sáng hoặc bị cháy sáng. Vui lòng giảm bớt ánh sáng chiếu trực tiếp vào mặt và chụp lại.",
            mean_illumination=round(mean_brightness, 2),
            audit_details={"mean_brightness": round(mean_brightness, 2), "max_allowed_brightness": settings.MAX_BRIGHTNESS},
        )

    # 2. Kiểm tra mốc giải phẫu
    if landmarks is None or len(landmarks.landmarks_px) < 478:
        return QualityGateResult(
            passed=False,
            rejection_reason="NO_FACE_DETECTED",
            rejection_message_vi="Không nhận diện được mốc giải phẫu vùng mắt. Vui lòng chụp ảnh rõ nét và đủ ánh sáng.",
            mean_illumination=round(mean_brightness, 2),
            audit_details={"landmarks_count": 0},
        )

    landmarks_px = landmarks.landmarks_px

    # 3. Ước lượng góc nghiêng 2D (Head Roll) per Risk 3:
    # Vector khóe mắt ngoài (33 - 263) và khóe mắt trong (133 - 362)
    p33 = landmarks_px[LANDMARK_RIGHT_EYE_OUTER]
    p263 = landmarks_px[LANDMARK_LEFT_EYE_OUTER]
    p133 = landmarks_px[LANDMARK_RIGHT_EYE_INNER]
    p362 = landmarks_px[LANDMARK_LEFT_EYE_INNER]

    roll_lat = float(np.degrees(np.arctan2(p263[1] - p33[1], p263[0] - p33[0])))
    roll_med = float(np.degrees(np.arctan2(p362[1] - p133[1], p362[0] - p133[0])))
    roll_2d = float((roll_lat + roll_med) / 2.0)

    pose_audit = PoseAudit(yaw=0.0, pitch=0.0, roll=round(roll_2d, 2))
    max_roll = getattr(settings, "PERIOCULAR_MAX_ROLL_DEG", 7.0)

    if abs(roll_2d) > max_roll:
        return QualityGateResult(
            passed=False,
            rejection_reason="HEAD_ROLL_EXCEEDED",
            rejection_message_vi="Đầu hoặc ảnh chụp đang bị nghiêng sang một bên. Vui lòng giữ thẳng máy và chụp lại.",
            pose=pose_audit,
            mean_illumination=round(mean_brightness, 2),
            audit_details={"roll_deg": round(roll_2d, 2), "max_allowed_roll_deg": max_roll},
        )

    # 4. Kiểm tra độ mở của mắt (Palpebral aperture ratio >= 0.15)
    width_od = float(np.linalg.norm(landmarks_px[LANDMARK_RIGHT_EYE_OUTER] - landmarks_px[LANDMARK_RIGHT_EYE_INNER]))
    height_od = float(np.linalg.norm(landmarks_px[LANDMARK_RIGHT_EYE_TOP] - landmarks_px[LANDMARK_RIGHT_EYE_BOTTOM]))
    ratio_od = height_od / max(width_od, 1.0)

    width_os = float(np.linalg.norm(landmarks_px[LANDMARK_LEFT_EYE_OUTER] - landmarks_px[LANDMARK_LEFT_EYE_INNER]))
    height_os = float(np.linalg.norm(landmarks_px[LANDMARK_LEFT_EYE_TOP] - landmarks_px[LANDMARK_LEFT_EYE_BOTTOM]))
    ratio_os = height_os / max(width_os, 1.0)

    if min(ratio_od, ratio_os) < settings.MIN_EYE_APERTURE_RATIO:
        return QualityGateResult(
            passed=False,
            rejection_reason="EYES_CLOSED",
            rejection_message_vi="Mắt đang nhắm hoặc nheo mi. Vui lòng mở mắt to tự nhiên khi chụp lại.",
            pose=pose_audit,
            aperture_ratio_od=round(ratio_od, 3),
            aperture_ratio_os=round(ratio_os, 3),
            mean_illumination=round(mean_brightness, 2),
            audit_details={
                "aperture_ratio_od": round(ratio_od, 3),
                "aperture_ratio_os": round(ratio_os, 3),
                "min_allowed_ratio": settings.MIN_EYE_APERTURE_RATIO,
            },
        )

    # 5. Kiểm tra độ sắc nét bằng phương sai Laplacian toàn ảnh crop
    sharpness = float(cv2.Laplacian(gray_image, cv2.CV_64F).var())
    sharpness_threshold = getattr(settings, "PERIOCULAR_SHARP_THRESHOLD", 20.0)
    if sharpness < sharpness_threshold:
        return QualityGateResult(
            passed=False,
            rejection_reason="BLUR_DETECTED",
            rejection_message_vi="Ảnh bị mờ nhòe do rung tay hoặc lấy nét sai. Vui lòng giữ chắc máy và chụp lại.",
            pose=pose_audit,
            sharpness_score=round(sharpness, 2),
            mean_illumination=round(mean_brightness, 2),
            audit_details={"sharpness_score": round(sharpness, 2), "min_allowed_sharpness": sharpness_threshold},
        )

    # 6. Kiểm tra lóa sáng trên mống mắt
    iris_diameter_od = landmarks.hvid_od_px
    iris_diameter_os = landmarks.hvid_os_px
    glare_od = compute_iris_glare_ratio(gray_image, landmarks_px[LANDMARK_IRIS_CENTER_OD], iris_diameter_od / 2.0)
    glare_os = compute_iris_glare_ratio(gray_image, landmarks_px[LANDMARK_IRIS_CENTER_OS], iris_diameter_os / 2.0)
    max_glare = max(glare_od, glare_os)

    if max_glare > settings.MAX_GLARE_RATIO:
        return QualityGateResult(
            passed=False,
            rejection_reason="GLARE_DETECTED",
            rejection_message_vi="Mống mắt bị lóa sáng do ánh đèn hoặc phản quang gắt. Vui lòng tránh nguồn sáng chiếu trực diện và chụp lại.",
            pose=pose_audit,
            glare_ratio=round(max_glare, 3),
            sharpness_score=round(sharpness, 2),
            mean_illumination=round(mean_brightness, 2),
            audit_details={
                "glare_ratio_od": round(glare_od, 3),
                "glare_ratio_os": round(glare_os, 3),
                "max_allowed_glare_ratio": settings.MAX_GLARE_RATIO,
            },
        )

    # 7. Đạt toàn bộ tiêu chuẩn Cổng Chất lượng Periocular
    return QualityGateResult(
        passed=True,
        pose=pose_audit,
        sharpness_score=round(sharpness, 2),
        mean_illumination=round(mean_brightness, 2),
        glare_ratio=round(max_glare, 3),
        aperture_ratio_od=round(ratio_od, 3),
        aperture_ratio_os=round(ratio_os, 3),
        audit_details={
            "pipeline": "PERIOCULAR_PIPELINE",
            "pose": pose_audit.model_dump(),
            "sharpness_score": round(sharpness, 2),
            "mean_illumination": round(mean_brightness, 2),
            "glare_ratio_od": round(glare_od, 3),
            "glare_ratio_os": round(glare_os, 3),
            "aperture_ratio_od": round(ratio_od, 3),
            "aperture_ratio_os": round(ratio_os, 3),
            "eye_width_od_px": round(width_od, 1),
            "eye_width_os_px": round(width_os, 1),
            "iris_diameter_od_px": round(iris_diameter_od, 1),
            "iris_diameter_os_px": round(iris_diameter_os, 1),
        },
    )
