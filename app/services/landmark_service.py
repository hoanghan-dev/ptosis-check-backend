"""
Dịch vụ Định vị Mốc Mống mắt & Xác lập Thước đo HVID (Module M3).
Sử dụng MediaPipe FaceMesh với refine_landmarks=True để trích xuất:
- Tâm mống mắt (mốc 468, 473)
- Rìa ngang limbus (mốc 469/471 cho OD, mốc 474/476 cho OS)
- Mốc bờ mi trên, mi dưới, cung mày
- Tính toán HVID_OD, HVID_OS và hệ số quy đổi vật lý S_iris (mm/px).
Tuyệt đối tuân thủ G2: Xử lý 100% trong RAM, không lưu đĩa.
Tuyệt đối tuân thủ G4: Không dùng từ ngữ thuộc danh mục cấm.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import cv2
import numpy as np

_LANDMARK_FACE_MESH_INIT_ERROR: Optional[str] = None
try:
    import mediapipe as mp
    mp_face_mesh = mp.solutions.face_mesh
    _LANDMARK_FACE_MESH = mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
    )
except Exception as e:
    _LANDMARK_FACE_MESH = None
    _LANDMARK_FACE_MESH_INIT_ERROR = str(e)

from app.core.config import settings
from app.utils.geometry import euclidean_distance
from app.utils.mediapipe_lock import mediapipe_lock

# Mốc mống mắt (Iris landmarks)
LANDMARK_IRIS_CENTER_OD = 468
LANDMARK_IRIS_EDGE_OD_LATERAL = 469
LANDMARK_IRIS_EDGE_OD_MEDIAL = 471

LANDMARK_IRIS_CENTER_OS = 473
LANDMARK_IRIS_EDGE_OS_MEDIAL = 474
LANDMARK_IRIS_EDGE_OS_LATERAL = 476

# Mốc bờ mi trên (Upper eyelid margin landmarks)
# OD: khóe ngoài 33 -> 160 -> 158 -> đỉnh bờ mi 159 -> 157 -> 173 -> khóe trong 133
LANDMARKS_UPPER_EYELID_OD = [33, 160, 158, 159, 157, 173, 133]
# OS: khóe trong 362 -> 384 -> 385 -> đỉnh bờ mi 386 -> 387 -> 388 -> khóe ngoài 263
LANDMARKS_UPPER_EYELID_OS = [362, 384, 385, 386, 387, 388, 263]

# Mốc bờ mi dưới (Lower eyelid margin landmarks)
LANDMARKS_LOWER_EYELID_OD = [33, 144, 145, 153, 133]
LANDMARKS_LOWER_EYELID_OS = [362, 380, 374, 373, 263]

# Mốc cung mày (Eyebrow landmarks)
LANDMARKS_EYEBROW_OD = [70, 63, 105, 66, 107]
LANDMARKS_EYEBROW_OS = [336, 296, 334, 293, 300]


@dataclass
class LandmarkResult:
    """Kết quả trích xuất mốc hình học và thước đo mống mắt."""
    landmarks_px: np.ndarray
    iris_center_od: Tuple[float, float]
    iris_center_os: Tuple[float, float]
    limbus_horizontal_od: Tuple[Tuple[float, float], Tuple[float, float]]
    limbus_horizontal_os: Tuple[Tuple[float, float], Tuple[float, float]]
    hvid_od_px: float
    hvid_os_px: float
    scale_factor_od: float
    scale_factor_os: float
    hvid_asymmetry_ratio: float
    is_hvid_biologically_consistent: bool
    upper_eyelid_od: List[Tuple[float, float]]
    lower_eyelid_od: List[Tuple[float, float]]
    upper_eyelid_os: List[Tuple[float, float]]
    lower_eyelid_os: List[Tuple[float, float]]
    eyebrow_od: List[Tuple[float, float]]
    eyebrow_os: List[Tuple[float, float]]
    warning_flags: List[str] = field(default_factory=list)


def detect_landmarks(image_bgr: np.ndarray) -> LandmarkResult:
    """
    Trích xuất mốc hình học khuôn mặt và tính toán thước đo mống mắt ngang (HVID):
    - Input: Mảng ảnh BGR hợp lệ trong RAM.
    - Output: LandmarkResult chứa tọa độ sub-pixel và hệ số quy đổi milimét.
    """
    if image_bgr is None or not isinstance(image_bgr, np.ndarray) or image_bgr.size == 0:
        raise ValueError("Dữ liệu ảnh đầu vào không hợp lệ hoặc rỗng.")

    if _LANDMARK_FACE_MESH is None:
        detail = f" Chi tiết lỗi: {_LANDMARK_FACE_MESH_INIT_ERROR}" if _LANDMARK_FACE_MESH_INIT_ERROR else ""
        raise RuntimeError(f"Module MediaPipe FaceMesh chưa được khởi tạo thành công.{detail}")

    height, width = image_bgr.shape[:2]
    rgb_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    with mediapipe_lock:
        results = _LANDMARK_FACE_MESH.process(rgb_image)

    if not results.multi_face_landmarks or len(results.multi_face_landmarks) == 0:
        raise ValueError("Không phát hiện được khuôn mặt trong ảnh.")

    face_landmarks = results.multi_face_landmarks[0].landmark
    if len(face_landmarks) < 478:
        raise ValueError(
            f"Khuôn mặt chỉ phát hiện được {len(face_landmarks)} mốc, thiếu mốc mống mắt (yêu cầu 478 mốc)."
        )

    # Chuyển đổi toàn bộ mốc sang tọa độ pixel tuyệt đối
    landmarks_px = np.array(
        [[float(lm.x) * float(width), float(lm.y) * float(height)] for lm in face_landmarks],
        dtype=np.float64,
    )

    # 1. Trích xuất tâm mống mắt (mốc 468 cho OD, mốc 473 cho OS)
    iris_center_od = (float(landmarks_px[LANDMARK_IRIS_CENTER_OD, 0]), float(landmarks_px[LANDMARK_IRIS_CENTER_OD, 1]))
    iris_center_os = (float(landmarks_px[LANDMARK_IRIS_CENTER_OS, 0]), float(landmarks_px[LANDMARK_IRIS_CENTER_OS, 1]))

    # 2. Trích xuất rìa mống mắt ngang (limbus)
    limbus_pt_od_1 = (float(landmarks_px[LANDMARK_IRIS_EDGE_OD_LATERAL, 0]), float(landmarks_px[LANDMARK_IRIS_EDGE_OD_LATERAL, 1]))
    limbus_pt_od_2 = (float(landmarks_px[LANDMARK_IRIS_EDGE_OD_MEDIAL, 0]), float(landmarks_px[LANDMARK_IRIS_EDGE_OD_MEDIAL, 1]))
    limbus_horizontal_od = (limbus_pt_od_1, limbus_pt_od_2)

    limbus_pt_os_1 = (float(landmarks_px[LANDMARK_IRIS_EDGE_OS_MEDIAL, 0]), float(landmarks_px[LANDMARK_IRIS_EDGE_OS_MEDIAL, 1]))
    limbus_pt_os_2 = (float(landmarks_px[LANDMARK_IRIS_EDGE_OS_LATERAL, 0]), float(landmarks_px[LANDMARK_IRIS_EDGE_OS_LATERAL, 1]))
    limbus_horizontal_os = (limbus_pt_os_1, limbus_pt_os_2)

    # 3. Tính HVID hai bên mắt
    hvid_od_px = euclidean_distance(limbus_pt_od_1, limbus_pt_od_2)
    hvid_os_px = euclidean_distance(limbus_pt_os_1, limbus_pt_os_2)

    if hvid_od_px <= 0 or hvid_os_px <= 0:
        raise ValueError("Đường kính mống mắt trích xuất không hợp lệ (<= 0 px).")

    # 4. Tính hệ số quy đổi milimét nội sinh S_iris = 11.71 / HVID
    assumed_diameter_mm = settings.ASSUMED_CORNEAL_DIAMETER_MM
    scale_factor_od = assumed_diameter_mm / hvid_od_px
    scale_factor_os = assumed_diameter_mm / hvid_os_px

    # 5. Kiểm tra nhất quán sinh học hai mắt theo công thức V2: |HVID_OD - HVID_OS| / max(OD, OS) <= 15%
    max_hvid = max(hvid_od_px, hvid_os_px)
    hvid_asymmetry_ratio = abs(hvid_od_px - hvid_os_px) / max(max_hvid, 1e-6)
    is_hvid_biologically_consistent = hvid_asymmetry_ratio <= settings.IRIS_MAX_BILATERAL_DIFF_RATIO

    warning_flags = []
    if not is_hvid_biologically_consistent:
        warning_flags.append("HVID_ASYMMETRY_EXCEEDED")

    # Kiểm tra tính khả thi của hệ số quy đổi (hỗ trợ cả ảnh crop cận cảnh và ảnh toàn mặt)
    min_scale = getattr(settings, "PERIOCULAR_MIN_SCALE_FACTOR", 0.015)
    max_scale = getattr(settings, "PERIOCULAR_MAX_SCALE_FACTOR", 0.60)
    if not (min_scale <= scale_factor_od <= max_scale and min_scale <= scale_factor_os <= max_scale):
        warning_flags.append("SCALE_FACTOR_OUTSIDE_EXPECTED_RANGE")

    # 6. Trích xuất đường viền bờ mi trên, mi dưới, cung mày
    upper_eyelid_od = [(float(landmarks_px[i, 0]), float(landmarks_px[i, 1])) for i in LANDMARKS_UPPER_EYELID_OD]
    lower_eyelid_od = [(float(landmarks_px[i, 0]), float(landmarks_px[i, 1])) for i in LANDMARKS_LOWER_EYELID_OD]

    upper_eyelid_os = [(float(landmarks_px[i, 0]), float(landmarks_px[i, 1])) for i in LANDMARKS_UPPER_EYELID_OS]
    lower_eyelid_os = [(float(landmarks_px[i, 0]), float(landmarks_px[i, 1])) for i in LANDMARKS_LOWER_EYELID_OS]

    eyebrow_od = [(float(landmarks_px[i, 0]), float(landmarks_px[i, 1])) for i in LANDMARKS_EYEBROW_OD]
    eyebrow_os = [(float(landmarks_px[i, 0]), float(landmarks_px[i, 1])) for i in LANDMARKS_EYEBROW_OS]

    return LandmarkResult(
        landmarks_px=landmarks_px,
        iris_center_od=iris_center_od,
        iris_center_os=iris_center_os,
        limbus_horizontal_od=limbus_horizontal_od,
        limbus_horizontal_os=limbus_horizontal_os,
        hvid_od_px=round(hvid_od_px, 2),
        hvid_os_px=round(hvid_os_px, 2),
        scale_factor_od=round(scale_factor_od, 4),
        scale_factor_os=round(scale_factor_os, 4),
        hvid_asymmetry_ratio=round(hvid_asymmetry_ratio, 4),
        is_hvid_biologically_consistent=is_hvid_biologically_consistent,
        upper_eyelid_od=upper_eyelid_od,
        lower_eyelid_od=lower_eyelid_od,
        upper_eyelid_os=upper_eyelid_os,
        lower_eyelid_os=lower_eyelid_os,
        eyebrow_od=eyebrow_od,
        eyebrow_os=eyebrow_os,
        warning_flags=warning_flags,
    )
