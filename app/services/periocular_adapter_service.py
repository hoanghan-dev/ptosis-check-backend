"""
Dịch vụ Tiếp Hợp Vùng Quanh Mắt (Periocular Adapter Service - Module M3-Crop).
Xử lý ảnh crop 2 mắt qua kỹ thuật Synthetic Canvas Injection:
- Đặt ROI hai mắt vào canvas nhân tạo mang giá trị trung bình màu da.
- Chạy MediaPipe FaceMesh với refine_landmarks=True để trích xuất 478 mốc.
- Ánh xạ ngược toàn bộ tọa độ mốc về hệ tọa độ gốc của ảnh crop.
- Tính toán HVID_OD, HVID_OS và hệ số quy đổi milimét nội sinh S_iris.
Tuân thủ G2: Xử lý 100% trong RAM, không lưu đĩa.
Tuân thủ G4: Tuyệt đối không dùng từ ngữ thuộc danh mục cấm.
"""

from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

_PERIOCULAR_FACE_MESH_INIT_ERROR: Optional[str] = None
try:
    import mediapipe as mp
    mp_face_mesh = mp.solutions.face_mesh
    _PERIOCULAR_FACE_MESH = mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
    )
except Exception as e:
    _PERIOCULAR_FACE_MESH = None
    _PERIOCULAR_FACE_MESH_INIT_ERROR = str(e)

from app.core.config import settings
from app.services.input_router_service import _get_canvas_configs
from app.services.landmark_service import (
    LANDMARK_IRIS_CENTER_OD,
    LANDMARK_IRIS_CENTER_OS,
    LANDMARK_IRIS_EDGE_OD_LATERAL,
    LANDMARK_IRIS_EDGE_OD_MEDIAL,
    LANDMARK_IRIS_EDGE_OS_MEDIAL,
    LANDMARK_IRIS_EDGE_OS_LATERAL,
    LANDMARKS_UPPER_EYELID_OD,
    LANDMARKS_UPPER_EYELID_OS,
    LANDMARKS_LOWER_EYELID_OD,
    LANDMARKS_LOWER_EYELID_OS,
    LANDMARKS_EYEBROW_OD,
    LANDMARKS_EYEBROW_OS,
    LandmarkResult,
)
from app.utils.geometry import euclidean_distance


def process_periocular_crop(
    image_bgr: np.ndarray,
    router_meta: Optional[Dict[str, Any]] = None,
) -> LandmarkResult:
    """
    Trích xuất mốc giải phẫu từ ảnh crop hai mắt thông qua Synthetic Canvas Injection.

    Args:
        image_bgr: Mảng ảnh crop BGR hợp lệ trong RAM.
        router_meta: Thông tin cấu hình từ InputRouter (nếu có).

    Returns:
        LandmarkResult với tọa độ chuẩn hóa về không gian pixel ảnh crop ban đầu.
    """
    if image_bgr is None or not isinstance(image_bgr, np.ndarray) or image_bgr.size == 0:
        raise ValueError("Dữ liệu ảnh crop đầu vào không hợp lệ hoặc rỗng.")

    if _PERIOCULAR_FACE_MESH is None:
        detail = f" Chi tiết lỗi: {_PERIOCULAR_FACE_MESH_INIT_ERROR}" if _PERIOCULAR_FACE_MESH_INIT_ERROR else ""
        raise RuntimeError(f"Module MediaPipe FaceMesh chưa được khởi tạo thành công.{detail}")

    h, w = image_bgr.shape[:2]
    mean_skin_color = np.median(image_bgr, axis=(0, 1)).astype(np.uint8)

    # Lấy danh sách cấu hình canvas thích ứng
    configs = _get_canvas_configs(w, h)

    # Nếu router_meta đã tìm thấy cấu hình tối ưu trước đó, ưu tiên thử cấu hình đó đầu tiên
    preferred_config = router_meta.get("canvas_config") if router_meta else None
    if preferred_config:
        configs.sort(key=lambda c: 0 if c[4] == preferred_config else 1)

    face_landmarks = None
    cw_final = 0
    ch_final = 0
    xo_final = 0
    yo_final = 0

    for cw, ch, xo, yo, label in configs:
        ch_eff = max(ch, yo + h + 10)
        cw_eff = max(cw, xo + w + 10)

        canvas = np.full((ch_eff, cw_eff, 3), mean_skin_color, dtype=np.uint8)
        canvas[yo:yo + h, xo:xo + w] = image_bgr

        canvas_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        results = _PERIOCULAR_FACE_MESH.process(canvas_rgb)

        if results.multi_face_landmarks and len(results.multi_face_landmarks) > 0:
            lms = results.multi_face_landmarks[0].landmark
            if len(lms) >= 478:
                face_landmarks = lms
                cw_final = cw_eff
                ch_final = ch_eff
                xo_final = xo
                yo_final = yo
                break

    if face_landmarks is None:
        raise ValueError("Không thể nhận diện mốc giải phẫu hai mắt qua canvas thích ứng.")

    # Ánh xạ ngược 478 mốc về hệ tọa độ gốc của ảnh crop (x_crop, y_crop)
    landmarks_px = np.array(
        [
            [
                float(lm.x) * float(cw_final) - float(xo_final),
                float(lm.y) * float(ch_final) - float(yo_final),
            ]
            for lm in face_landmarks
        ],
        dtype=np.float64,
    )

    # 1. Tâm mống mắt (mốc 468 cho OD, mốc 473 cho OS)
    iris_center_od = (
        float(landmarks_px[LANDMARK_IRIS_CENTER_OD, 0]),
        float(landmarks_px[LANDMARK_IRIS_CENTER_OD, 1]),
    )
    iris_center_os = (
        float(landmarks_px[LANDMARK_IRIS_CENTER_OS, 0]),
        float(landmarks_px[LANDMARK_IRIS_CENTER_OS, 1]),
    )

    # 2. Rìa mống mắt ngang (limbus)
    limbus_pt_od_1 = (
        float(landmarks_px[LANDMARK_IRIS_EDGE_OD_LATERAL, 0]),
        float(landmarks_px[LANDMARK_IRIS_EDGE_OD_LATERAL, 1]),
    )
    limbus_pt_od_2 = (
        float(landmarks_px[LANDMARK_IRIS_EDGE_OD_MEDIAL, 0]),
        float(landmarks_px[LANDMARK_IRIS_EDGE_OD_MEDIAL, 1]),
    )
    limbus_horizontal_od = (limbus_pt_od_1, limbus_pt_od_2)

    limbus_pt_os_1 = (
        float(landmarks_px[LANDMARK_IRIS_EDGE_OS_MEDIAL, 0]),
        float(landmarks_px[LANDMARK_IRIS_EDGE_OS_MEDIAL, 1]),
    )
    limbus_pt_os_2 = (
        float(landmarks_px[LANDMARK_IRIS_EDGE_OS_LATERAL, 0]),
        float(landmarks_px[LANDMARK_IRIS_EDGE_OS_LATERAL, 1]),
    )
    limbus_horizontal_os = (limbus_pt_os_1, limbus_pt_os_2)

    # 3. Tính HVID hai bên mắt
    hvid_od_px = euclidean_distance(limbus_pt_od_1, limbus_pt_od_2)
    hvid_os_px = euclidean_distance(limbus_pt_os_1, limbus_pt_os_2)

    if hvid_od_px <= 0 or hvid_os_px <= 0:
        raise ValueError("Đường kính mống mắt trích xuất từ ảnh crop không hợp lệ (<= 0 px).")

    # 4. Tính hệ số quy đổi milimét nội sinh S_iris = 11.71 / HVID
    assumed_diameter_mm = settings.ASSUMED_CORNEAL_DIAMETER_MM
    scale_factor_od = assumed_diameter_mm / hvid_od_px
    scale_factor_os = assumed_diameter_mm / hvid_os_px

    # 5. Kiểm tra nhất quán sinh học hai mắt theo công thức V2: |HVID_OD - HVID_OS| / max(OD, OS) <= 15%
    max_hvid = max(hvid_od_px, hvid_os_px)
    hvid_asymmetry_ratio = abs(hvid_od_px - hvid_os_px) / max(max_hvid, 1e-6)
    is_hvid_biologically_consistent = hvid_asymmetry_ratio <= settings.IRIS_MAX_BILATERAL_DIFF_RATIO

    warning_flags = ["PROCESSED_VIA_PERIOCULAR_PIPELINE"]
    if not is_hvid_biologically_consistent:
        warning_flags.append("HVID_ASYMMETRY_EXCEEDED")

    # Kiểm tra thang đo mm/px (cho phép dải đo rộng hơn trên ảnh crop cận cảnh: 0.015 - 0.60 mm/px)
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
