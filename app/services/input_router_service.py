"""
Dịch vụ Điều Tuyến Đầu Vào Đa Nhánh (Input Router Service).
Phân loại ảnh đầu vào thành FULL_FACE, DUAL_EYE_CROP hoặc UNSUPPORTED.
Tuân thủ G2: Xử lý 100% trong RAM, không lưu đĩa.
Tuân thủ G4: Tuyệt đối không dùng từ ngữ thuộc danh mục cấm.
"""

from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

_ROUTER_FACE_MESH_INIT_ERROR: Optional[str] = None
try:
    import mediapipe as mp
    mp_face_mesh = mp.solutions.face_mesh
    _ROUTER_FACE_MESH = mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
    )
except Exception as e:
    _ROUTER_FACE_MESH = None
    _ROUTER_FACE_MESH_INIT_ERROR = str(e)


class InputType(str, Enum):
    FULL_FACE = "FULL_FACE"
    DUAL_EYE_CROP = "DUAL_EYE_CROP"
    UNSUPPORTED = "UNSUPPORTED"


# Chỉ số mốc hình học MediaPipe
LANDMARK_NOSE_TIP = 1
LANDMARK_CHIN = 152
LANDMARK_IRIS_CENTER_OD = 468
LANDMARK_IRIS_CENTER_OS = 473
LANDMARK_RIGHT_EYE_OUTER = 33
LANDMARK_LEFT_EYE_OUTER = 263


def _get_canvas_configs(w: int, h: int) -> List[Tuple[int, int, int, int, str]]:
    """
    Sinh danh sách cấu hình Canvas Padding thích ứng (Synthetic Canvas Injection):
    Trả về danh sách tuple: (canvas_width, canvas_height, x_offset, y_offset, config_label).
    """
    cw_a = int(w * 1.4)
    ch_a = int(cw_a * 1.3)
    yo_a = int(ch_a * 0.33)
    xo_a = int((cw_a - w) / 2)

    cw_b = int(w * 1.3)
    ch_b = int(h * 2.2)
    yo_b = int(ch_b * 0.25)
    xo_b = int((cw_b - w) / 2)

    ch_c = int(max(h * 3.0, w * 1.2))
    cw_c = int(w * 1.5)
    yo_c = int(ch_c * 0.35)
    xo_c = int((cw_c - w) / 2)

    return [
        (cw_a, ch_a, xo_a, yo_a, "Face-Proportional"),
        (cw_b, ch_b, xo_b, yo_b, "Compact-Wide"),
        (cw_c, ch_c, xo_c, yo_c, "Height-Proportional"),
    ]


def classify_input_image(image_bgr: np.ndarray) -> Tuple[InputType, Dict[str, Any]]:
    """
    Phân loại dạng ảnh đầu vào thông qua Bộ lọc Đa Tiêu chí (Cascaded Router per Risk 1):
    1. Kiểm tra mảng ảnh hợp lệ trong RAM.
    2. Giai đoạn 1: MediaPipe FaceMesh Probe trực tiếp trên ảnh gốc.
       Nếu nhận diện đủ 478 mốc và cấu trúc khuôn mặt hoàn chỉnh (có cằm, sống mũi) -> FULL_FACE.
    3. Giai đoạn 2: Probe ảnh crop 2 mắt qua Canvas Padding Thích ứng (Synthetic Canvas Injection).
       Nếu nhận diện được 2 mắt nằm trong giới hạn ảnh crop -> DUAL_EYE_CROP.
    4. Giai đoạn 3: Nếu cả 2 đều thất bại -> UNSUPPORTED.
    """
    if image_bgr is None or not isinstance(image_bgr, np.ndarray) or image_bgr.size == 0:
        return InputType.UNSUPPORTED, {"reason": "EMPTY_OR_INVALID_IMAGE"}

    if _ROUTER_FACE_MESH is None:
        return InputType.UNSUPPORTED, {
            "reason": "FACEMESH_NOT_AVAILABLE",
            "detail": _ROUTER_FACE_MESH_INIT_ERROR,
        }

    h, w = image_bgr.shape[:2]
    aspect_ratio = round(w / max(h, 1), 2)
    rgb_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # -------------------------------------------------------------------------
    # GIAI ĐOẠN 1: Probe Toàn Khuôn Mặt (Full-Face Direct Probe)
    # -------------------------------------------------------------------------
    res_direct = _ROUTER_FACE_MESH.process(rgb_image)
    if res_direct.multi_face_landmarks and len(res_direct.multi_face_landmarks) > 0:
        face_lms = res_direct.multi_face_landmarks[0].landmark
        if len(face_lms) >= 478:
            chin_y = face_lms[LANDMARK_CHIN].y * h
            nose_y = face_lms[LANDMARK_NOSE_TIP].y * h
            # Nếu mốc cằm ở dưới sống mũi, xác nhận đây là ảnh toàn mặt hoặc bán thân mặt
            if chin_y > nose_y:
                return InputType.FULL_FACE, {
                    "method": "FACEMESH_DIRECT",
                    "aspect_ratio": aspect_ratio,
                    "landmarks_count": len(face_lms),
                }

    # -------------------------------------------------------------------------
    # GIAI ĐOẠN 2: Probe Ảnh Crop Hai Mắt (Dual-Eye Crop Canvas Injection Probe)
    # -------------------------------------------------------------------------
    mean_skin_color = np.median(image_bgr, axis=(0, 1)).astype(np.uint8)
    configs = _get_canvas_configs(w, h)

    for cw, ch, xo, yo, label in configs:
        ch_eff = max(ch, yo + h + 10)
        cw_eff = max(cw, xo + w + 10)

        canvas = np.full((ch_eff, cw_eff, 3), mean_skin_color, dtype=np.uint8)
        canvas[yo:yo + h, xo:xo + w] = image_bgr

        canvas_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        res_canvas = _ROUTER_FACE_MESH.process(canvas_rgb)

        if res_canvas.multi_face_landmarks and len(res_canvas.multi_face_landmarks) > 0:
            c_lms = res_canvas.multi_face_landmarks[0].landmark
            if len(c_lms) >= 478:
                # Ánh xạ ngược tọa độ tâm mống mắt về hệ quy chiếu ảnh crop
                x_od = c_lms[LANDMARK_IRIS_CENTER_OD].x * cw_eff - xo
                y_od = c_lms[LANDMARK_IRIS_CENTER_OD].y * ch_eff - yo
                x_os = c_lms[LANDMARK_IRIS_CENTER_OS].x * cw_eff - xo
                y_os = c_lms[LANDMARK_IRIS_CENTER_OS].y * ch_eff - yo

                margin_x = w * 0.08
                margin_y = h * 0.15

                od_inside = (-margin_x <= x_od <= w + margin_x) and (-margin_y <= y_od <= h + margin_y)
                os_inside = (-margin_x <= x_os <= w + margin_x) and (-margin_y <= y_os <= h + margin_y)

                # Mắt phải (OD) phải nằm ở phía bên trái ảnh người quan sát (x_od < x_os)
                if od_inside and os_inside and x_od < x_os:
                    return InputType.DUAL_EYE_CROP, {
                        "method": "CANVAS_INJECTION",
                        "canvas_config": label,
                        "aspect_ratio": aspect_ratio,
                        "iris_od": [round(float(x_od), 1), round(float(y_od), 1)],
                        "iris_os": [round(float(x_os), 1), round(float(y_os), 1)],
                        "canvas_shape": [cw_eff, ch_eff],
                        "canvas_offset": [xo, yo],
                    }

    # -------------------------------------------------------------------------
    # GIAI ĐOẠN 3: Không Thể Xác Định Khuôn Mặt Hoặc Hai Mắt
    # -------------------------------------------------------------------------
    return InputType.UNSUPPORTED, {
        "reason": "NO_FACE_OR_EYES_DETECTED",
        "aspect_ratio": aspect_ratio,
    }
