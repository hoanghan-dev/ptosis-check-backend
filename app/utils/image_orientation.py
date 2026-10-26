"""
Tiện ích Tự Động Định Hướng Ảnh (Module M1 / Phase B: EXIF & 4-Direction Auto-Orientation).
Thực hiện toàn bộ quá trình giải mã, trích xuất EXIF và xoay 4 hướng (Probe 4 directions)
100% trong bộ nhớ RAM tạm thời (In-Memory Zero Retention - G2 Compliance).
Tuyệt đối không ghi tệp trung gian hay byte đệm ra đĩa.
"""

import io
from typing import Any, Dict, Optional, Tuple
import cv2
import numpy as np
from PIL import Image, ImageOps

from app.core.config import settings
from app.services.quality_gate_service import estimate_head_pose_3d

# Khởi tạo singleton MediaPipe FaceMesh probe trong bộ nhớ
_PROBE_FACE_MESH = None
_PROBE_FACE_MESH_INIT_ERROR: Optional[str] = None

try:
    import mediapipe as mp

    mp_face_mesh = mp.solutions.face_mesh
    _PROBE_FACE_MESH = mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
    )
except Exception as e:
    _PROBE_FACE_MESH = None
    _PROBE_FACE_MESH_INIT_ERROR = str(e)


# Các chỉ số mốc giải phẫu phục vụ kiểm tra thăng bằng khóe mắt (Canthus Roll)
LANDMARK_RIGHT_EYE_OUTER = 33  # Đuôi mắt phải (bên trái theo góc nhìn ảnh)
LANDMARK_LEFT_EYE_OUTER = 263  # Đuôi mắt trái (bên phải theo góc nhìn ảnh)


def decode_and_orient_image(contents: bytes) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    """
    Giải mã luồng byte ảnh tải lên và tự động xoay về hướng đứng tự nhiên:
    1. Ưu tiên giải mã thẻ xoay EXIF Orientation (Tags 1-8) in-memory bằng Pillow ImageOps.
    2. Fallback xoay 4 hướng (0°, 90°, 270°, 180°): Nếu không có EXIF hoặc ảnh sau giải mã
       không phát hiện khuôn mặt nằm ngang ở 0°, chạy FaceMesh probe tìm góc xoay tối ưu
       thỏa mãn độ nghiêng khóe mắt |canthus_roll| <= 15° và tư thế đầu chuẩn xác.

    Args:
        contents: Luồng byte nhị phân của tệp ảnh từ Multipart upload.

    Returns:
        Tuple chứa:
          - image_bgr: Ma trận pixel BGR OpenCV sau khi đã xoay chuẩn hóa (hoặc None nếu lỗi giải mã).
          - orientation_info: Từ điển chứa applied_rotation_deg, method, exif_orientation, canthus_roll_deg.
    """
    applied_rotation_deg = 0
    method = "NONE"
    exif_orientation = None

    if not contents or len(contents) == 0:
        return None, {
            "applied_rotation_deg": 0,
            "method": "EMPTY_INPUT",
            "exif_orientation": None,
        }

    # -------------------------------------------------------------------------
    # Bước 1: Giải mã và Xử lý Thẻ EXIF Orientation Trong RAM (In-Memory)
    # -------------------------------------------------------------------------
    try:
        pil_img = Image.open(io.BytesIO(contents))
        exif = pil_img.getexif()
        if exif:
            exif_orientation = exif.get(0x0112)  # Tag 0x0112: EXIF Orientation
            if exif_orientation is not None and exif_orientation != 1:
                transposed = ImageOps.exif_transpose(pil_img)
                if exif_orientation == 6:
                    applied_rotation_deg = 90
                elif exif_orientation == 3:
                    applied_rotation_deg = 180
                elif exif_orientation == 8:
                    applied_rotation_deg = 270
                method = "EXIF"
                pil_img = transposed

        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")
        image_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    except Exception:
        # Fallback giải mã trực tiếp bằng OpenCV buffer nếu Pillow không đọc được header
        nparr = np.frombuffer(contents, np.uint8)
        image_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image_bgr is None:
            return None, {
                "applied_rotation_deg": 0,
                "method": "DECODE_FAILED",
                "exif_orientation": None,
            }

    # Nếu cấu hình tắt Auto-Orientation -> Trả về ma trận hiện tại ngay
    auto_orient = getattr(settings, "AUTO_ORIENTATION_ENABLED", True)
    if not auto_orient or _PROBE_FACE_MESH is None:
        return image_bgr, {
            "applied_rotation_deg": applied_rotation_deg,
            "method": method,
            "exif_orientation": exif_orientation,
        }

    # -------------------------------------------------------------------------
    # Bước 2: Chốt Chặn Fallback Thử Xoay 4 Hướng (Probe 4 Directions)
    # -------------------------------------------------------------------------
    max_roll_threshold = getattr(settings, "MAX_ORIENTATION_PROBE_ROLL_DEG", 15.0)

    def _evaluate_candidate(img: np.ndarray) -> Tuple[bool, Optional[float], float]:
        """Đánh giá tư thế đầu và góc nghiêng khóe mắt của một ảnh ứng viên."""
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = _PROBE_FACE_MESH.process(rgb)
        if not res.multi_face_landmarks:
            return False, None, float("inf")

        lm = res.multi_face_landmarks[0].landmark
        h, w = img.shape[:2]
        landmarks_px = np.array([[pt.x * w, pt.y * h] for pt in lm], dtype=np.float64)

        p33 = landmarks_px[LANDMARK_RIGHT_EYE_OUTER]
        p263 = landmarks_px[LANDMARK_LEFT_EYE_OUTER]

        # Kiểm tra thứ tự mắt: mắt trái người xem (p33) phải ở bên trái mắt phải người xem (p263)
        if p263[0] <= p33[0]:
            return False, None, float("inf")

        pitch, yaw, roll = estimate_head_pose_3d(landmarks_px, w, h)

        if abs(roll) <= max_roll_threshold and abs(pitch) <= 15.0 and abs(yaw) <= 20.0:
            score = abs(roll) * 2.0 + abs(pitch) + abs(yaw)
            return True, float(roll), score
        return False, float(roll), float("inf")

    # Kiểm tra góc 0 độ trước
    is_valid_0, roll_0, score_0 = _evaluate_candidate(image_bgr)
    if is_valid_0:
        return image_bgr, {
            "applied_rotation_deg": applied_rotation_deg,
            "method": method if method != "NONE" else "NONE",
            "exif_orientation": exif_orientation,
            "canthus_roll_deg": roll_0,
        }

    # Nếu góc 0 độ không đạt, thử các góc còn lại (90°, 270°, 180°)
    probe_candidates = [
        (90, cv2.ROTATE_90_CLOCKWISE),
        (270, cv2.ROTATE_90_COUNTERCLOCKWISE),
        (180, cv2.ROTATE_180),
    ]

    best_candidate_deg = 0
    best_img = image_bgr
    best_roll = None
    best_score = float("inf")

    for candidate_deg, cv2_code in probe_candidates:
        candidate_img = cv2.rotate(image_bgr, cv2_code)
        is_valid, roll, score = _evaluate_candidate(candidate_img)
        if is_valid and score < best_score:
            best_score = score
            best_candidate_deg = candidate_deg
            best_img = candidate_img
            best_roll = roll

    if best_score < float("inf"):
        total_rotation = (applied_rotation_deg + best_candidate_deg) % 360
        return best_img, {
            "applied_rotation_deg": total_rotation,
            "method": "PROBE",
            "exif_orientation": exif_orientation,
            "canthus_roll_deg": best_roll,
        }

    # Nếu không góc nào thỏa mãn tiêu chuẩn, giữ nguyên ảnh ban đầu
    return image_bgr, {
        "applied_rotation_deg": applied_rotation_deg,
        "method": method,
        "exif_orientation": exif_orientation,
    }
