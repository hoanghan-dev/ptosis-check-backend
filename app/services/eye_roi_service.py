"""
Dịch vụ Điều phối và Trích xuất Vùng ROI Mắt Đa Chế Độ (Eye ROI Service - Phase 2).
Hỗ trợ cả 2 chế độ ảnh đầu vào:
1. 'fullface': Trích xuất qua MediaPipe FaceMesh kết hợp giải thuật mesh_eyes với đệm viền an toàn (15% ngang, 50% đứng).
2. 'eye_crop': Trích xuất qua bộ phân tầng Haar Cascade ghép cặp đối xứng ngang (paired_eye_rois)
   kết hợp mô hình học sâu eye_crop_seed.pt (tiny_unet_eye_v3) để tinh chỉnh tâm hộp bao.
Tuân thủ tuyệt đối G2: Xử lý 100% trong bộ nhớ RAM, không ghi đĩa.
Tuân thủ tuyệt đối G4: Không dùng từ ngữ thuộc danh mục cấm.
"""

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
import torch
import torch.nn as nn

from app.core.config import settings

# Khởi tạo đối tượng FaceMesh dùng cho trích xuất ROI
_MP_FACE_MESH = None
_MP_FACE_MESH_CROP = None

try:
    import mediapipe as mp
    _mp_face_mesh = mp.solutions.face_mesh
    _MP_FACE_MESH = _mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
    )
    _MP_FACE_MESH_CROP = _mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.3,
    )
except Exception:
    _MP_FACE_MESH = None
    _MP_FACE_MESH_CROP = None


@dataclass
class EyeRoi:
    """Thông tin hộp bao ROI của một bên mắt."""
    eye_side: str  # "OD" (Mắt phải) hoặc "OS" (Mắt trái)
    bbox: Tuple[int, int, int, int]  # (x0, y0, x1, y1) trong hệ tọa độ ảnh gốc
    bbox_normalized: Tuple[float, float, float, float]  # (x0_norm, y0_norm, x1_norm, y1_norm) [0, 1]
    center_hint: Optional[Tuple[float, float]] = None  # (x_center, y_center) ước tính
    method: str = "UNKNOWN"
    confidence: float = 1.0
    canonical_iris_center: Optional[Tuple[float, float]] = None
    canonical_iris_landmarks: Optional[Dict[str, Tuple[float, float]]] = None
    canonical_iris_circle: Optional[Tuple[Tuple[float, float], float]] = None
    canonical_eyelid_contour: Optional[List[Tuple[float, float]]] = None
    canonical_upper_lid_arc: Optional[List[Tuple[float, float]]] = None
    canonical_lower_lid_arc: Optional[List[Tuple[float, float]]] = None


@dataclass
class BilateralEyeRoiResult:
    """Kết quả định vị và bóc tách cặp ROI hai mắt OD và OS."""
    od: EyeRoi  # Vùng mắt phải (nằm ở phía bên trái ảnh chụp)
    os: EyeRoi  # Vùng mắt trái (nằm ở phía bên phải ảnh chụp)
    detection_method: str
    image_type: str
    is_valid: bool
    status_message: str


def _conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    """Khối tích chập 2 tầng Conv2d + ReLU chuẩn từ mô hình Tiny U-Net V2."""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(),
    )


class EyeCropSeedLocalizer(nn.Module):
    """
    Kiến trúc mạng Tiny U-Net Eye V3 (eye_crop_seed.pt) nạp từ V2.
    Đầu vào: Tensor 5 kênh (3 kênh màu RGB + 2 kênh tọa độ không gian [-1, 1]).
    Đầu ra: Heatmap 2 kênh (kênh 0: đồng tử, kênh 1: phản quang giác mạc) và logits trạng thái nhìn thấy.
    """
    def __init__(self, targets: int = 2):
        super().__init__()
        self.spatial_logits = True
        self.e1 = _conv_block(5, 16)
        self.e2 = _conv_block(16, 32)
        self.e3 = _conv_block(32, 64)
        self.d2 = _conv_block(96, 32)
        self.d1 = _conv_block(48, 16)
        self.heat = nn.Conv2d(16, targets, 1)
        self.visible = nn.Linear(64, targets * 3)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        height, width = x.shape[-2:]
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, height, device=x.device, dtype=x.dtype),
            torch.linspace(-1, 1, width, device=x.device, dtype=x.dtype),
            indexing="ij",
        )
        coordinates = torch.stack((xx, yy))[None].expand(x.shape[0], -1, -1, -1)
        x = torch.cat((x, coordinates), dim=1)

        a = self.e1(x)
        b = self.e2(nn.functional.max_pool2d(a, 2))
        c = self.e3(nn.functional.max_pool2d(b, 2))

        d = self.d2(torch.cat((
            nn.functional.interpolate(c, size=b.shape[-2:], mode="bilinear", align_corners=False),
            b,
        ), dim=1))
        d = self.d1(torch.cat((
            nn.functional.interpolate(d, size=a.shape[-2:], mode="bilinear", align_corners=False),
            a,
        ), dim=1))

        heat = self.heat(d)
        visible_logits = self.visible(c.mean(dim=(2, 3))).reshape(-1, 2, 3)
        return heat, visible_logits


# Bộ nhớ đệm mô hình hạt giống trong RAM
_EYE_CROP_SEED_MODEL: Optional[EyeCropSeedLocalizer] = None


def get_eye_crop_seed_model() -> Optional[EyeCropSeedLocalizer]:
    """Nạp hoặc lấy mô hình EyeCropSeedLocalizer lưu trong RAM."""
    global _EYE_CROP_SEED_MODEL
    if _EYE_CROP_SEED_MODEL is not None:
        return _EYE_CROP_SEED_MODEL

    weights_path = settings.WEIGHTS_DIR / "eye_crop_seed.pt"
    if not weights_path.exists():
        return None

    try:
        model = EyeCropSeedLocalizer(targets=2)
        checkpoint = torch.load(weights_path, map_location=settings.TORCH_DEVICE, weights_only=False)
        state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
        model.load_state_dict(state_dict)
        model.to(settings.TORCH_DEVICE)
        model.eval()
        _EYE_CROP_SEED_MODEL = model
        return _EYE_CROP_SEED_MODEL
    except Exception:
        return None


def _get_cascade_classifier() -> Optional[cv2.CascadeClassifier]:
    """Khởi tạo bộ phân loại Haar Cascade chuyên dụng cho vùng mắt."""
    # Ưu tiên tệp đóng gói trong thư mục weights
    weights_cascade = settings.WEIGHTS_DIR / "haarcascade_eye_tree_eyeglasses.xml"
    if weights_cascade.exists():
        classifier = cv2.CascadeClassifier(str(weights_cascade))
        if not classifier.empty():
            return classifier

    # Dự phòng từ thư viện cv2.data nếu có
    cv2_data = getattr(cv2, "data", None)
    if cv2_data and hasattr(cv2_data, "haarcascades"):
        path = cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml"
        if Path(path).exists():
            classifier = cv2.CascadeClassifier(path)
            if not classifier.empty():
                return classifier

    return None


def mesh_eyes(landmarks, width: int, height: int) -> Tuple[EyeRoi, EyeRoi]:
    """
    Trích xuất hộp bao 2 mắt chuẩn xác từ MediaPipe FaceMesh theo giải thuật mesh_eyes (V2).
    - Mắt phải (OD): Mốc tâm 468, vòng mống mắt (469..472), đường viền khe mi (33..246).
    - Mắt trái (OS): Mốc tâm 473, vòng mống mắt (474..477), đường viền khe mi (362..398).
    - Bảo lưu toàn vẹn mốc giải phẫu phục vụ neo mống mắt và bờ mi chuẩn lâm sàng.
    """
    groups = [
        (
            "OD",
            468,
            (469, 470, 471, 472),
            (33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246),
            (33, 246, 161, 160, 159, 158, 157, 173, 133),  # upper lid arc
            (33, 7, 163, 144, 145, 153, 154, 155, 133),    # lower lid arc
        ),
        (
            "OS",
            473,
            (474, 475, 476, 477),
            (362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398),
            (362, 398, 384, 385, 386, 387, 388, 466, 263),  # upper lid arc
            (362, 382, 381, 380, 374, 373, 390, 249, 263),  # lower lid arc
        ),
    ]

    def point(idx: int) -> Tuple[float, float]:
        return (float(landmarks[idx].x * width), float(landmarks[idx].y * height))

    results = []
    for side, center_idx, ring, contour, upper_arc, lower_arc in groups:
        center = point(center_idx)
        ring_pts = [point(i) for i in ring]
        lid_pts = [point(i) for i in contour]
        upper_pts = [point(i) for i in upper_arc]
        lower_pts = [point(i) for i in lower_arc]

        iris_left = min(ring_pts, key=lambda p: p[0])
        iris_right = max(ring_pts, key=lambda p: p[0])
        iris_top = min(ring_pts, key=lambda p: p[1])
        iris_bottom = max(ring_pts, key=lambda p: p[1])
        canonical_iris_landmarks = {
            "iris_left": iris_left,
            "iris_right": iris_right,
            "iris_top": iris_top,
            "iris_bottom": iris_bottom,
        }

        # Khớp đường tròn mống mắt tối thiểu từ mốc biên
        iris_pts_np = np.array(ring_pts, dtype=np.float32)
        (circle_center, radius) = cv2.minEnclosingCircle(iris_pts_np)
        canonical_iris_circle = ((float(circle_center[0]), float(circle_center[1])), float(radius))

        iris_height = max(1.0, iris_bottom[1] - iris_top[1])

        xs = [p[0] for p in lid_pts]
        ys = [p[1] for p in lid_pts]

        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        dx = (max_x - min_x) * 0.15
        dy = max(max_y - min_y, iris_height) * 0.30

        x0 = max(0, int(round(min_x - dx)))
        y0 = max(0, int(round(min_y - dy)))
        x1 = min(width, int(round(max_x + dx)))
        y1 = min(height, int(round(max_y + dy)))

        bbox = (x0, y0, x1, y1)
        bbox_norm = (x0 / width, y0 / height, x1 / width, y1 / height)

        results.append(EyeRoi(
            eye_side=side,
            bbox=bbox,
            bbox_normalized=bbox_norm,
            center_hint=center,
            method="FACEMESH_MESH_EYES",
            confidence=0.98,
            canonical_iris_center=center,
            canonical_iris_landmarks=canonical_iris_landmarks,
            canonical_iris_circle=canonical_iris_circle,
            canonical_eyelid_contour=lid_pts,
            canonical_upper_lid_arc=upper_pts,
            canonical_lower_lid_arc=lower_pts,
        ))

    # Đảm bảo OD luôn có hoành độ nhỏ hơn OS (OD nằm bên trái ảnh, OS nằm bên phải ảnh)
    results = sorted(results, key=lambda r: r.bbox[0])
    results[0].eye_side = "OD"
    results[1].eye_side = "OS"
    return results[0], results[1]


def paired_eye_rois(image_bgr: np.ndarray) -> Optional[Tuple[Tuple[int, int, int, int], Tuple[int, int, int, int]]]:
    """
    Phát hiện cặp ROI hai mắt độc lập qua Haar Cascade kết hợp sàng lọc hình học đối xứng ngang (V2).
    Trả về: (bbox_od, bbox_os) trong đó bbox = (x0, y0, x1, y1).
    """
    height, width = image_bgr.shape[:2]
    scale = min(1.0, 1280.0 / max(height, width))
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    if scale < 1.0:
        gray = cv2.resize(gray, (round(width * scale), round(height * scale)))

    detector = _get_cascade_classifier()
    if detector is None:
        return None

    # Phát hiện các ứng viên hộp bao mắt
    found = detector.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3, minSize=(20, 20))
    if len(found) < 2:
        return None

    boxes = sorted([list(map(int, b)) for b in found], key=lambda b: b[2] * b[3], reverse=True)[:20]
    pairs = []
    for i, a in enumerate(boxes):
        for b in boxes[i + 1:]:
            left, right = sorted([a, b], key=lambda c: c[0] + c[2] / 2.0)
            ax, ay = left[0] + left[2] / 2.0, left[1] + left[3] / 2.0
            bx, by = right[0] + right[2] / 2.0, right[1] + right[3] / 2.0

            # Điều kiện khoảng cách ngang tối thiểu
            if bx - ax <= max(left[2], right[2]) * 0.5:
                continue
            # Điều kiện chênh lệch độ cao thẳng đứng
            if abs(ay - by) > 0.6 * max(left[3], right[3]):
                continue
            # Điều kiện tương đồng kích thước diện tích
            if min(left[2], right[2]) / max(left[2], right[2]) < 0.4:
                continue

            score = left[2] * left[3] + right[2] * right[3]
            pairs.append((score, left, right))

    if not pairs:
        return None

    _, left, right = max(pairs, key=lambda r: r[0])

    # Chuyển đổi về tọa độ ảnh gốc và lấy 60% chiều cao trung tâm
    def box_to_roi(b: List[int]) -> Tuple[int, int, int, int]:
        x, y, w, h = b
        x0 = max(0, round(x / scale))
        y0 = max(0, round((y + 0.15 * h) / scale))
        x1 = min(width, round((x + w) / scale))
        y1 = min(height, round((y + 0.85 * h) / scale))
        return (x0, y0, x1, y1)

    box_left = box_to_roi(left)
    box_right = box_to_roi(right)
    return box_left, box_right


def refine_eye_bbox_with_seed(
    image_bgr: np.ndarray,
    bbox: Tuple[int, int, int, int],
    eye_side: str,
) -> Tuple[int, int, int, int]:
    """
    Sử dụng mô hình Tiny U-Net (eye_crop_seed.pt) để định vị tâm kỳ vọng
    và căn chỉnh lại hộp bao ROI sao cho tâm nhãn cầu nằm ở trung tâm crop.
    """
    model = get_eye_crop_seed_model()
    if model is None:
        return bbox

    img_h, img_w = image_bgr.shape[:2]
    x0, y0, x1, y1 = bbox
    crop = image_bgr[y0:y1, x0:x1]
    ch, cw = crop.shape[:2]
    if ch < 10 or cw < 10:
        return bbox

    try:
        resized = cv2.resize(crop, (192, 192))
        tensor = torch.from_numpy(resized.transpose(2, 0, 1)).float()[None] / 255.0
        tensor = tensor.to(settings.TORCH_DEVICE)

        with torch.no_grad():
            heat, _ = model(tensor)
            heat = heat.flatten(2).softmax(-1).reshape_as(heat)

        # Kênh 0: vị trí đồng tử
        flat_idx = heat[0, 0].argmax().item()
        peak_x_norm = (flat_idx % 192) / 192.0
        peak_y_norm = (flat_idx // 192) / 192.0

        # Tọa độ thực của tâm trong ảnh gốc
        center_x = x0 + peak_x_norm * cw
        center_y = y0 + peak_y_norm * ch

        # Giới hạn kích thước hộp bao vừa vặn giải phẫu: chiều rộng khoảng 1.1*cw, chiều cao giới hạn để không tràn trán và má
        target_w = int(cw * 1.10)
        target_h = int(min(max(ch * 1.25, target_w * 0.70), img_h * 0.85))
        half_w = target_w // 2
        half_h = target_h // 2

        new_x0 = max(0, int(round(center_x - half_w)))
        new_y0 = max(0, int(round(center_y - half_h)))
        new_x1 = min(img_w, int(round(center_x + half_w)))
        new_y1 = min(img_h, int(round(center_y + half_h)))

        return (new_x0, new_y0, new_x1, new_y1)
    except Exception:
        return bbox


def extract_bilateral_eye_rois(
    image_bgr: np.ndarray,
    image_type: str = "fullface",
) -> BilateralEyeRoiResult:
    """
    Điều phối bóc tách hộp bao 2 mắt chuẩn hóa cho hệ sinh thái PtosisCheck-1.1:
    - Nếu image_type == 'fullface':
        1. Thử giải thuật mesh_eyes qua MediaPipe FaceMesh.
        2. Nếu FaceMesh không tìm thấy mặt (hoặc nghiêng quá lớn), tự động chuyển sang paired_eye_rois.
    - Nếu image_type == 'eye_crop':
        1. Kích hoạt paired_eye_rois (Haar Cascade đối xứng ngang).
        2. Nếu tìm thấy cặp mắt, tinh chỉnh bằng mô hình eye_crop_seed.pt.
        3. Nếu Haar Cascade thất bại, thử FaceMesh với padding phản chiếu (reflection border).
        4. Nếu vẫn không thấy, áp dụng chia đôi đối xứng 2 nửa (bilateral half split) theo tỷ lệ khung hình.
    """
    if image_bgr is None or image_bgr.size == 0:
        empty_roi = EyeRoi("OD", (0, 0, 0, 0), (0.0, 0.0, 0.0, 0.0), confidence=0.0)
        return BilateralEyeRoiResult(
            od=empty_roi,
            os=empty_roi,
            detection_method="FAILED",
            image_type=image_type,
            is_valid=False,
            status_message="Ảnh đầu vào rỗng hoặc không hợp lệ.",
        )

    height, width = image_bgr.shape[:2]

    # 1. Chế độ ảnh toàn mặt (fullface)
    if image_type == "fullface":
        if _MP_FACE_MESH is not None:
            rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            res = _MP_FACE_MESH.process(rgb)
            if res.multi_face_landmarks:
                od_roi, os_roi = mesh_eyes(res.multi_face_landmarks[0].landmark, width, height)
                return BilateralEyeRoiResult(
                    od=od_roi,
                    os=os_roi,
                    detection_method="fullface_facemesh_mesh_eyes",
                    image_type="fullface",
                    is_valid=True,
                    status_message="Trích xuất ROI hai mắt thành công qua MediaPipe FaceMesh.",
                )

        # Dự phòng tự động khi ảnh fullface không tìm thấy khuôn mặt
        cascade_pair = paired_eye_rois(image_bgr)
        if cascade_pair is not None:
            raw_od, raw_os = cascade_pair
            refined_od = refine_eye_bbox_with_seed(image_bgr, raw_od, "OD")
            refined_os = refine_eye_bbox_with_seed(image_bgr, raw_os, "OS")

            od_roi = EyeRoi(
                eye_side="OD",
                bbox=refined_od,
                bbox_normalized=(refined_od[0] / width, refined_od[1] / height, refined_od[2] / width, refined_od[3] / height),
                method="PAIRED_CASCADE_FALLBACK",
                confidence=0.85,
            )
            os_roi = EyeRoi(
                eye_side="OS",
                bbox=refined_os,
                bbox_normalized=(refined_os[0] / width, refined_os[1] / height, refined_os[2] / width, refined_os[3] / height),
                method="PAIRED_CASCADE_FALLBACK",
                confidence=0.85,
            )
            return BilateralEyeRoiResult(
                od=od_roi,
                os=os_roi,
                detection_method="fullface_cascade_fallback",
                image_type="fullface",
                is_valid=True,
                status_message="Trích xuất ROI hai mắt thành công qua bộ phân tầng Cascade dự phòng.",
            )

    # 2. Chế độ ảnh crop cận cảnh 2 mắt (eye_crop)
    if image_type == "eye_crop":
        # Bước A: Thử giải thuật Haar Cascade ghép cặp ngang
        cascade_pair = paired_eye_rois(image_bgr)
        if cascade_pair is not None:
            raw_od, raw_os = cascade_pair
            refined_od = refine_eye_bbox_with_seed(image_bgr, raw_od, "OD")
            refined_os = refine_eye_bbox_with_seed(image_bgr, raw_os, "OS")

            od_roi = EyeRoi(
                eye_side="OD",
                bbox=refined_od,
                bbox_normalized=(refined_od[0] / width, refined_od[1] / height, refined_od[2] / width, refined_od[3] / height),
                method="PAIRED_CASCADE_SEED_REFINED",
                confidence=0.92,
            )
            os_roi = EyeRoi(
                eye_side="OS",
                bbox=refined_os,
                bbox_normalized=(refined_os[0] / width, refined_os[1] / height, refined_os[2] / width, refined_os[3] / height),
                method="PAIRED_CASCADE_SEED_REFINED",
                confidence=0.92,
            )
            return BilateralEyeRoiResult(
                od=od_roi,
                os=os_roi,
                detection_method="eye_crop_paired_cascade_seed",
                image_type="eye_crop",
                is_valid=True,
                status_message="Trích xuất ROI hai mắt cận cảnh thành công qua Cascade và tinh chỉnh hạt giống.",
            )

        # Bước B: Dự phòng bằng MediaPipe FaceMesh kèm đệm phản chiếu (Reflection Padding)
        if _MP_FACE_MESH_CROP is not None:
            pad_top = int(height * 0.50)
            pad_bottom = int(height * 0.50)
            pad_left = int(width * 0.20)
            pad_right = int(width * 0.20)
            padded = cv2.copyMakeBorder(
                image_bgr, pad_top, pad_bottom, pad_left, pad_right,
                borderType=cv2.BORDER_REFLECT,
            )
            ph, pw = padded.shape[:2]
            rgb_padded = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
            res_crop = _MP_FACE_MESH_CROP.process(rgb_padded)

            if res_crop.multi_face_landmarks:
                padded_lms = res_crop.multi_face_landmarks[0].landmark
                # Chuyển đổi mốc về tọa độ ảnh gốc trước khi bóc tách
                class _UnpaddedLandmark:
                    def __init__(self, x, y):
                        self.x = (x * pw - pad_left) / float(width)
                        self.y = (y * ph - pad_top) / float(height)

                unpadded_lms = [_UnpaddedLandmark(lm.x, lm.y) for lm in padded_lms]
                od_roi, os_roi = mesh_eyes(unpadded_lms, width, height)
                return BilateralEyeRoiResult(
                    od=od_roi,
                    os=os_roi,
                    detection_method="eye_crop_facemesh_padded",
                    image_type="eye_crop",
                    is_valid=True,
                    status_message="Trích xuất ROI hai mắt cận cảnh thành công qua FaceMesh đệm viền phản chiếu.",
                )

        # Bước C: Phân chia 2 nửa đối xứng (Bilateral Split Heuristic) nếu ảnh có tỷ lệ khung ngang phù hợp
        aspect_ratio = width / max(height, 1)
        if aspect_ratio >= 1.2:
            half_w = width // 2
            # Hộp bao mắt phải OD: nửa trái ảnh
            raw_od = (int(half_w * 0.1), int(height * 0.1), int(half_w * 0.9), int(height * 0.9))
            # Hộp bao mắt trái OS: nửa phải ảnh
            raw_os = (half_w + int(half_w * 0.1), int(height * 0.1), half_w + int(half_w * 0.9), int(height * 0.9))

            refined_od = refine_eye_bbox_with_seed(image_bgr, raw_od, "OD")
            refined_os = refine_eye_bbox_with_seed(image_bgr, raw_os, "OS")

            od_roi = EyeRoi(
                eye_side="OD",
                bbox=refined_od,
                bbox_normalized=(refined_od[0] / width, refined_od[1] / height, refined_od[2] / width, refined_od[3] / height),
                method="BILATERAL_SPLIT_SEED_REFINED",
                confidence=0.75,
            )
            os_roi = EyeRoi(
                eye_side="OS",
                bbox=refined_os,
                bbox_normalized=(refined_os[0] / width, refined_os[1] / height, refined_os[2] / width, refined_os[3] / height),
                method="BILATERAL_SPLIT_SEED_REFINED",
                confidence=0.75,
            )
            return BilateralEyeRoiResult(
                od=od_roi,
                os=os_roi,
                detection_method="eye_crop_bilateral_split",
                image_type="eye_crop",
                is_valid=True,
                status_message="Trích xuất ROI hai mắt thành công qua phân chia đối xứng và tinh chỉnh hạt giống.",
            )

    # Không xác định được hai mắt
    failed_roi = EyeRoi("OD", (0, 0, 0, 0), (0.0, 0.0, 0.0, 0.0), confidence=0.0)
    return BilateralEyeRoiResult(
        od=failed_roi,
        os=failed_roi,
        detection_method="FAILED",
        image_type=image_type,
        is_valid=False,
        status_message="Không định vị được cặp vùng mắt hợp lệ trên ảnh.",
    )
