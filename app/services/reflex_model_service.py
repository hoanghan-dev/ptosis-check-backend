"""
Dịch vụ Định vị Phản quang Giác mạc Purkinje I & Tâm Đồng tử Dự phòng (Reflex Model Service - Phase 5).
Kế thừa kiến trúc mạng nơ-ron học sâu và bộ lọc quang học từ V2:
1. Mạng FCN pr_model (reflex_localizer.pt, 256x256):
   - Dự đoán bản đồ nhiệt không gian cho phản xạ giác mạc Purkinje I.
2. Mạng DirectPupilCNN (pupil_direct.pt, 256x256) kết hợp hàm tiên nghiệm mống mắt mềm (soft_prior):
   - Giải mã tâm kỳ vọng expected_center cho vị trí tâm đồng tử độc lập.
3. Bộ lọc quang học reflex_roi_qc:
   - Ràng buộc vùng tìm kiếm trong giao của đĩa mống mắt + mask khe mi EyelidUNet + bán kính <= 0.4 * R_iris.
   - Lọc vệt lóa kéo dài (aspect ratio <= 2.0), tính trọng tâm cường độ sáng sub-pixel (I > 230).
4. Thiết lập chuỗi dự phòng 3 cấp chuẩn hóa (Fallback Chain):
   - Cấp 1 (Chuẩn vàng): Phản xạ Purkinje I -> reference_point_used = "CORNEAL_LIGHT_REFLEX"
   - Cấp 2 (Dự phòng đồng tử): Tâm đồng tử độc lập -> reference_point_used = "PUPIL_CENTER_FALLBACK", fallback_type = "PUPIL_DETECTOR"
   - Cấp 3 (Dự phòng mống mắt): Tâm mống mắt -> reference_point_used = "PUPIL_CENTER_FALLBACK", fallback_type = "IRIS_CENTER"
Tuân thủ tuyệt đối G2: Xử lý 100% trong RAM, không ghi đĩa.
Tuân thủ tuyệt đối G4: Không dùng từ ngữ thuộc danh mục cấm.
"""

from dataclasses import dataclass, field
import math
from typing import List, Optional, Tuple
import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from app.core.config import settings
from app.services.eyelid_model_service import EyelidSegmentationResult
from app.services.iris_model_service import IrisGeometryResult


def _conv_block(in_c: int, out_c: int) -> nn.Sequential:
    """Khối tích chập 2 tầng Conv2d + ReLU chuẩn."""
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(out_c, out_c, 3, padding=1),
        nn.ReLU(),
    )


class LocalizerFCN(nn.Module):
    """
    Kiến trúc mạng FCN bản đồ nhiệt không gian 256x256:
    Dùng cho cả reflex_localizer.pt (in_channels=5) và pupil_direct.pt (in_channels=6).
    """
    def __init__(self, in_channels: int = 5, targets: int = 2):
        super().__init__()
        self.in_channels = in_channels
        self.targets = targets
        self.e1 = _conv_block(in_channels, 16)
        self.e2 = _conv_block(16, 32)
        self.e3 = _conv_block(32, 64)
        self.d2 = _conv_block(96, 32)
        self.d1 = _conv_block(48, 16)
        self.heat = nn.Conv2d(16, targets, 1)
        self.visible = nn.Linear(64, targets * 3)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h, w = x.shape[-2:]
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, h, device=x.device, dtype=x.dtype),
            torch.linspace(-1, 1, w, device=x.device, dtype=x.dtype),
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
        visible_logits = self.visible(c.mean(dim=(2, 3))).reshape(-1, self.targets, 3)
        return heat, visible_logits


# Bộ nhớ đệm mô hình học sâu trong RAM
_REFLEX_MODEL: Optional[LocalizerFCN] = None
_PUPIL_MODEL: Optional[LocalizerFCN] = None


def get_reflex_model() -> Optional[LocalizerFCN]:
    """Nạp hoặc lấy mô hình reflex_localizer.pt trong RAM."""
    global _REFLEX_MODEL
    if _REFLEX_MODEL is not None:
        return _REFLEX_MODEL

    path = settings.WEIGHTS_DIR / "reflex_localizer.pt"
    if not path.exists():
        return None

    try:
        model = LocalizerFCN(in_channels=5, targets=2)
        cp = torch.load(path, map_location=settings.TORCH_DEVICE, weights_only=False)
        sd = cp["state_dict"] if "state_dict" in cp else cp
        model.load_state_dict(sd)
        model.to(settings.TORCH_DEVICE)
        model.eval()
        _REFLEX_MODEL = model
        return _REFLEX_MODEL
    except Exception:
        return None


def get_pupil_model() -> Optional[LocalizerFCN]:
    """Nạp hoặc lấy mô hình pupil_direct.pt trong RAM."""
    global _PUPIL_MODEL
    if _PUPIL_MODEL is not None:
        return _PUPIL_MODEL

    path = settings.WEIGHTS_DIR / "pupil_direct.pt"
    if not path.exists():
        return None

    try:
        model = LocalizerFCN(in_channels=6, targets=2)
        cp = torch.load(path, map_location=settings.TORCH_DEVICE, weights_only=False)
        sd = cp["state_dict"] if "state_dict" in cp else cp
        model.load_state_dict(sd)
        model.to(settings.TORCH_DEVICE)
        model.eval()
        _PUPIL_MODEL = model
        return _PUPIL_MODEL
    except Exception:
        return None


@dataclass
class ReflexModelResult:
    """Kết quả định vị điểm tham chiếu chuẩn Purkinje I hoặc tâm đồng tử dự phòng."""
    eye_side: str  # "OD" hoặc "OS"
    reference_point: Tuple[float, float]  # Tọa độ [x, y] được chọn làm điểm tham chiếu chuẩn
    reference_point_used: str  # "CORNEAL_LIGHT_REFLEX" hoặc "PUPIL_CENTER_FALLBACK"
    fallback_type: Optional[str] = None  # None, "PUPIL_DETECTOR", hoặc "IRIS_CENTER"
    confidence_score: float = 1.0
    has_fallback: bool = False
    glint_aspect_ratio: Optional[float] = None
    reflex_coordinates: Optional[Tuple[float, float]] = None
    pupil_coordinates: Optional[Tuple[float, float]] = None
    warning_flags: List[str] = field(default_factory=list)


def _prepare_256(pil_image: Image.Image) -> Tuple[np.ndarray, Tuple[float, float, int, int]]:
    """Letterbox chuẩn hóa ảnh crop về 256x256 giữ nguyên tỷ lệ."""
    width, height = pil_image.size
    factor = min(256.0 / width, 256.0 / height)
    w, h = max(1, round(width * factor)), max(1, round(height * factor))
    left, top = (256 - w) // 2, (256 - h) // 2

    canvas = Image.new("RGB", (256, 256))
    canvas.paste(pil_image.resize((w, h)), (left, top))
    arr = np.asarray(canvas, dtype=np.float32).transpose(2, 0, 1) / 255.0
    transform = (w / width, h / height, left, top)
    return arr, transform


def _compute_soft_prior(
    iris_geom: IrisGeometryResult,
    box: Tuple[int, int, int, int],
    is_flip: bool,
    letterbox: Tuple[float, float, int, int],
    size: int = 256,
) -> np.ndarray:
    """Tính toán hàm tiên nghiệm Gauss mềm 2D đại diện cho đĩa mống mắt (soft_prior từ V2)."""
    sx, sy, left, top = letterbox
    crop_w = box[2] - box[0]
    cx_local = iris_geom.iris_center[0] - box[0]
    if is_flip:
        cx_local = (crop_w - 1) - cx_local
    cy_local = iris_geom.iris_center[1] - box[1]

    x = cx_local * sx + left
    y = cy_local * sy + top
    rx = (iris_geom.hvid_px / 2.0) * sx
    ry = (iris_geom.vvid_px / 2.0) * sy

    yy, xx = np.mgrid[:size, :size]
    prior = np.exp(-0.5 * (((xx - x) / max(rx, 1e-4)) ** 2 + ((yy - y) / max(ry, 1e-4)) ** 2))[None].astype(np.float32)
    return prior


def _decode_expected_center(logits: torch.Tensor, valid_mask: torch.Tensor) -> Tuple[float, float]:
    """Giải mã tâm kỳ vọng không gian trên hàm phân bố xác suất softmax (expected_center từ V2)."""
    h, w = logits.shape[-2:]
    probs = logits.masked_fill(~valid_mask, -1e9).flatten(1).softmax(-1).reshape_as(logits)
    yy, xx = torch.meshgrid(
        torch.arange(h, device=logits.device, dtype=logits.dtype),
        torch.arange(w, device=logits.device, dtype=logits.dtype),
        indexing="ij",
    )
    center_x = (probs * xx).sum((1, 2))[0].item()
    center_y = (probs * yy).sum((1, 2))[0].item()
    return (float(center_x), float(center_y))


def _extract_optical_glint(
    crop_bgr: np.ndarray,
    box: Tuple[int, int, int, int],
    iris_center: Tuple[float, float],
    iris_radius: float,
    pupil_point: Tuple[float, float],
    eyelid_mask_crop: Optional[np.ndarray],
    reference_hint: Optional[Tuple[float, float]] = None,
) -> Optional[Tuple[Tuple[float, float], float]]:
    """
    Trích xuất đốm phản xạ Purkinje I qua bộ lọc quang học hình thái học kháng lác (Hirschberg V2):
    1. Mở rộng vùng tìm kiếm: Bán kính rộng <= 1.30 * R_iris (loại bỏ ràng buộc 0.4*R_iris tâm đồng tử gây mất phản quang khi lác/lé).
    2. Phân ngưỡng động dựa trên max luma (I >= max(180, min(PURKINJE_MIN_INTENSITY, max_val - 32))).
    3. Bảo toàn đốm sáng nhọn 1-2 pixel (không dùng morphological opening phá hủy điểm ảnh phản xạ).
    4. Lọc vệt lóa kéo dài (aspect ratio <= 2.8 khi diện tích >= 4 px).
    5. Tính trọng tâm cường độ sáng sub-pixel (Intensity Centroid).
    """
    ch, cw = crop_bgr.shape[:2]
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)

    x0, y0, _, _ = box
    cx_loc = iris_center[0] - x0
    cy_loc = iris_center[1] - y0

    # 1. Tạo mặt nạ vùng giác mạc / mống mắt kháng lác
    yy, xx = np.mgrid[:ch, :cw]
    dist_iris = np.sqrt((xx - cx_loc) ** 2 + (yy - cy_loc) ** 2)
    broad_radius = max(16.0, iris_radius * 1.10)
    search_mask = (dist_iris <= broad_radius)

    if eyelid_mask_crop is not None and eyelid_mask_crop.shape == (ch, cw):
        # Giãn nhẹ mặt nạ khe mi 5x5 để không cắt nhầm đốm sáng sát bờ mi
        dilated_eyelid = cv2.dilate(
            eyelid_mask_crop.astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
            iterations=1,
        )
        search_mask = search_mask & (dilated_eyelid > 0)

    if not search_mask.any():
        return None

    masked_pixels = gray[search_mask]
    if masked_pixels.size == 0:
        return None

    max_val = float(np.max(masked_pixels))
    # Ngưỡng luma tối thiểu lâm sàng cho phản quang Purkinje I
    min_peak_required = 180.0
    if max_val < min_peak_required:
        return None

    # 2. Phân ngưỡng cường độ sáng thích ứng
    threshold_val = max(min_peak_required, min(float(settings.PURKINJE_MIN_INTENSITY), max_val - 32.0))
    glint_binary = ((gray >= threshold_val) & search_mask).astype(np.uint8)

    # 3. Phân tích thành phần liên thông
    count, labels, stats, _ = cv2.connectedComponentsWithStats(glint_binary, 8)
    candidates = []

    target_anchor = reference_hint if reference_hint is not None else pupil_point

    for label in range(1, count):
        bx, by, bw, bh, area = map(int, stats[label])
        if area < 1 or area > settings.REFLEX_MAX_AREA_PX:
            continue
        aspect = max(bw, bh) / max(min(bw, bh), 1)
        if aspect > settings.REFLEX_MAX_ASPECT_RATIO and area >= 4:
            continue

        comp_mask = (labels[by:by + bh, bx:bx + bw] == label)
        ys, xs = np.nonzero(comp_mask)
        if len(xs) == 0:
            continue

        patch_gray = gray[by:by + bh, bx:bx + bw].astype(np.float32)
        weights = np.maximum(patch_gray[comp_mask] - threshold_val + 1.0, 1e-3)

        sub_x = float(np.average(xs + bx, weights=weights)) + x0
        sub_y = float(np.average(ys + by, weights=weights)) + y0
        peak_intensity = float(patch_gray[comp_mask].max())

        dist_anchor = math.hypot(sub_x - target_anchor[0], sub_y - target_anchor[1])
        score = (
            dist_anchor
            + 0.12 * float(area)
            + 1.5 * max(0.0, aspect - 1.0)
            - 0.03 * (peak_intensity - threshold_val)
        )

        candidates.append({
            "point": (round(sub_x, 2), round(sub_y, 2)),
            "aspect": round(aspect, 2),
            "peak": peak_intensity,
            "dist_anchor": dist_anchor,
            "score": score,
        })

    if not candidates:
        return None

    # Ưu tiên đốm có điểm số đa biến tối ưu nhất
    candidates.sort(key=lambda c: c["score"])
    best = candidates[0]
    return (best["point"], best["aspect"])


def locate_reference_point(
    image_bgr: np.ndarray,
    bbox: Tuple[int, int, int, int],
    iris_geom: IrisGeometryResult,
    eyelid_seg: Optional[EyelidSegmentationResult] = None,
    eye_side: str = "OD",
) -> ReflexModelResult:
    """
    Xác định điểm tham chiếu chuẩn hình thái học cho mắt qua chuỗi dự phòng 3 cấp:
    - Cấp 1 (Ưu tiên cao nhất): Phản xạ giác mạc Purkinje I từ pr_model + bộ lọc quang học reflex_roi_qc.
    - Cấp 2 (Dự phòng mất phản quang / flash yếu): Tâm đồng tử DirectPupilCNN (expected_center).
    - Cấp 3 (Dự phòng che lấp đồng tử): Tâm mống mắt hình học IrisFour.
    """
    x0, y0, x1, y1 = bbox
    crop = image_bgr[y0:y1, x0:x1]
    ch, cw = crop.shape[:2]
    pil_crop = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))

    is_flip = (eye_side.upper() == "OS")
    proc_crop = pil_crop.transpose(Image.Transpose.FLIP_LEFT_RIGHT) if is_flip else pil_crop

    arr_256, lb = _prepare_256(proc_crop)
    sx, sy, left, top = lb

    # --- BƯỚC 1: ĐỊNH VỊ TÂM ĐỒNG TỬ ĐỘC LẬP TỪ DIRECTPUPILCNN ---
    pupil_model = get_pupil_model()
    pupil_point: Optional[Tuple[float, float]] = None

    if pupil_model is not None:
        try:
            prior = _compute_soft_prior(iris_geom, bbox, is_flip, lb, size=256)
            tensor_pupil = torch.from_numpy(np.concatenate((arr_256, prior), axis=0)[None]).float().to(settings.TORCH_DEVICE)

            yy_grid, xx_grid = np.mgrid[:256, :256]
            valid_mask = torch.from_numpy(
                (xx_grid >= left) & (xx_grid < left + round(crop.shape[1] * sx)) &
                (yy_grid >= top) & (yy_grid < top + round(crop.shape[0] * sy))
            )[None].to(settings.TORCH_DEVICE)

            with torch.no_grad():
                heat_pupil, _ = pupil_model(tensor_pupil)
                raw_pupil_256 = _decode_expected_center(heat_pupil[:, 0], valid_mask)

            px_loc = (raw_pupil_256[0] - left) / sx
            py_loc = (raw_pupil_256[1] - top) / sy
            if is_flip:
                px_loc = (cw - 1) - px_loc

            pupil_point = (round(px_loc + x0, 2), round(py_loc + y0, 2))
        except Exception:
            pupil_point = None

    if pupil_point is None:
        pupil_point = iris_geom.iris_center

    # --- BƯỚC 2: ĐỊNH VỊ PHẢN QUANG PURKINJE I TỪ PR_MODEL ---
    pr_model = get_reflex_model()
    reflex_point_cnn: Optional[Tuple[float, float]] = None

    if pr_model is not None:
        try:
            tensor_pr = torch.from_numpy(arr_256[None]).float().to(settings.TORCH_DEVICE)
            with torch.no_grad():
                heat_pr, _ = pr_model(tensor_pr)
                heat_pr = heat_pr.flatten(2).softmax(-1).reshape_as(heat_pr)

            reflex_heat = heat_pr[0, 1].cpu().numpy()
            flat_idx = int(reflex_heat.argmax())
            rx_256, ry_256 = flat_idx % 256, flat_idx // 256

            local_rx = (rx_256 - left) / sx
            local_ry = (ry_256 - top) / sy
            if is_flip:
                local_rx = (cw - 1) - local_rx

            reflex_point_cnn = (round(local_rx + x0, 2), round(local_ry + y0, 2))
        except Exception:
            reflex_point_cnn = None

    # --- BƯỚC 3: BỘ LỌC QUANG HỌC REFLEX_ROI_QC & KHẲNG ĐỊNH PURKINJE I ---
    eyelid_mask_crop = eyelid_seg.binary_mask if (eyelid_seg is not None and eyelid_seg.binary_mask is not None) else None
    iris_radius = iris_geom.hvid_px / 2.0

    optical_glint = _extract_optical_glint(
        crop_bgr=crop,
        box=bbox,
        iris_center=iris_geom.iris_center,
        iris_radius=iris_radius,
        pupil_point=pupil_point,
        eyelid_mask_crop=eyelid_mask_crop,
        reference_hint=reflex_point_cnn,
    )

    # Đánh giá tính hợp lệ của đốm phản quang Purkinje I
    valid_reflex = False
    final_reflex_point = None
    glint_aspect = None

    if optical_glint is not None:
        opt_pt, opt_aspect = optical_glint
        dist_from_iris = math.hypot(opt_pt[0] - iris_geom.iris_center[0], opt_pt[1] - iris_geom.iris_center[1])
        if dist_from_iris <= (0.55 * iris_geom.hvid_px):
            glint_aspect = opt_aspect
            final_reflex_point = opt_pt
            valid_reflex = True

    # --- BƯỚC 4: THIẾT LẬP CHUỖI DỰ PHÒNG 3 CẤP (TIER 1 -> TIER 2 -> TIER 3) ---
    warning_flags = []

    # Cấp 1: Phản xạ giác mạc Purkinje I hợp lệ
    if valid_reflex and final_reflex_point is not None:
        # Cảnh báo nếu phản xạ bị lệch xa do lác/lé (Strabismus reflex deviation)
        dist_pupil = math.hypot(final_reflex_point[0] - pupil_point[0], final_reflex_point[1] - pupil_point[1])
        if dist_pupil > (0.35 * iris_geom.hvid_px):
            warning_flags.append("STRABISMUS_DISPLACED_REFLEX")

        return ReflexModelResult(
            eye_side=eye_side.upper(),
            reference_point=final_reflex_point,
            reference_point_used="CORNEAL_LIGHT_REFLEX",
            fallback_type=None,
            confidence_score=settings.PURKINJE_NORMAL_CONFIDENCE,
            has_fallback=False,
            glint_aspect_ratio=glint_aspect,
            reflex_coordinates=final_reflex_point,
            pupil_coordinates=pupil_point,
            warning_flags=warning_flags,
        )

    # Cấp 2: Dự phòng sang tâm đồng tử độc lập DirectPupilCNN
    warning_flags.append("REFLEX_FALLBACK_TO_PUPIL")
    dist_pupil_iris = math.dist(pupil_point, iris_geom.iris_center)

    if dist_pupil_iris <= (settings.PUPIL_MAX_OFFSET_FROM_IRIS * iris_geom.hvid_px):
        return ReflexModelResult(
            eye_side=eye_side.upper(),
            reference_point=pupil_point,
            reference_point_used="PUPIL_CENTER_FALLBACK",
            fallback_type="PUPIL_DETECTOR",
            confidence_score=settings.PUPIL_FALLBACK_CONFIDENCE,
            has_fallback=True,
            glint_aspect_ratio=None,
            reflex_coordinates=None,
            pupil_coordinates=pupil_point,
            warning_flags=warning_flags,
        )

    # Cấp 3: Dự phòng sang tâm mống mắt hình học IrisFour
    warning_flags.append("PUPIL_FALLBACK_TO_IRIS_CENTER")
    return ReflexModelResult(
        eye_side=eye_side.upper(),
        reference_point=iris_geom.iris_center,
        reference_point_used="PUPIL_CENTER_FALLBACK",
        fallback_type="IRIS_CENTER",
        confidence_score=settings.IRIS_CENTER_FALLBACK_CONFIDENCE,
        has_fallback=True,
        glint_aspect_ratio=None,
        reflex_coordinates=None,
        pupil_coordinates=pupil_point,
        warning_flags=warning_flags,
    )
