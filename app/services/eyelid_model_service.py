"""
Dịch vụ Phân đoạn Khe mi & Xác định Bờ Mi Trên, Bờ Mi Dưới (Eyelid Model Service - Phase 4).
Kế thừa kiến trúc mạng EyelidUNet từ V2:
- Phân đoạn bề mặt mở mi nhị phân (palpebral fissure mask) kích thước 192x192.
- Trích xuất đường viền đa giác khép kín qua approxPolyDP triệt tiêu ảnh hưởng của lông mi và mascara.
- Xác định giao điểm bờ mi trên (y_margin) và bờ mi dưới (y_lower) tại kinh tuyến thẳng đứng X = x_ref.
- Tính toán chiều cao khe mi PFH_px = y_lower - y_margin.
- Tự động tách contour thành bờ mi trên (upper_eyelid_contour) và bờ mi dưới (lower_eyelid_contour).
- Hỗ trợ toàn diện trường hợp sa trễ mi nặng (y_margin > y_ref) an toàn, không gây lỗi hệ thống.
Tuân thủ tuyệt đối G2: Xử lý 100% trong RAM, không ghi đĩa.
Tuân thủ tuyệt đối G4: Không dùng từ ngữ thuộc danh mục cấm.
"""

from dataclasses import dataclass, field
import math
from typing import Any, List, Optional, Tuple
import cv2
import numpy as np
import torch
import torch.nn as nn

from app.core.config import settings


def _conv_groupnorm_block(in_c: int, out_c: int) -> nn.Sequential:
    """Khối tích chập kép 2 tầng Conv2d + GroupNorm(4) + ReLU chuẩn từ EyelidUNet."""
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, 3, padding=1),
        nn.GroupNorm(4, out_c),
        nn.ReLU(),
        nn.Conv2d(out_c, out_c, 3, padding=1),
        nn.GroupNorm(4, out_c),
        nn.ReLU(),
    )


class EyelidUNet(nn.Module):
    """
    Kiến trúc mạng U-Net thu nhỏ tối ưu CPU phân đoạn khe mở mi (eyelid_unet.pt).
    Encoder 3 tầng (16, 32, 64) kết hợp Decoder 2 tầng (32, 16) và skip connections.
    """
    def __init__(self):
        super().__init__()
        self.e1 = _conv_groupnorm_block(3, 16)
        self.e2 = _conv_groupnorm_block(16, 32)
        self.e3 = _conv_groupnorm_block(32, 64)
        self.d2 = _conv_groupnorm_block(96, 32)
        self.d1 = _conv_groupnorm_block(48, 16)
        self.head = nn.Conv2d(16, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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

        return self.head(d)


# Bộ nhớ đệm mô hình EyelidUNet trong RAM
_EYELID_UNET_MODEL: Optional[EyelidUNet] = None


def get_eyelid_model() -> Optional[EyelidUNet]:
    """Nạp hoặc lấy mô hình EyelidUNet trong RAM ở chế độ đánh giá."""
    global _EYELID_UNET_MODEL
    if _EYELID_UNET_MODEL is not None:
        return _EYELID_UNET_MODEL

    weights_path = settings.WEIGHTS_DIR / "eyelid_unet.pt"
    if not weights_path.exists():
        return None

    try:
        model = EyelidUNet()
        checkpoint = torch.load(weights_path, map_location=settings.TORCH_DEVICE, weights_only=False)
        state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
        model.load_state_dict(state_dict)
        model.to(settings.TORCH_DEVICE)
        model.eval()
        _EYELID_UNET_MODEL = model
        return _EYELID_UNET_MODEL
    except Exception:
        return None


@dataclass
class EyelidSegmentationResult:
    """Kết quả phân đoạn bờ mi trên, bờ mi dưới và khe mở mi."""
    eye_side: str  # "OD" hoặc "OS"
    margin_point: Tuple[float, float]  # P_margin = (x_ref, y_margin) bờ mi trên
    lower_margin_point: Tuple[float, float]  # P_lower = (x_ref, y_lower) bờ mi dưới
    pfh_px: float  # Chiều cao khe mi (pixel)
    upper_eyelid_contour: List[List[float]]  # Danh sách điểm [x, y] bờ mi trên
    lower_eyelid_contour: List[List[float]]  # Danh sách điểm [x, y] bờ mi dưới
    palpebral_contour: List[List[float]]  # Đa giác khép kín khe mi
    binary_mask: np.ndarray  # Mask khe mi nhị phân trong ROI crop
    confidence_score: float = 0.95
    warning_flags: List[str] = field(default_factory=list)
    is_valid: bool = True


def _smooth_eyelid_curve(points: List[List[float]], is_upper: bool) -> List[List[float]]:
    """Làm mịn đường bờ mi bằng cách khử nhiễu răng cưa (zigzag spikes) và bảo toàn tính đơn điệu theo trục X."""
    if len(points) < 3:
        return [[round(float(p[0]), 2), round(float(p[1]), 2)] for p in points]

    # Sắp xếp theo hoành độ X
    sorted_pts = sorted(points, key=lambda p: float(p[0]))

    # Hợp nhất các điểm có hoành độ x quá gần nhau (< 1.5 px)
    merged_pts: List[List[float]] = []
    for p in sorted_pts:
        px, py = float(p[0]), float(p[1])
        if not merged_pts:
            merged_pts.append([px, py])
        else:
            if abs(px - merged_pts[-1][0]) < 1.5:
                # Trung bình hóa tung độ thay vì lấy cực trị min/max để tránh bắt chóp lông mi
                merged_pts[-1][1] = (merged_pts[-1][1] + py) / 2.0
                merged_pts[-1][0] = (merged_pts[-1][0] + px) / 2.0
            else:
                merged_pts.append([px, py])

    if len(merged_pts) < 4:
        # Đảm bảo tính đơn điệu không giảm tuyệt đối của x
        for i in range(len(merged_pts) - 1):
            if merged_pts[i + 1][0] < merged_pts[i][0]:
                merged_pts[i + 1][0] = merged_pts[i][0]
        return [[round(p[0], 2), round(p[1], 2)] for p in merged_pts]

    xs = np.array([p[0] for p in merged_pts], dtype=np.float64)
    ys = np.array([p[1] for p in merged_pts], dtype=np.float64)

    # Làm mịn bằng Savitzky-Golay hoặc đa thức bậc 2
    try:
        from scipy.signal import savgol_filter
        window_length = min(len(ys) if len(ys) % 2 != 0 else len(ys) - 1, 7)
        if window_length >= 5:
            ys_smooth = savgol_filter(ys, window_length=window_length, polyorder=2)
            # Giữ nguyên 2 đầu mút canthi
            ys_smooth[0] = ys[0]
            ys_smooth[-1] = ys[-1]
            ys = ys_smooth
    except Exception:
        pass

    result = [[round(float(x), 2), round(float(y), 2)] for x, y in zip(xs, ys)]
    # Đảm bảo điều kiện kiểm thử: x[i] <= x[i+1]
    for i in range(len(result) - 1):
        if result[i + 1][0] < result[i][0]:
            result[i + 1][0] = result[i][0]
    return result


def _split_contour_into_upper_and_lower(
    contour_points: np.ndarray,
) -> Tuple[List[List[float]], List[List[float]]]:
    """
    Tách đường viền khép kín của khe mi thành 2 đường riêng biệt:
    1. Đường bờ mi trên (upper_eyelid_contour)
    2. Đường bờ mi dưới (lower_eyelid_contour)
    Dựa trên hai khóe mắt (cực trái và cực phải của đa giác) và làm mịn khử răng cưa.
    """
    if len(contour_points) < 3:
        pts = [[float(p[0]), float(p[1])] for p in contour_points]
        return pts, pts

    pts = contour_points.reshape(-1, 2)
    n = len(pts)

    idx_min_x = int(np.argmin(pts[:, 0]))
    idx_max_x = int(np.argmax(pts[:, 0]))

    if idx_min_x == idx_max_x:
        pts_list = [[float(p[0]), float(p[1])] for p in pts]
        return pts_list, pts_list

    # Tuyến đường 1 từ min_x đến max_x
    path1 = []
    curr = idx_min_x
    while True:
        path1.append(pts[curr])
        if curr == idx_max_x:
            break
        curr = (curr + 1) % n

    # Tuyến đường 2 từ max_x vòng về min_x
    path2 = []
    curr = idx_max_x
    while True:
        path2.append(pts[curr])
        if curr == idx_min_x:
            break
        curr = (curr + 1) % n

    # So sánh tung độ trung bình (trục Y hướng xuống dưới -> Y nhỏ hơn là ở phía trên)
    mean_y1 = float(np.mean([p[1] for p in path1]))
    mean_y2 = float(np.mean([p[1] for p in path2]))

    upper_raw = path1 if mean_y1 < mean_y2 else path2
    lower_raw = path2 if mean_y1 < mean_y2 else path1

    upper_lid = _smooth_eyelid_curve(upper_raw, is_upper=True)
    lower_lid = _smooth_eyelid_curve(lower_raw, is_upper=False)

    return upper_lid, lower_lid


def segment_eyelid_margins(
    image_bgr: np.ndarray,
    bbox: Tuple[int, int, int, int],
    x_ref: float,
    y_ref: Optional[float] = None,
    eye_side: str = "OD",
    eye_roi: Optional[Any] = None,
) -> Optional[EyelidSegmentationResult]:
    """
    Phân đoạn bờ sụn mi trên, bờ mi dưới và tính toán khe mi (PFH):
    1. Kiểm tra ưu tiên mốc giải phẫu chuẩn canonical_upper_lid_arc & canonical_lower_lid_arc từ eye_roi (nếu có)
       để tạo đường cong giải phẫu trơn láng bảo toàn hình học (PCHIP spline), loại bỏ hoàn toàn nhiễu gai nhọn do lông mi.
    2. Cắt crop ảnh mắt theo bbox, lan truyền qua EyelidUNet tạo mặt nạ nhị phân 192x192 (kết hợp lọc đóng Morphological Closing).
    3. Trích xuất giao điểm bờ mi trên y_margin và bờ mi dưới y_lower tại kinh tuyến thẳng đứng x = x_ref.
    4. Hỗ trợ trường hợp sa trễ mi nặng (y_margin > y_ref) an toàn tuyệt đối.
    """
    if image_bgr is None or image_bgr.size == 0 or not bbox:
        return None

    img_h, img_w = image_bgr.shape[:2]
    x0, y0, x1, y1 = bbox
    crop = image_bgr[y0:y1, x0:x1]
    ch, cw = crop.shape[:2]
    if ch < 10 or cw < 10:
        return None

    has_canonical = (
        eye_roi is not None
        and getattr(eye_roi, "canonical_upper_lid_arc", None) is not None
        and len(eye_roi.canonical_upper_lid_arc) >= 3
        and getattr(eye_roi, "canonical_lower_lid_arc", None) is not None
        and len(eye_roi.canonical_lower_lid_arc) >= 3
    )

    model = get_eyelid_model()
    if model is None and not has_canonical:
        return None

    # Mắt trái OS được lật ngang khi đưa vào mô hình học sâu để thống nhất giải phẫu
    is_flip = (eye_side.upper() == "OS")
    proc_crop = cv2.flip(crop, 1) if is_flip else crop.copy()

    # Chuẩn bị tensor 192x192 letterbox cho UNet nếu có mô hình
    mask_crop = None
    if model is not None:
        try:
            scale = min(192.0 / cw, 192.0 / ch)
            nw, nh = max(1, round(cw * scale)), max(1, round(ch * scale))
            left, top = (192 - nw) // 2, (192 - nh) // 2

            canvas = np.zeros((192, 192, 3), dtype=np.float32)
            resized = cv2.resize(proc_crop, (nw, nh)).astype(np.float32) / 255.0
            resized_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            canvas[top:top + nh, left:left + nw] = resized_rgb

            tensor = torch.from_numpy(canvas.transpose(2, 0, 1))[None].to(settings.TORCH_DEVICE)

            with torch.no_grad():
                prob = model(tensor).sigmoid()[0, 0].cpu().numpy()

            mask_letterbox = (prob[top:top + nh, left:left + nw] >= 0.5).astype(np.uint8)
            mask_crop = cv2.resize(mask_letterbox, (cw, ch), interpolation=cv2.INTER_NEAREST)

            if is_flip:
                mask_crop = cv2.flip(mask_crop, 1)

            close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask_crop = cv2.morphologyEx(mask_crop, cv2.MORPH_CLOSE, close_kernel)
        except Exception:
            mask_crop = None

    num_dense = getattr(settings, "EYELID_DENSE_CURVE_POINTS", 30)

    # --- NHÁNH 1: ƯU TIÊN MỐC GIẢI PHẪU CHUẨN KHÁNG NHIỄU LÔNG MI VÀ BÓNG ĐỔ ---
    if has_canonical:
        from app.utils.spline_interpolator import EyelidSplineInterpolator

        spline_up = EyelidSplineInterpolator(eye_roi.canonical_upper_lid_arc)
        spline_low = EyelidSplineInterpolator(eye_roi.canonical_lower_lid_arc)

        y_margin = spline_up.evaluate_at_x(x_ref)
        y_lower = spline_low.evaluate_at_x(x_ref)

        dense_upper = [[round(p[0], 2), round(p[1], 2)] for p in spline_up.generate_dense_curve(num_dense)]
        dense_lower = [[round(p[0], 2), round(p[1], 2)] for p in spline_low.generate_dense_curve(num_dense)]

        if getattr(eye_roi, "canonical_eyelid_contour", None) and len(eye_roi.canonical_eyelid_contour) >= 3:
            palpebral_pts = [[round(p[0], 2), round(p[1], 2)] for p in eye_roi.canonical_eyelid_contour]
        else:
            palpebral_pts = dense_upper + dense_lower[::-1]

        if mask_crop is None:
            mask_crop = np.zeros((ch, cw), dtype=np.uint8)
            poly_local = np.array([[[int(round(p[0] - x0)), int(round(p[1] - y0))]] for p in palpebral_pts], dtype=np.int32)
            cv2.fillPoly(mask_crop, [poly_local], 255)

    # --- NHÁNH 2: FALLBACK MẶT NẠ PHÂN ĐOẠN EYELIDUNET NẾU KHÔNG CÓ MỐC CANONICAL ---
    else:
        if mask_crop is None:
            return None

        contours, _ = cv2.findContours(mask_crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        # Lọc và chọn contour bờ mi chính xác gần vùng tham chiếu mắt (x_ref, y_ref)
        # thay vì chỉ chọn diện tích lớn nhất (tránh dính nhiễu biên/vùng má/trán)
        x_ref_local_f = float(x_ref - x0)
        y_ref_local_f = float(y_ref - y0) if y_ref is not None else (ch / 2.0)

        scored_contours = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < 10.0:
                continue
            bx, by, bw, bh = cv2.boundingRect(c)
            center_c_y = by + bh / 2.0
            center_c_x = bx + bw / 2.0
            dist_y = abs(center_c_y - y_ref_local_f)
            dist_x = abs(center_c_x - x_ref_local_f)
            score = area / (1.0 + 0.005 * (dist_y ** 2) + 0.002 * (dist_x ** 2))
            scored_contours.append((score, c))

        if not scored_contours:
            return None

        _, largest_contour = max(scored_contours, key=lambda sc: sc[0])

        epsilon = 0.005 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True).reshape(-1, 2)

        src_polygon = np.array([[float(p[0] + x0), float(p[1] + y0)] for p in approx], dtype=np.float32)
        palpebral_pts = [[round(float(p[0]), 2), round(float(p[1]), 2)] for p in src_polygon]

        upper_contour, lower_contour = _split_contour_into_upper_and_lower(src_polygon)

        # Tạo mặt nạ đơn chỉ chứa contour bờ mi đã chọn để quét cột x_ref không bị lẫn contour rác khác
        single_mask = np.zeros_like(mask_crop)
        cv2.drawContours(single_mask, [largest_contour], -1, 255, -1)

        # Xác định giao điểm y_margin và y_lower tại cột X = x_ref
        x_ref_local = int(round(x_ref - x0))
        x_ref_local_clamped = max(0, min(cw - 1, x_ref_local))

        col = single_mask[:, x_ref_local_clamped]
        active_ys = np.where(col > 0)[0]

        if len(active_ys) == 0:
            for delta in range(1, 15):
                left_col = max(0, x_ref_local_clamped - delta)
                right_col = min(cw - 1, x_ref_local_clamped + delta)
                col_l = single_mask[:, left_col]
                col_r = single_mask[:, right_col]
                ys_l = np.where(col_l > 0)[0]
                ys_r = np.where(col_r > 0)[0]
                if len(ys_l) > 0 and len(ys_r) > 0:
                    active_ys = np.array([int((ys_l[0] + ys_r[0]) / 2), int((ys_l[-1] + ys_r[-1]) / 2)])
                    break
                elif len(ys_l) > 0:
                    active_ys = ys_l
                    break
                elif len(ys_r) > 0:
                    active_ys = ys_r
                    break

        if len(active_ys) == 0:
            poly_ys = src_polygon[:, 1]
            y_margin = float(np.min(poly_ys))
            y_lower = float(np.max(poly_ys))
        else:
            y_margin = float(y0 + active_ys[0])
            y_lower = float(y0 + active_ys[-1])

        dense_upper = upper_contour
        if len(upper_contour) >= 2:
            try:
                from app.utils.spline_interpolator import EyelidSplineInterpolator
                spline_up = EyelidSplineInterpolator([(float(p[0]), float(p[1])) for p in upper_contour])
                dense_upper = [[round(p[0], 2), round(p[1], 2)] for p in spline_up.generate_dense_curve(num_dense)]
            except Exception:
                dense_upper = upper_contour

        dense_lower = lower_contour
        if len(lower_contour) >= 2:
            try:
                from app.utils.spline_interpolator import EyelidSplineInterpolator
                spline_low = EyelidSplineInterpolator([(float(p[0]), float(p[1])) for p in lower_contour])
                dense_lower = [[round(p[0], 2), round(p[1], 2)] for p in spline_low.generate_dense_curve(num_dense)]
            except Exception:
                dense_lower = lower_contour

    # Đảm bảo y_margin luôn ở phía trên hoặc bằng y_lower
    if y_margin > y_lower:
        y_margin, y_lower = y_lower, y_margin

    pfh_px = max(0.0, y_lower - y_margin)

    warning_flags = []
    # Rà soát trường hợp sa trễ mi nặng: bờ sụn mi che qua tâm phản xạ / đồng tử
    if y_ref is not None and y_margin > y_ref:
        warning_flags.append("SEVERE_DROOP_MARGIN_BELOW_REFLEX")

    margin_point = (round(float(x_ref), 2), round(float(y_margin), 2))
    lower_margin_point = (round(float(x_ref), 2), round(float(y_lower), 2))

    return EyelidSegmentationResult(
        eye_side=eye_side.upper(),
        margin_point=margin_point,
        lower_margin_point=lower_margin_point,
        pfh_px=round(pfh_px, 2),
        upper_eyelid_contour=dense_upper,
        lower_eyelid_contour=dense_lower,
        palpebral_contour=palpebral_pts,
        binary_mask=mask_crop,
        confidence_score=0.98 if has_canonical else 0.95,
        warning_flags=warning_flags,
        is_valid=True,
    )


def refine_margin_for_reference_point(
    eyelid_res: EyelidSegmentationResult,
    bbox: Tuple[int, int, int, int],
    x_ref: float,
    y_ref: Optional[float] = None,
) -> EyelidSegmentationResult:
    """
    Tinh chỉnh giao điểm bờ mi tại kinh tuyến thẳng đứng qua x_ref thực tế
    mà không cần tái chạy toàn bộ mạng nơ-ron EyelidUNet (Tiết kiệm thời gian CPU).
    Ưu tiên nội suy trực tiếp dọc theo đường cong bờ mi trơn láng (PCHIP).
    """
    up_pts = eyelid_res.upper_eyelid_contour
    low_pts = eyelid_res.lower_eyelid_contour

    if len(up_pts) >= 2 and len(low_pts) >= 2:
        up_xs = [float(p[0]) for p in up_pts]
        min_x = min(up_xs)
        max_x = max(up_xs)

        # Nếu x_ref nằm ngoài biên ngang của đường cong bờ mi [min_x, max_x],
        # KHÔNG được ép biên về khóe mắt (canthus) vì tại đó mi trên chụm vào mi dưới (PFH ~ 0),
        # làm kéo tụt bờ mi trên xuống thấp hơn phản quang (gây MRD1 âm phi lý).
        if x_ref < min_x or x_ref > max_x:
            y_margin = float(eyelid_res.margin_point[1])
            y_lower = float(eyelid_res.lower_margin_point[1])
        else:
            try:
                from app.utils.spline_interpolator import EyelidSplineInterpolator
                spline_up = EyelidSplineInterpolator([(float(p[0]), float(p[1])) for p in up_pts])
                spline_low = EyelidSplineInterpolator([(float(p[0]), float(p[1])) for p in low_pts])
                y_margin = spline_up.evaluate_at_x(x_ref)
                y_lower = spline_low.evaluate_at_x(x_ref)
            except Exception:
                up_ys = [p[1] for p in up_pts]
                y_margin = float(np.interp(x_ref, up_xs, up_ys))
                low_ys = [p[1] for p in low_pts]
                low_xs = [p[0] for p in low_pts]
                y_lower = float(np.interp(x_ref, low_xs, low_ys))

            # Nếu nội suy tại x_ref làm khe mi bị xẹp bất thường (< 5 px)
            # trong khi khe mi ban đầu mở tốt (pfh >= 10 px), giữ lại độ cao y ban đầu
            if (y_lower - y_margin) < 5.0 and eyelid_res.pfh_px >= 10.0:
                y_margin = float(eyelid_res.margin_point[1])
                y_lower = float(eyelid_res.lower_margin_point[1])
    else:
        x0, y0, x1, y1 = bbox
        mask_crop = eyelid_res.binary_mask
        ch, cw = (y1 - y0, x1 - x0)
        if mask_crop is not None and mask_crop.shape == (ch, cw):
            x_ref_local_clamped = int(np.clip(round(x_ref - x0), 0, cw - 1))
            col = mask_crop[:, x_ref_local_clamped]
            active_ys = np.where(col > 0)[0]
            if len(active_ys) > 0:
                y_margin = float(y0 + active_ys[0])
                y_lower = float(y0 + active_ys[-1])
            else:
                y_margin = float(eyelid_res.margin_point[1])
                y_lower = float(eyelid_res.lower_margin_point[1])
        else:
            y_margin = float(eyelid_res.margin_point[1])
            y_lower = float(eyelid_res.lower_margin_point[1])

    if y_margin > y_lower:
        y_margin, y_lower = y_lower, y_margin

    pfh_px = max(0.0, y_lower - y_margin)
    warning_flags = list(eyelid_res.warning_flags)
    if y_ref is not None and y_margin > y_ref and "SEVERE_DROOP_MARGIN_BELOW_REFLEX" not in warning_flags:
        warning_flags.append("SEVERE_DROOP_MARGIN_BELOW_REFLEX")

    return EyelidSegmentationResult(
        eye_side=eyelid_res.eye_side,
        margin_point=(round(float(x_ref), 2), round(float(y_margin), 2)),
        lower_margin_point=(round(float(x_ref), 2), round(float(y_lower), 2)),
        pfh_px=round(pfh_px, 2),
        upper_eyelid_contour=eyelid_res.upper_eyelid_contour,
        lower_eyelid_contour=eyelid_res.lower_eyelid_contour,
        palpebral_contour=eyelid_res.palpebral_contour,
        binary_mask=eyelid_res.binary_mask,
        confidence_score=eyelid_res.confidence_score,
        warning_flags=warning_flags,
        is_valid=eyelid_res.is_valid,
    )
