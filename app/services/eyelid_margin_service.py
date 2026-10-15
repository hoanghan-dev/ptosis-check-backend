"""
Dịch vụ Phân đoạn Bờ Sụn mi Trên (Module M5: Upper Eyelid Margin Extractor).
Trích xuất giao điểm của bờ tự do mi trên với kinh tuyến thẳng đứng đi qua
điểm tham chiếu (x_ref):
- Lấy tập mốc giải phẫu bờ mi trên từ MediaPipe FaceMesh (Module M3).
- Khớp đường cong nội suy B-spline bậc ba qua tập mốc.
- Xác định giao điểm ban đầu tại cột X = x_ref -> y_init.
- Quét gradient cường độ sáng cục bộ dọc kinh tuyến x_ref trong khoảng y_init +- 6 px
  để tinh chỉnh vị trí bờ sụn mi thực tế và loại trừ nhiễu từ ngọn lông mi / mascara.
- Xác định điểm đo lâm sàng chuẩn P_margin = (x_ref, y_margin).
Tuân thủ tuyệt đối G2 (In-Memory Zero-Retention) và G4 (Non-Diagnostic Safety).
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import cv2
import numpy as np

from app.core.config import settings
from app.utils.spline_interpolator import EyelidSplineInterpolator


@dataclass
class MarginResult:
    """Kết quả phân đoạn bờ sụn mi trên và tọa độ giao điểm chuẩn."""
    margin_point: Tuple[float, float]               # P_margin = (x_ref, y_margin)
    y_init_spline: float                            # Tung độ ban đầu từ nội suy B-spline
    y_margin_refined: float                         # Tung độ sau khi quét gradient tinh chỉnh
    spline_curve_points: List[Tuple[float, float]]  # Danh sách điểm đường cong B-spline mịn
    gradient_magnitude: float                       # Độ lớn gradient tại bờ sụn mi
    is_gradient_refined: bool                       # True nếu tìm thấy đỉnh gradient rõ nét
    warning_flags: List[str] = field(default_factory=list)


def extract_upper_margin(
    landmarks_upper: List[Tuple[float, float]],
    reflex_point: Tuple[float, float],
    image_gray: np.ndarray,
) -> MarginResult:
    """
    Trích xuất bờ sụn mi trên tại kinh tuyến thẳng đứng x_ref kết hợp B-spline và gradient.

    Args:
        landmarks_upper: Danh sách các mốc bờ mi trên [(x, y), ...].
        reflex_point: Tọa độ điểm tham chiếu (x_ref, y_ref) từ Module M4.
        image_gray: Mảng ảnh xám (hoặc BGR, sẽ tự chuyển đổi sang xám trong RAM).

    Returns:
        MarginResult chứa tọa độ P_margin và đường cong bờ mi.
    """
    if not landmarks_upper or len(landmarks_upper) < 2:
        raise ValueError("Danh sách mốc bờ mi trên không hợp lệ hoặc thiếu điểm.")

    if reflex_point is None or len(reflex_point) < 2:
        raise ValueError("Điểm tham chiếu phản quang / đồng tử không hợp lệ.")

    if image_gray is None or not isinstance(image_gray, np.ndarray) or image_gray.size == 0:
        raise ValueError("Dữ liệu ảnh đầu vào không hợp lệ hoặc rỗng.")

    if len(image_gray.shape) == 3:
        image_gray = cv2.cvtColor(image_gray, cv2.COLOR_BGR2GRAY)

    height, width = image_gray.shape[:2]
    x_ref = float(reflex_point[0])

    # 1. Khớp đường cong nội suy B-spline bậc ba qua tập mốc bờ mi trên
    interpolator = EyelidSplineInterpolator(landmarks_upper)

    # 2. Tìm giao điểm spline với cột X = x_ref -> y_init
    y_init = interpolator.evaluate_at_x(x_ref)

    # Tạo các điểm mịn trên đường cong B-spline phục vụ hiển thị
    spline_curve_points = interpolator.generate_dense_curve(
        num_points=settings.EYELID_DENSE_CURVE_POINTS
    )

    # 3. Quét gradient cường độ sáng dọc cột x_ref trong khoảng y_init +- 6 px
    window_px = settings.EYELID_GRADIENT_WINDOW_PX
    x_col = int(np.clip(int(np.round(x_ref)), 0, width - 1))
    y_int = int(np.round(y_init))

    y_min = max(1, y_int - window_px)
    y_max = min(height - 2, y_int + window_px)

    warning_flags: List[str] = []

    if y_max <= y_min:
        return MarginResult(
            margin_point=(round(x_ref, 2), round(y_init, 2)),
            y_init_spline=round(y_init, 2),
            y_margin_refined=round(y_init, 2),
            spline_curve_points=spline_curve_points,
            gradient_magnitude=0.0,
            is_gradient_refined=False,
            warning_flags=["MARGIN_SCAN_OUT_OF_IMAGE_BOUNDS"],
        )

    # Lấy dải 3 cột xung quanh x_col để giảm nhiễu hạt pixel đơn lẻ
    col_x1 = max(0, x_col - 1)
    col_x2 = min(width, x_col + 2)

    # Tính profile cường độ sáng trung bình theo chiều ngang tại mỗi dòng y
    # y nằm trong khoảng [y_min - 1, y_max + 1] để tính đạo hàm trung tâm
    scan_y_indices = np.arange(y_min - 1, y_max + 2)
    scan_y_indices = np.clip(scan_y_indices, 0, height - 1)
    strip = image_gray[scan_y_indices, col_x1:col_x2].astype(np.float32)
    profile = np.mean(strip, axis=1)

    # Tính đạo hàm dọc bậc nhất (đạo hàm trung tâm: dI/dy = I(y-1) - I(y+1))
    # Chuyển tiếp từ da mi trên (sáng hơn) xuống bờ mi / khe mi tối (I giảm theo chiều y tăng)
    # cho gradient dương rõ rệt: G_y = profile[i - 1] - profile[i + 1]
    gradients = []
    y_eval_list = list(range(y_min, y_max + 1))

    for idx, y_curr in enumerate(y_eval_list):
        # idx tương ứng với y_curr, profile[idx] tương ứng y_curr - 1, profile[idx + 2] tương ứng y_curr + 1
        g_y = float(profile[idx] - profile[idx + 2])
        gradients.append((y_curr, g_y, abs(g_y)))

    # 4. Tìm kiếm đỉnh biên độ tương phản bờ sụn mi
    min_thresh = float(settings.EYELID_MIN_GRADIENT_THRESHOLD)

    # Lọc các ứng viên có gradient dương đáng kể (bước nhảy giảm sáng từ da xuống khe mi)
    # hoặc độ lớn gradient lớn hơn ngưỡng
    positive_candidates = [
        (y, gy, abs_g) for y, gy, abs_g in gradients if gy >= min_thresh
    ]

    best_y = y_init
    max_gradient = 0.0
    is_refined = False

    if positive_candidates:
        # Chọn ứng viên có độ lớn gradient mạnh và gần y_init nhất
        # Trọng số kết hợp: gradient / (1 + 0.1 * |y - y_init|)
        best_candidate = max(
            positive_candidates,
            key=lambda c: c[1] / (1.0 + 0.15 * abs(c[0] - y_init)),
        )
        cand_y = best_candidate[0]
        max_gradient = best_candidate[1]

        # Tinh chỉnh sub-pixel bằng nội suy parabol 3 điểm quanh đỉnh
        cand_idx = y_eval_list.index(cand_y)
        if 0 < cand_idx < len(gradients) - 1:
            g_prev = gradients[cand_idx - 1][1]
            g_peak = gradients[cand_idx][1]
            g_next = gradients[cand_idx + 1][1]

            denom = 2.0 * (g_prev - 2.0 * g_peak + g_next)
            if abs(denom) > 1e-5:
                delta = (g_prev - g_next) / denom
                # Giới hạn delta trong khoảng [-0.5, 0.5]
                delta = float(np.clip(delta, -0.5, 0.5))
                best_y = cand_y + delta
            else:
                best_y = float(cand_y)
        else:
            best_y = float(cand_y)

        # Giới hạn độ lệch tối đa so với y_init theo cấu hình
        max_dev = float(settings.EYELID_MAX_EDGE_DEVIATION_PX)
        best_y = float(np.clip(best_y, y_init - max_dev, y_init + max_dev))
        is_refined = True
    else:
        # Nếu không có đỉnh gradient rõ rệt, giữ nguyên vị trí mốc B-spline
        best_y = y_init
        is_refined = False
        warning_flags.append("EYELID_MARGIN_GRADIENT_NOT_FOUND_USING_SPLINE")

    return MarginResult(
        margin_point=(round(x_ref, 2), round(best_y, 2)),
        y_init_spline=round(y_init, 2),
        y_margin_refined=round(best_y, 2),
        spline_curve_points=spline_curve_points,
        gradient_magnitude=round(max_gradient, 2),
        is_gradient_refined=is_refined,
        warning_flags=warning_flags,
    )
