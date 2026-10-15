"""
Dịch vụ Phát hiện Phản quang Giác mạc Purkinje & Cơ chế Dự phòng (Module M4).
Xác định đốm sáng phản xạ Purkinje I trên giác mạc từ đèn flash cận trục:
- Cắt ROI đĩa mống mắt dựa trên tâm và bán kính mống mắt (từ Module M3).
- Phân ngưỡng điểm sáng cực đại: I(x, y) >= Percentile_95 VÀ I(x, y) >= 230.
- Lọc đa phản xạ: Loại bỏ đốm ngoài vùng trung tâm 60% đĩa mống mắt.
- Tính trọng tâm cường độ sáng sub-pixel (Intensity Centroid).
- Cơ chế Fallback (FR-015): Tự động chuyển sang tâm đồng tử / mống mắt khi
  không có flash hoặc đa phản xạ nhập nhằng, gán cờ REFLEX_FALLBACK_TO_PUPIL
  và hạ điểm tin cậy xuống 0.65.
Tuân thủ tuyệt đối G2 (In-Memory Zero-Retention) và G4 (Non-Diagnostic Safety).
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import cv2
import numpy as np

from app.core.config import settings


@dataclass
class ReflexResult:
    """Kết quả phát hiện phản quang giác mạc hoặc điểm tham chiếu dự phòng."""
    reflex_point: Tuple[float, float]
    reference_point_used: str  # "CORNEAL_LIGHT_REFLEX" hoặc "PUPIL_CENTER"
    confidence_score: float
    has_fallback: bool
    glint_intensity: Optional[float] = None
    glint_area_px: Optional[int] = None
    warning_flags: List[str] = field(default_factory=list)


def detect_corneal_reflex(
    image_gray: np.ndarray,
    iris_center: Tuple[float, float],
    iris_radius: float,
) -> ReflexResult:
    """
    Phát hiện đốm phản xạ Purkinje I trên giác mạc trong vùng mống mắt.

    Args:
        image_gray: Ảnh xám (hoặc BGR, sẽ tự động chuyển đổi sang xám trong RAM).
        iris_center: Tọa độ (x, y) tâm mống mắt từ Module M3.
        iris_radius: Bán kính mống mắt theo pixel (HVID / 2).

    Returns:
        ReflexResult chứa tọa độ điểm tham chiếu sub-pixel và cờ cảnh báo dự phòng.
    """
    if image_gray is None or not isinstance(image_gray, np.ndarray) or image_gray.size == 0:
        raise ValueError("Dữ liệu ảnh đầu vào không hợp lệ hoặc rỗng.")

    if len(image_gray.shape) == 3:
        image_gray = cv2.cvtColor(image_gray, cv2.COLOR_BGR2GRAY)

    cx, cy = float(iris_center[0]), float(iris_center[1])
    r_iris = float(iris_radius)

    # Điểm dự phòng mặc định là tâm mống mắt / đồng tử
    fallback_point = (round(cx, 2), round(cy, 2))

    if r_iris <= 2.0:
        return ReflexResult(
            reflex_point=fallback_point,
            reference_point_used="PUPIL_CENTER",
            confidence_score=settings.PURKINJE_FALLBACK_CONFIDENCE,
            has_fallback=True,
            warning_flags=["REFLEX_FALLBACK_TO_PUPIL", "INVALID_IRIS_RADIUS"],
        )

    height, width = image_gray.shape[:2]

    # 1. Cắt ROI đĩa mống mắt (với lề an toàn)
    x1 = max(0, int(np.floor(cx - r_iris)))
    y1 = max(0, int(np.floor(cy - r_iris)))
    x2 = min(width, int(np.ceil(cx + r_iris)))
    y2 = min(height, int(np.ceil(cy + r_iris)))

    if x2 <= x1 or y2 <= y1:
        return ReflexResult(
            reflex_point=fallback_point,
            reference_point_used="PUPIL_CENTER",
            confidence_score=settings.PURKINJE_FALLBACK_CONFIDENCE,
            has_fallback=True,
            warning_flags=["REFLEX_FALLBACK_TO_PUPIL", "ROI_OUT_OF_BOUNDS"],
        )

    roi = image_gray[y1:y2, x1:x2].astype(np.float32)
    roi_h, roi_w = roi.shape

    # Tạo lưới tọa độ cục bộ trong ROI tương ứng với tọa độ toàn cục
    y_indices, x_indices = np.indices((roi_h, roi_w))
    global_x = x1 + x_indices
    global_y = y1 + y_indices

    # Tính khoảng cách Euclidean từ mỗi pixel tới tâm mống mắt
    dist_from_center = np.sqrt((global_x - cx) ** 2 + (global_y - cy) ** 2)

    # Mặt nạ đĩa mống mắt đầy đủ
    iris_mask = dist_from_center <= r_iris
    iris_pixels = roi[iris_mask]

    if iris_pixels.size == 0:
        return ReflexResult(
            reflex_point=fallback_point,
            reference_point_used="PUPIL_CENTER",
            confidence_score=settings.PURKINJE_FALLBACK_CONFIDENCE,
            has_fallback=True,
            warning_flags=["REFLEX_FALLBACK_TO_PUPIL"],
        )

    # 2. Tìm đốm sáng cực đại: I(x, y) >= Percentile_95 VÀ I(x, y) >= 230
    p95 = float(np.percentile(iris_pixels, settings.PURKINJE_PERCENTILE))
    threshold_val = max(float(settings.PURKINJE_MIN_INTENSITY), p95)

    # 3. Lọc đa phản xạ: Chỉ giữ đốm trong vùng trung tâm 60% đĩa mống mắt
    central_radius = r_iris * settings.PURKINJE_CENTRAL_DISC_RATIO
    central_mask = dist_from_center <= central_radius

    # Mặt nạ các pixel sáng thỏa mãn cả 2 điều kiện và nằm trong 60% tâm
    candidate_mask = (roi >= threshold_val) & central_mask
    candidate_uint8 = candidate_mask.astype(np.uint8)

    # 4. Phân tích thành phần liên thông (Connected Components)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        candidate_uint8, connectivity=8
    )

    valid_components = []
    for lbl in range(1, num_labels):
        area = int(stats[lbl, cv2.CC_STAT_AREA])
        comp_mask = labels == lbl
        peak_intensity = float(np.max(roi[comp_mask]))
        valid_components.append({
            "label": lbl,
            "area": area,
            "peak_intensity": peak_intensity,
        })

    # Nếu không có thành phần nào thỏa mãn -> Fallback sang tâm đồng tử
    if not valid_components:
        return ReflexResult(
            reflex_point=fallback_point,
            reference_point_used="PUPIL_CENTER",
            confidence_score=settings.PURKINJE_FALLBACK_CONFIDENCE,
            has_fallback=True,
            warning_flags=["REFLEX_FALLBACK_TO_PUPIL"],
        )

    # Sắp xếp các thành phần theo diện tích giảm dần
    valid_components.sort(key=lambda c: c["area"], reverse=True)

    # Kiểm tra tính nhập nhằng (đa phản quang cạnh tranh trong 60% trung tâm)
    if len(valid_components) > 1:
        comp_primary = valid_components[0]
        comp_secondary = valid_components[1]
        # Nếu có từ 2 đốm sáng có diện tích tương đương (đốm 2 >= 40% đốm 1) và kích thước không phải nhiễu
        if comp_secondary["area"] >= max(2, int(0.40 * comp_primary["area"])):
            return ReflexResult(
                reflex_point=fallback_point,
                reference_point_used="PUPIL_CENTER",
                confidence_score=settings.PURKINJE_FALLBACK_CONFIDENCE,
                has_fallback=True,
                warning_flags=["REFLEX_FALLBACK_TO_PUPIL", "AMBIGUOUS_MULTI_REFLECTION"],
            )

    # Chọn đốm phản quang chính
    chosen_label = valid_components[0]["label"]
    chosen_mask = labels == chosen_label
    chosen_area = valid_components[0]["area"]
    chosen_peak = valid_components[0]["peak_intensity"]

    # 5. Tính trọng tâm cường độ sáng sub-pixel (Intensity Centroid):
    # x_reflex = sum(x * I) / sum(I), y_reflex = sum(y * I) / sum(I)
    weights = roi[chosen_mask]
    total_weight = float(np.sum(weights))

    if total_weight <= 0:
        return ReflexResult(
            reflex_point=fallback_point,
            reference_point_used="PUPIL_CENTER",
            confidence_score=settings.PURKINJE_FALLBACK_CONFIDENCE,
            has_fallback=True,
            warning_flags=["REFLEX_FALLBACK_TO_PUPIL"],
        )

    xs = global_x[chosen_mask]
    ys = global_y[chosen_mask]

    subpixel_x = float(np.sum(xs * weights) / total_weight)
    subpixel_y = float(np.sum(ys * weights) / total_weight)

    return ReflexResult(
        reflex_point=(round(subpixel_x, 2), round(subpixel_y, 2)),
        reference_point_used="CORNEAL_LIGHT_REFLEX",
        confidence_score=settings.PURKINJE_NORMAL_CONFIDENCE,
        has_fallback=False,
        glint_intensity=round(chosen_peak, 2),
        glint_area_px=chosen_area,
        warning_flags=[],
    )
