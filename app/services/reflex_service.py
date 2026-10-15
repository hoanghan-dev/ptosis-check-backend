"""
Dịch vụ Phát hiện Phản quang Giác mạc Purkinje V2 & Cơ chế Dự phòng (Module M4).
Xác định đốm sáng phản xạ Purkinje I trên giác mạc từ đèn flash cận trục:
- Cắt ROI đĩa mống mắt dựa trên tâm và bán kính mống mắt (từ Module M3).
- Phân ngưỡng điểm sáng cực đại: I(x, y) >= Percentile_95 VÀ I(x, y) >= 230.
- Lọc đa phản xạ: Loại bỏ đốm ngoài vùng trung tâm 60% đĩa mống mắt.
- Lọc vệt lóa kéo dài (Aspect Ratio <= 2.8) và ngưỡng diện tích cực đại.
- Hàm chấm điểm ứng viên đa biến (Multi-Objective Scoring).
- Chốt chặn phân cụm ứng viên cạnh tranh (Cluster Guard <= 8 px).
- Tính trọng tâm cường độ sáng sub-pixel (Intensity Centroid).
- Cơ chế Fallback: Tự động chuyển sang tâm đồng tử / mống mắt khi
  không có flash hoặc đa phản xạ nhập nhằng, gán cờ REFLEX_FALLBACK_TO_PUPIL
  và sử dụng điểm tham chiếu PUPIL_CENTER_FALLBACK.
Tuân thủ tuyệt đối G2 (In-Memory Zero-Retention) và G4 (Non-Diagnostic Safety).
"""

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import List, Optional, Tuple
import cv2
import numpy as np

from app.core.config import settings


class ReflexDetectionStatus(str, Enum):
    """Trạng thái nhận diện phản xạ giác mạc từ thuật toán V2."""
    SUCCESS = "SUCCESS"
    LARGE_OR_ELONGATED_GLARE = "LARGE_OR_ELONGATED_GLARE"
    CLUSTERED_REFLEX_CANDIDATES = "CLUSTERED_REFLEX_CANDIDATES"
    NO_REFLEX_FOUND = "NO_REFLEX_FOUND"
    AMBIGUOUS_MULTI_REFLECTION = "AMBIGUOUS_MULTI_REFLECTION"


@dataclass
class ReflexResult:
    """Kết quả phát hiện phản quang giác mạc hoặc điểm tham chiếu dự phòng."""
    reflex_point: Tuple[float, float]
    reference_point_used: str  # "CORNEAL_LIGHT_REFLEX" hoặc "PUPIL_CENTER_FALLBACK"
    confidence_score: float
    has_fallback: bool
    status: ReflexDetectionStatus = ReflexDetectionStatus.SUCCESS
    glint_intensity: Optional[float] = None
    glint_area_px: Optional[int] = None
    glint_aspect_ratio: Optional[float] = None
    warning_flags: List[str] = field(default_factory=list)


def detect_corneal_reflex(
    image_gray: np.ndarray,
    iris_center: Tuple[float, float],
    iris_radius: float,
) -> ReflexResult:
    """
    Phát hiện đốm phản xạ Purkinje I trên giác mạc trong vùng mống mắt (Thuật toán V2).

    Args:
        image_gray: Ảnh xám (hoặc BGR, sẽ tự động chuyển đổi sang xám trong RAM).
        iris_center: Tọa độ (x, y) tâm mống mắt từ Module M3.
        iris_radius: Bán kính mống mắt theo pixel (HVID / 2).

    Returns:
        ReflexResult chứa tọa độ điểm tham chiếu sub-pixel, trạng thái V2 và cờ cảnh báo.
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
            reference_point_used="PUPIL_CENTER_FALLBACK",
            confidence_score=settings.PURKINJE_FALLBACK_CONFIDENCE,
            has_fallback=True,
            status=ReflexDetectionStatus.NO_REFLEX_FOUND,
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
            reference_point_used="PUPIL_CENTER_FALLBACK",
            confidence_score=settings.PURKINJE_FALLBACK_CONFIDENCE,
            has_fallback=True,
            status=ReflexDetectionStatus.NO_REFLEX_FOUND,
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
            reference_point_used="PUPIL_CENTER_FALLBACK",
            confidence_score=settings.PURKINJE_FALLBACK_CONFIDENCE,
            has_fallback=True,
            status=ReflexDetectionStatus.NO_REFLEX_FOUND,
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

    # Ngưỡng diện tích cực đại (giới hạn tuyệt đối và tương đối theo pi * R^2)
    max_area = min(
        settings.REFLEX_MAX_AREA_PX,
        int(math.pi * (r_iris ** 2) * settings.REFLEX_MAX_AREA_IRIS_RATIO),
    )

    candidates = []
    rejected_glare_count = 0

    for lbl in range(1, num_labels):
        comp_w = int(stats[lbl, cv2.CC_STAT_WIDTH])
        comp_h = int(stats[lbl, cv2.CC_STAT_HEIGHT])
        area = int(stats[lbl, cv2.CC_STAT_AREA])

        # Tỷ lệ trục dài / trục ngắn
        aspect_ratio = float(max(comp_w, comp_h)) / float(max(1, min(comp_w, comp_h)))

        # Bộ lọc vệt lóa & diện tích cực đại từ V2
        if area > max_area or (aspect_ratio > settings.REFLEX_MAX_ASPECT_RATIO and area >= 4):
            rejected_glare_count += 1
            continue

        comp_mask = labels == lbl
        peak_intensity = float(np.max(roi[comp_mask]))

        # Tọa độ trọng tâm thành phần trong hệ tọa độ toàn cục
        comp_cx = x1 + float(centroids[lbl, 0])
        comp_cy = y1 + float(centroids[lbl, 1])
        dist_val = math.hypot(comp_cx - cx, comp_cy - cy)

        # Hàm chấm điểm ứng viên đa biến (Multi-Objective Scoring)
        score = (
            dist_val
            + 0.12 * float(area)
            + 1.5 * max(0.0, aspect_ratio - 1.0)
            - 0.03 * (peak_intensity - threshold_val)
        )

        candidates.append({
            "label": lbl,
            "area": area,
            "aspect_ratio": aspect_ratio,
            "peak_intensity": peak_intensity,
            "x": comp_cx,
            "y": comp_cy,
            "score": score,
        })

    # Nếu không có ứng viên nào vượt qua bộ lọc
    if not candidates:
        if rejected_glare_count > 0:
            return ReflexResult(
                reflex_point=fallback_point,
                reference_point_used="PUPIL_CENTER_FALLBACK",
                confidence_score=settings.PURKINJE_FALLBACK_CONFIDENCE,
                has_fallback=True,
                status=ReflexDetectionStatus.LARGE_OR_ELONGATED_GLARE,
                warning_flags=["REFLEX_FALLBACK_TO_PUPIL", "REFLEX_GLARE_ARTIFACT"],
            )
        return ReflexResult(
            reflex_point=fallback_point,
            reference_point_used="PUPIL_CENTER_FALLBACK",
            confidence_score=settings.PURKINJE_FALLBACK_CONFIDENCE,
            has_fallback=True,
            status=ReflexDetectionStatus.NO_REFLEX_FOUND,
            warning_flags=["REFLEX_FALLBACK_TO_PUPIL"],
        )

    # Sắp xếp danh sách ứng viên theo điểm số tăng dần (điểm thấp nhất = tối ưu nhất)
    candidates.sort(key=lambda c: c["score"])

    # 5. Kiểm tra phân cụm ứng viên cạnh tranh (Cluster Guard)
    cluster_dist_limit = max(
        settings.REFLEX_CLUSTER_DISTANCE_PX,
        r_iris * settings.REFLEX_CLUSTER_DISTANCE_IRIS_RATIO,
    )

    has_competing_cluster = False
    if len(candidates) > 1:
        best = candidates[0]
        for other in candidates[1:]:
            d = math.hypot(other["x"] - best["x"], other["y"] - best["y"])
            score_gap = other["score"] - best["score"]
            if d <= cluster_dist_limit and score_gap <= settings.REFLEX_CLUSTER_SCORE_GAP:
                has_competing_cluster = True
                break

    if has_competing_cluster:
        return ReflexResult(
            reflex_point=fallback_point,
            reference_point_used="PUPIL_CENTER_FALLBACK",
            confidence_score=settings.PURKINJE_FALLBACK_CONFIDENCE,
            has_fallback=True,
            status=ReflexDetectionStatus.CLUSTERED_REFLEX_CANDIDATES,
            warning_flags=["REFLEX_FALLBACK_TO_PUPIL", "CLUSTERED_REFLEX_CANDIDATES"],
        )

    # 6. Kiểm tra tính nhập nhằng (đa phản quang xa nhau có diện tích tương đương)
    if len(candidates) > 1:
        cand_by_area = sorted(candidates, key=lambda c: c["area"], reverse=True)
        if cand_by_area[1]["area"] >= max(2, int(0.40 * cand_by_area[0]["area"])):
            return ReflexResult(
                reflex_point=fallback_point,
                reference_point_used="PUPIL_CENTER_FALLBACK",
                confidence_score=settings.PURKINJE_FALLBACK_CONFIDENCE,
                has_fallback=True,
                status=ReflexDetectionStatus.AMBIGUOUS_MULTI_REFLECTION,
                warning_flags=["REFLEX_FALLBACK_TO_PUPIL", "AMBIGUOUS_MULTI_REFLECTION"],
            )

    # 7. Trích xuất ứng viên tối ưu nhất
    chosen = candidates[0]
    chosen_label = chosen["label"]
    chosen_mask = labels == chosen_label
    chosen_area = chosen["area"]
    chosen_aspect = chosen["aspect_ratio"]
    chosen_peak = chosen["peak_intensity"]

    # 8. Tính trọng tâm cường độ sáng sub-pixel (Intensity Centroid):
    # x_reflex = sum(x * I) / sum(I), y_reflex = sum(y * I) / sum(I)
    weights = roi[chosen_mask]
    total_weight = float(np.sum(weights))

    if total_weight <= 0:
        return ReflexResult(
            reflex_point=fallback_point,
            reference_point_used="PUPIL_CENTER_FALLBACK",
            confidence_score=settings.PURKINJE_FALLBACK_CONFIDENCE,
            has_fallback=True,
            status=ReflexDetectionStatus.NO_REFLEX_FOUND,
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
        status=ReflexDetectionStatus.SUCCESS,
        glint_intensity=round(chosen_peak, 2),
        glint_area_px=chosen_area,
        glint_aspect_ratio=round(chosen_aspect, 2),
        warning_flags=[],
    )
