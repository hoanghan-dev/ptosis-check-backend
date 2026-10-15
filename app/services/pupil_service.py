"""
Dịch vụ Định vị Đồng tử Độc lập trong Vùng Mống Mắt (Module M4.5).
Cung cấp giải pháp dự phòng Cấp 1 khi phản quang giác mạc Purkinje I bị mất hoặc không xác định:
- Cắt ROI đĩa mống mắt dựa trên tâm và bán kính mống mắt (Module M3).
- Tạo mặt nạ khử lóa sáng giác mạc (bright_mask >= 180).
- Đánh giá độ tương phản vùng ROI (P85 - P15 >= 12.0) để loại trừ vùng phẳng đồng màu.
- Quét đa ngưỡng bách phân vị vùng tối [18%, 28%, 38%].
- Phân đoạn đồng tử bằng OpenCV CPU (connectedComponentsWithStats, arcLength).
- Lọc hình thái học: diện tích hợp lệ, độ lệch tâm <= 25% đường kính mống mắt, độ tròn >= 0.35.
- Chấm điểm tối ưu: Score = Circularity - (Offset / Diameter_iris).
- Trích xuất tâm đồng tử sub-pixel độc lập với MediaPipe.

Tuân thủ tuyệt đối G2 (In-Memory Zero-Retention) và G4 (Non-Diagnostic Safety).
"""

from dataclasses import dataclass
import math
from typing import List, Optional, Tuple
import cv2
import numpy as np

from app.core.config import settings


@dataclass
class PupilResult:
    """Kết quả định vị đồng tử độc lập trong vùng mống mắt."""
    pupil_point: Optional[Tuple[float, float]]
    pupil_diameter_px: Optional[float]
    confidence_score: float
    status: str  # "DETECTED", "PUPIL_NOT_FOUND", "LOW_QUALITY_INPUT"
    circularity: Optional[float] = None
    offset_from_iris_px: Optional[float] = None


def detect_pupil_in_iris_roi(
    image_bgr: np.ndarray,
    iris_center: Tuple[float, float],
    iris_radius: float,
) -> PupilResult:
    """
    Định vị tâm và ước lượng đường kính đồng tử độc lập trong ROI mống mắt.

    Args:
        image_bgr: Mảng ảnh BGR (hoặc ảnh xám) trong bộ nhớ RAM.
        iris_center: Tọa độ (x, y) tâm mống mắt từ Module M3.
        iris_radius: Bán kính mống mắt theo pixel (HVID / 2).

    Returns:
        PupilResult chứa tọa độ tâm đồng tử sub-pixel, đường kính và độ tin cậy.
    """
    if image_bgr is None or not isinstance(image_bgr, np.ndarray) or image_bgr.size == 0:
        raise ValueError("Dữ liệu ảnh đầu vào không hợp lệ hoặc rỗng.")

    cx, cy = float(iris_center[0]), float(iris_center[1])
    r_iris = float(iris_radius)

    if r_iris <= 2.0:
        return PupilResult(
            pupil_point=None,
            pupil_diameter_px=None,
            confidence_score=0.0,
            status="LOW_QUALITY_INPUT",
        )

    height, width = image_bgr.shape[:2]

    # Chuyển đổi sang ảnh xám nếu đầu vào là ảnh màu 3 kênh
    is_color = len(image_bgr.shape) == 3 and image_bgr.shape[2] >= 3
    if is_color:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = image_bgr.copy()

    # 1. Cắt ROI mống mắt với bán kính tìm kiếm (Search Radius = 0.85 * R)
    search_r = r_iris * settings.PUPIL_SEARCH_RADIUS_FACTOR
    x1 = max(0, int(np.floor(cx - search_r)))
    y1 = max(0, int(np.floor(cy - search_r)))
    x2 = min(width, int(np.ceil(cx + search_r)))
    y2 = min(height, int(np.ceil(cy + search_r)))

    if x2 <= x1 or y2 <= y1:
        return PupilResult(
            pupil_point=None,
            pupil_diameter_px=None,
            confidence_score=0.0,
            status="LOW_QUALITY_INPUT",
        )

    roi_gray = gray[y1:y2, x1:x2]
    roi_h, roi_w = roi_gray.shape

    # Lưới tọa độ cục bộ ROI ánh xạ sang hệ tọa độ toàn cục
    y_idx, x_idx = np.indices((roi_h, roi_w))
    gx = x1 + x_idx
    gy = y1 + y_idx

    # Khoảng cách hình học từ mỗi pixel đến tâm mống mắt
    dist_from_center = np.sqrt((gx - cx) ** 2 + (gy - cy) ** 2)
    search_mask = dist_from_center <= search_r

    # Mặt nạ loại trừ điểm lóa sáng giác mạc (bright_mask >= 180)
    if is_color:
        roi_bgr = image_bgr[y1:y2, x1:x2]
        bright_mask = (roi_bgr[..., 0] >= 180) & (roi_bgr[..., 1] >= 180) & (roi_bgr[..., 2] >= 180)
    else:
        bright_mask = roi_gray >= 180

    valid_iris_mask = search_mask & (~bright_mask)
    valid_pixels = roi_gray[valid_iris_mask]

    if valid_pixels.size < 16:
        return PupilResult(
            pupil_point=None,
            pupil_diameter_px=None,
            confidence_score=0.0,
            status="PUPIL_NOT_FOUND",
        )

    # 2. Đánh giá độ tương phản vùng ROI (P85 - P15 >= 12.0)
    p15 = float(np.percentile(valid_pixels, 15.0))
    p85 = float(np.percentile(valid_pixels, 85.0))
    contrast_diff = p85 - p15

    if contrast_diff < settings.PUPIL_MIN_CONTRAST_DIFF:
        return PupilResult(
            pupil_point=None,
            pupil_diameter_px=None,
            confidence_score=0.0,
            status="PUPIL_NOT_FOUND",
        )

    # 3. Quét đa ngưỡng bách phân vị vùng tối [18%, 28%, 38%]
    iris_diameter = 2.0 * r_iris
    min_area = math.pi * ((r_iris * settings.PUPIL_MIN_RADIUS_RATIO) ** 2)
    max_area = math.pi * ((r_iris * settings.PUPIL_MAX_RADIUS_RATIO) ** 2)
    max_offset = iris_diameter * settings.PUPIL_MAX_OFFSET_FROM_IRIS

    candidates: List[dict] = []
    percentiles = [18.0, 28.0, 38.0]

    for pct in percentiles:
        thresh = float(np.percentile(valid_pixels, pct))
        dark_mask = (roi_gray <= thresh) & valid_iris_mask
        dark_uint8 = dark_mask.astype(np.uint8)

        # Phân đoạn thành phần liên thông
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            dark_uint8, connectivity=8
        )

        for lbl in range(1, num_labels):
            area = int(stats[lbl, cv2.CC_STAT_AREA])
            if area < min_area or area > max_area:
                continue

            # Tọa độ tâm khối cục bộ và toàn cục
            comp_cx = x1 + float(centroids[lbl, 0])
            comp_cy = y1 + float(centroids[lbl, 1])
            offset = math.hypot(comp_cx - cx, comp_cy - cy)

            if offset > max_offset:
                continue

            # Tính chu vi và độ tròn qua cv2.findContours
            comp_bin = (labels == lbl).astype(np.uint8)
            contours, _ = cv2.findContours(comp_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue

            perimeter = cv2.arcLength(contours[0], closed=True)
            if perimeter <= 0:
                continue

            circularity = float(4.0 * math.pi * float(area) / (perimeter ** 2))
            if circularity < settings.PUPIL_MIN_CIRCULARITY:
                continue

            # Ước lượng đường kính tương đương hình tròn
            pupil_diameter = 2.0 * math.sqrt(float(area) / math.pi)

            # Hàm chấm điểm: Score = Circularity - (Offset / Diameter_iris)
            score = circularity - (offset / max(iris_diameter, 1e-6))

            candidates.append({
                "x": comp_cx,
                "y": comp_cy,
                "area": area,
                "diameter": pupil_diameter,
                "circularity": circularity,
                "offset": offset,
                "score": score,
            })

    if not candidates:
        return PupilResult(
            pupil_point=None,
            pupil_diameter_px=None,
            confidence_score=0.0,
            status="PUPIL_NOT_FOUND",
        )

    # 4. Chọn ứng viên có điểm số đánh giá cao nhất
    candidates.sort(key=lambda c: c["score"], reverse=True)
    best = candidates[0]

    return PupilResult(
        pupil_point=(round(best["x"], 2), round(best["y"], 2)),
        pupil_diameter_px=round(best["diameter"], 2),
        confidence_score=settings.PUPIL_FALLBACK_CONFIDENCE,
        status="DETECTED",
        circularity=round(best["circularity"], 4),
        offset_from_iris_px=round(best["offset"], 2),
    )
