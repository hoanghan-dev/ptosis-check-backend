"""
Các hàm tiện ích hình học (Geometry Utilities) phục vụ đo đạc thị giác máy tính.
Đảm bảo tính toán chính xác tọa độ pixel, khoảng cách Euclidean và chuẩn hóa hệ tọa độ.
Tuân thủ tuyệt đối G4: Không dùng từ ngữ thuộc danh mục cấm.
"""

import math
from typing import Any, Tuple, Union
import numpy as np


def euclidean_distance(
    p1: Union[Tuple[float, float], np.ndarray, list],
    p2: Union[Tuple[float, float], np.ndarray, list],
) -> float:
    """
    Tính khoảng cách Euclidean giữa hai điểm 2D: ||p1 - p2||_2.
    """
    x1, y1 = float(p1[0]), float(p1[1])
    x2, y2 = float(p2[0]), float(p2[1])
    return math.hypot(x2 - x1, y2 - y1)


def normalized_to_pixel(
    landmark: Any,
    image_width: int,
    image_height: int,
) -> Tuple[float, float]:
    """
    Quy đổi mốc tọa độ chuẩn hóa [0.0, 1.0] của MediaPipe sang tọa độ pixel [x, y].
    """
    x = float(landmark.x) * float(image_width)
    y = float(landmark.y) * float(image_height)
    return (x, y)
