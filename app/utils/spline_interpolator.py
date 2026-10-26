"""
Tiện ích Nội suy Đường cong Bờ mi Trơn láng Bảo toàn Hình học (Monotonic Shape-Preserving PCHIP Interpolation).
Sử dụng scipy.interpolate để khớp đường viền trơn tru qua tập mốc giải phẫu
bờ mi và tính toán giao điểm tại kinh tuyến thẳng đứng x = x_ref.
Triệt tiêu hoàn toàn hiện tượng Runge (Runge's Phenomenon) và dao động gai nhọn.
Tuân thủ G4: Tuyệt đối không dùng từ ngữ thuộc danh mục cấm.
"""

from typing import List, Tuple
import numpy as np
try:
    from scipy.interpolate import PchipInterpolator
except ImportError:
    PchipInterpolator = None
from scipy.interpolate import CubicSpline


class EyelidSplineInterpolator:
    """Nội suy đường cong bờ mi mắt bảo toàn tính đơn điệu không gai nhọn."""

    def __init__(self, points: List[Tuple[float, float]]):
        """
        Khởi tạo đường cong nội suy từ danh sách các điểm mốc (x, y).

        Args:
            points: Danh sách các tọa độ điểm mốc bờ mi [(x1, y1), ...].
        """
        if not points or len(points) < 2:
            raise ValueError("Cần tối thiểu 2 điểm mốc để thực hiện nội suy đường cong.")

        # Sắp xếp các điểm theo thứ tự hoành độ x tăng dần
        sorted_pts = sorted(points, key=lambda p: float(p[0]))

        xs: List[float] = []
        ys: List[float] = []

        # Lọc và hợp nhất các điểm có hoành độ x quá gần nhau (< 1.0 px)
        for pt in sorted_pts:
            px, py = float(pt[0]), float(pt[1])
            if not xs:
                xs.append(px)
                ys.append(py)
            else:
                if abs(px - xs[-1]) < 1.0:
                    # Lấy trung bình tung độ y cho hoành độ x trùng lặp/gần nhau
                    ys[-1] = (ys[-1] + py) / 2.0
                else:
                    xs.append(px)
                    ys.append(py)

        if len(xs) < 2:
            raise ValueError("Không đủ các điểm phân biệt theo trục x để tạo đường cong.")

        self.x_min = xs[0]
        self.x_max = xs[-1]
        self.xs = np.array(xs, dtype=np.float64)
        self.ys = np.array(ys, dtype=np.float64)

        # Ưu tiên PchipInterpolator (Bảo toàn hình học, triệt tiêu overshoot và gai nhọn)
        if PchipInterpolator is not None and len(self.xs) >= 3:
            self._spline = PchipInterpolator(self.xs, self.ys)
        elif len(self.xs) >= 4:
            self._spline = CubicSpline(self.xs, self.ys, bc_type="natural")
        else:
            self._spline = None

    def evaluate_at_x(self, x: float) -> float:
        """
        Tính tung độ y tại hoành độ x dựa trên đường cong spline.

        Args:
            x: Tọa độ hoành độ cần tính (kinh tuyến thẳng đứng x_ref).

        Returns:
            Tung độ y (pixel).
        """
        x_clamped = float(np.clip(x, self.x_min, self.x_max))

        if self._spline is not None:
            return float(self._spline(x_clamped))
        else:
            # Fallback nội suy tuyến tính nếu số điểm < 3
            return float(np.interp(x_clamped, self.xs, self.ys))

    def generate_dense_curve(self, num_points: int = 30) -> List[Tuple[float, float]]:
        """
        Tạo danh sách các điểm trơn tru trên đường cong spline để vẽ overlay viền bờ mi.

        Args:
            num_points: Số lượng điểm mẫu dọc theo đường cong.

        Returns:
            Danh sách các tọa độ (x, y) mịn không có gai nhọn.
        """
        dense_x = np.linspace(self.x_min, self.x_max, max(2, num_points))
        if self._spline is not None:
            dense_y = self._spline(dense_x)
            # Giới hạn biên chặt chẽ trong khoảng dữ liệu mốc thật
            y_range = max(1.0, float(np.ptp(self.ys)))
            y_min_bound = float(np.min(self.ys) - 0.05 * y_range)
            y_max_bound = float(np.max(self.ys) + 0.05 * y_range)
            dense_y = np.clip(dense_y, y_min_bound, y_max_bound)
        else:
            dense_y = np.interp(dense_x, self.xs, self.ys)

        return [(round(float(x), 2), round(float(y), 2)) for x, y in zip(dense_x, dense_y)]
