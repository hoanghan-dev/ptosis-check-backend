"""
Tiện ích Nội suy Đường cong B-spline Bậc ba (Cubic Spline Interpolation).
Sử dụng scipy.interpolate để khớp đường viền trơn tru qua tập mốc giải phẫu
bờ mi trên và tính toán giao điểm tại kinh tuyến thẳng đứng x = x_ref.
Tuân thủ G4: Tuyệt đối không dùng từ ngữ thuộc danh mục cấm.
"""

from typing import List, Tuple
import numpy as np
from scipy.interpolate import CubicSpline


class EyelidSplineInterpolator:
    """Nội suy đường cong B-spline bậc ba cho bờ mi mắt."""

    def __init__(self, points: List[Tuple[float, float]]):
        """
        Khởi tạo đường cong nội suy từ danh sách các điểm mốc (x, y).

        Args:
            points: Danh sách các tọa độ điểm mốc bờ mi trên [(x1, y1), ...].
        """
        if not points or len(points) < 2:
            raise ValueError("Cần tối thiểu 2 điểm mốc để thực hiện nội suy đường cong.")

        # Sắp xếp các điểm theo thứ tự hoành độ x tăng dần
        sorted_pts = sorted(points, key=lambda p: float(p[0]))

        xs: List[float] = []
        ys: List[float] = []

        # Lọc và hợp nhất các điểm có hoành độ x quá gần nhau (< 1e-3 px)
        for pt in sorted_pts:
            px, py = float(pt[0]), float(pt[1])
            if not xs:
                xs.append(px)
                ys.append(py)
            else:
                if abs(px - xs[-1]) < 1e-3:
                    # Lấy trung bình tung độ y cho hoành độ x trùng lặp
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

        # Sử dụng CubicSpline tự nhiên (natural: đạo hàm bậc hai bằng 0 tại hai đầu)
        # Nếu số điểm < 4, sử dụng nội suy tuyến tính hoặc bậc thấp hơn
        if len(self.xs) >= 4:
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
        # Giới hạn x trong khoảng biên của tập mốc bờ mi
        x_clamped = float(np.clip(x, self.x_min, self.x_max))

        if self._spline is not None:
            return float(self._spline(x_clamped))
        else:
            # Fallback nội suy tuyến tính nếu số điểm < 4
            return float(np.interp(x_clamped, self.xs, self.ys))

    def generate_dense_curve(self, num_points: int = 30) -> List[Tuple[float, float]]:
        """
        Tạo danh sách các điểm trơn tru trên đường cong spline để vẽ overlay viền bờ mi.

        Args:
            num_points: Số lượng điểm mẫu dọc theo đường cong.

        Returns:
            Danh sách các tọa độ (x, y) mịn.
        """
        dense_x = np.linspace(self.x_min, self.x_max, max(2, num_points))
        if self._spline is not None:
            dense_y = self._spline(dense_x)
        else:
            dense_y = np.interp(dense_x, self.xs, self.ys)

        return [(round(float(x), 2), round(float(y), 2)) for x, y in zip(dense_x, dense_y)]
