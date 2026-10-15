"""
Phase 0 - Hiệu chuẩn hệ số tương đối V2 theo bán kính mống mắt R_iris.

Script THUẦN ĐỌC (read-only) và độc lập: không sửa bất kỳ file nào trong
backend/app/ hay frontend/src/. Chỉ ghi duy nhất báo cáo thống kê tham số
`calibration_report.json` (không chứa dữ liệu ảnh người dùng).

Nguồn dữ liệu:
  - "real": ảnh tests/fixtures/sample_face.jpg, nạp vào RAM (np.frombuffer +
    cv2.imdecode), lấy R_iris từ MediaPipe qua detect_landmarks (đọc-chỉ).
  - "synthetic": ảnh tổng hợp với R_iris ∈ {20, 30, 45, 60, 80} px. Đốm sáng
    được sinh với tỷ lệ lấy từ dải kỳ vọng trong kế hoạch
    (k_cluster ≈ 0.18-0.25, k_area ≈ 0.03-0.06). Đây là bước TỰ KIỂM CHỨNG
    của bộ đo, KHÔNG phải bằng chứng thực nghiệm.

Hệ số:
  k_cluster = d_cluster / R_iris
  k_area    = Area_glare / R_iris^2
  k_aspect  = max(w, h) / min(w, h) của đốm phản quang (không thứ nguyên)
  k_search  = R_search / R_iris (xác nhận 0.60 cho trung tâm mống mắt)

Mọi tham số được đánh dấu TODO_PILOT để căn chỉnh thêm trong thử nghiệm thực tế.

Tuân thủ G2 (không ghi ảnh xuống đĩa) và G4 (ngôn ngữ phi can thiệp chuyên sâu).

Chạy:  python backend/scripts/calibrate_v2_coefficients.py
"""

import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

REPORT_PATH = Path(__file__).resolve().parent / "calibration_report.json"
SAMPLE_IMAGE_PATH = BACKEND_DIR / "tests" / "fixtures" / "sample_face.jpg"

SYNTHETIC_R_IRIS = [20, 30, 45, 60, 80]
SAMPLES_PER_RADIUS = 100
RANDOM_SEED = 20261015
SUPERSAMPLE = 4  # vẽ ở độ phân giải gấp 4 rồi thu nhỏ INTER_AREA để giảm sai số làm tròn
PAIR_GLINT_RADIUS_FACTOR = 0.35  # bán kính mỗi đốm cặp = 0.35 * d_cluster (người dùng xác nhận) TODO_PILOT
RELIABLE_R_IRIS = [60, 80]  # chỉ các mức R này dùng để kết luận tổng hợp

# Dải kỳ vọng theo improve_implement_plan.md (mục 0.2)
K_CLUSTER_RANGE = (0.18, 0.25)
K_AREA_RANGE = (0.03, 0.06)

# Cấu hình ngưỡng sáng giống mô tả Module M4 (Percentile_95, sàn 230, đĩa trung tâm 60%)
PERCENTILE = 95.0
MIN_INTENSITY = 230.0
CENTRAL_DISC_RATIO = 0.60  # TODO_PILOT


def _stats(values: List[float]) -> Dict[str, Optional[float]]:
    if not values:
        return {"count": 0, "mean": None, "min": None, "max": None, "std": None}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": int(arr.size),
        "mean": round(float(arr.mean()), 4),
        "min": round(float(arr.min()), 4),
        "max": round(float(arr.max()), 4),
        "std": round(float(arr.std()), 4),
    }


def measure_glints(
    gray: np.ndarray, center: Tuple[float, float], r_iris: float
) -> Dict[str, List[float]]:
    """Đo k_area, k_aspect và k_cluster của các đốm sáng trong đĩa trung tâm mống mắt."""
    height, width = gray.shape[:2]
    cx, cy = center
    x1, y1 = max(0, int(math.floor(cx - r_iris))), max(0, int(math.floor(cy - r_iris)))
    x2, y2 = min(width, int(math.ceil(cx + r_iris))), min(height, int(math.ceil(cy + r_iris)))
    out: Dict[str, List[float]] = {"k_area": [], "k_aspect": [], "k_cluster": []}
    if x2 <= x1 or y2 <= y1:
        return out

    roi = gray[y1:y2, x1:x2].astype(np.float32)
    yy, xx = np.indices(roi.shape)
    dist = np.sqrt((x1 + xx - cx) ** 2 + (y1 + yy - cy) ** 2)
    iris_pixels = roi[dist <= r_iris]
    if iris_pixels.size == 0:
        return out

    threshold = max(MIN_INTENSITY, float(np.percentile(iris_pixels, PERCENTILE)))
    mask = ((roi >= threshold) & (dist <= r_iris * CENTRAL_DISC_RATIO)).astype(np.uint8)
    num, _, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

    points: List[Tuple[float, float]] = []
    for lbl in range(1, num):
        w = int(stats[lbl, cv2.CC_STAT_WIDTH])
        h = int(stats[lbl, cv2.CC_STAT_HEIGHT])
        area = int(stats[lbl, cv2.CC_STAT_AREA])
        out["k_area"].append(area / (r_iris ** 2))
        out["k_aspect"].append(float(max(w, h)) / float(max(1, min(w, h))))
        points.append((float(centroids[lbl, 0]), float(centroids[lbl, 1])))

    if len(points) >= 2:
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                out["k_cluster"].append(
                    math.hypot(points[i][0] - points[j][0], points[i][1] - points[j][1]) / r_iris
                )
    return out


def _render_base(r_iris: int) -> Tuple[np.ndarray, int, float]:
    """Nền siêu lấy mẫu (SUPERSAMPLE x): củng mạc, mống mắt, đồng tử. Trả về (ảnh lớn, kích thước gốc, tâm)."""
    size = int(r_iris * 3)
    cx = size / 2.0
    big = np.full((size * SUPERSAMPLE, size * SUPERSAMPLE), 200, dtype=np.uint8)  # củng mạc
    c_big = (int(cx * SUPERSAMPLE), int(cx * SUPERSAMPLE))
    cv2.circle(big, c_big, r_iris * SUPERSAMPLE, 90, -1, lineType=cv2.LINE_AA)  # mống mắt
    cv2.circle(big, c_big, max(2, int(r_iris * 0.4)) * SUPERSAMPLE, 25, -1, lineType=cv2.LINE_AA)  # đồng tử
    return big, size, cx


def _draw_glint(big: np.ndarray, point: Tuple[float, float], radius: float) -> None:
    """Vẽ đốm sáng ở độ phân giải siêu lấy mẫu để giảm sai số làm tròn bán kính."""
    cv2.circle(
        big,
        (int(round(point[0] * SUPERSAMPLE)), int(round(point[1] * SUPERSAMPLE))),
        max(1, int(round(radius * SUPERSAMPLE))),
        255,
        -1,
    )


def _downsample(big: np.ndarray, size: int) -> np.ndarray:
    return cv2.resize(big, (size, size), interpolation=cv2.INTER_AREA)


def render_single_glint(
    r_iris: int, rng: np.random.Generator
) -> Tuple[np.ndarray, Tuple[float, float], float]:
    """Một đốm phản quang chuẩn để đo k_area và k_aspect. Trả về (ảnh, tâm, k_area đặt)."""
    big, size, cx = _render_base(r_iris)
    k_area = float(rng.uniform(*K_AREA_RANGE))
    radius = math.sqrt(k_area * r_iris ** 2 / math.pi)
    angle = float(rng.uniform(0.0, 2.0 * math.pi))
    point = (cx + 0.05 * r_iris * math.cos(angle), cx + 0.05 * r_iris * math.sin(angle))
    _draw_glint(big, point, radius)
    return _downsample(big, size), (cx, cx), k_area


def render_pair_glints(
    r_iris: int, rng: np.random.Generator
) -> Tuple[np.ndarray, Tuple[float, float], float]:
    """Hai đốm cạnh tranh để đo k_cluster; bán kính mỗi đốm = PAIR_GLINT_RADIUS_FACTOR * d. Trả về (ảnh, tâm, k_cluster đặt)."""
    big, size, cx = _render_base(r_iris)
    k_cluster = float(rng.uniform(*K_CLUSTER_RANGE))
    d = k_cluster * r_iris
    radius = PAIR_GLINT_RADIUS_FACTOR * d
    angle = float(rng.uniform(0.0, 2.0 * math.pi))
    p1 = (cx + 0.05 * r_iris * math.cos(angle + 1.0), cx + 0.05 * r_iris * math.sin(angle + 1.0))
    p2 = (p1[0] + d * math.cos(angle), p1[1] + d * math.sin(angle))
    _draw_glint(big, p1, radius)
    _draw_glint(big, p2, radius)
    return _downsample(big, size), (cx, cx), k_cluster


def run_synthetic() -> Dict[str, object]:
    rng = np.random.default_rng(RANDOM_SEED)
    per_radius: Dict[str, object] = {}
    reliable: Dict[str, List[float]] = {
        "k_area_input": [], "k_area_measured": [], "k_aspect": [],
        "k_cluster_input": [], "k_cluster_measured": [],
    }

    for r_iris in SYNTHETIC_R_IRIS:
        k_area_in: List[float] = []
        k_area_out: List[float] = []
        k_area_err: List[float] = []
        k_aspect: List[float] = []
        for _ in range(SAMPLES_PER_RADIUS):
            img, center, k_in = render_single_glint(r_iris, rng)
            m = measure_glints(img, center, float(r_iris))
            if len(m["k_area"]) == 1:  # đúng một đốm được tách ra
                k_area_in.append(k_in)
                k_area_out.append(m["k_area"][0])
                k_area_err.append((m["k_area"][0] - k_in) / k_in)
                k_aspect.append(m["k_aspect"][0])

        k_cl_in: List[float] = []
        k_cl_out: List[float] = []
        resolved = 0
        for _ in range(SAMPLES_PER_RADIUS):
            img, center, k_in = render_pair_glints(r_iris, rng)
            m = measure_glints(img, center, float(r_iris))
            if len(m["k_cluster"]) == 1:  # đúng hai đốm tách rời
                resolved += 1
                k_cl_in.append(k_in)
                k_cl_out.append(m["k_cluster"][0])

        is_reliable = r_iris in RELIABLE_R_IRIS
        per_radius[str(r_iris)] = {
            "reliability": "RELIABLE" if is_reliable else "HIGH_ROUNDING_ERROR_LOW_R",
            "single_glint_samples": SAMPLES_PER_RADIUS,
            "single_glint_resolved": len(k_area_out),
            "k_area_input": _stats(k_area_in),
            "k_area_measured": _stats(k_area_out),
            "k_area_relative_error": _stats(k_area_err),
            "k_aspect": _stats(k_aspect),
            "pair_glint_samples": SAMPLES_PER_RADIUS,
            "pair_glint_resolved": resolved,
            "k_cluster_input": _stats(k_cl_in),
            "k_cluster_measured": _stats(k_cl_out),
        }
        if is_reliable:
            reliable["k_area_input"].extend(k_area_in)
            reliable["k_area_measured"].extend(k_area_out)
            reliable["k_aspect"].extend(k_aspect)
            reliable["k_cluster_input"].extend(k_cl_in)
            reliable["k_cluster_measured"].extend(k_cl_out)

    return {
        "source": "synthetic",
        "note": "Tự kiểm chứng bộ đo trên ảnh tổng hợp; không phải bằng chứng thực nghiệm. TODO_PILOT",
        "r_iris_px_levels": SYNTHETIC_R_IRIS,
        "reliable_r_iris_px_levels": RELIABLE_R_IRIS,
        "samples_per_radius": SAMPLES_PER_RADIUS,
        "random_seed": RANDOM_SEED,
        "supersample_factor": SUPERSAMPLE,
        "pair_glint_radius_factor": PAIR_GLINT_RADIUS_FACTOR,
        "input_ranges": {"k_cluster": list(K_CLUSTER_RANGE), "k_area": list(K_AREA_RANGE)},
        "per_r_iris": per_radius,
        "overall_reliable_levels": {key: _stats(vals) for key, vals in reliable.items()},
    }


def run_real() -> Dict[str, object]:
    result: Dict[str, object] = {"source": "real", "image": "tests/fixtures/sample_face.jpg"}
    if not SAMPLE_IMAGE_PATH.exists():
        result["status"] = "IMAGE_NOT_FOUND"
        return result
    try:
        from app.services.landmark_service import detect_landmarks  # đọc-chỉ

        # Nạp ảnh vào RAM, không ghi đĩa (G2)
        data = np.frombuffer(SAMPLE_IMAGE_PATH.read_bytes(), dtype=np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if image is None:
            result["status"] = "DECODE_FAILED"
            return result
        lm = detect_landmarks(image)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        eyes = {
            "OD": (lm.iris_center_od, lm.hvid_od_px / 2.0),
            "OS": (lm.iris_center_os, lm.hvid_os_px / 2.0),
        }
        per_eye: Dict[str, object] = {}
        for name, (center, r_iris) in eyes.items():
            m = measure_glints(gray, center, r_iris)
            per_eye[name] = {
                "r_iris_px": round(r_iris, 2),
                **{key: _stats(vals) for key, vals in m.items()},
            }
        result["status"] = "OK"
        result["per_eye"] = per_eye
        result["k_search_confirmed"] = CENTRAL_DISC_RATIO
    except Exception as exc:  # báo cáo lỗi đọc-chỉ, không dừng toàn script
        result["status"] = "MEASUREMENT_UNAVAILABLE"
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def main() -> int:
    report = {
        "phase": 0,
        "description": "Hệ số tương đối V2 theo R_iris (TODO_PILOT: căn chỉnh thêm khi thử nghiệm thực tế)",
        "definitions": {
            "k_cluster": "d_cluster / R_iris",
            "k_area": "Area_glare / R_iris^2",
            "k_aspect": "max(w, h) / min(w, h) của đốm phản quang",
            "k_search": "R_search / R_iris",
        },
        "k_search": {"value": CENTRAL_DISC_RATIO, "TODO_PILOT": True},
        "synthetic": run_synthetic(),
        "real": run_real(),
    }
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Calibration report written: {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
