"""
Helper utility phục vụ kiểm thử đơn vị và tích hợp.
Định vị tệp ảnh benchmark thực tế trên cả môi trường máy chủ và Docker container.
"""

from pathlib import Path

_CURRENT_DIR = Path(__file__).resolve().parent

CANDIDATE_PATHS = [
    _CURRENT_DIR / "fixtures" / "sample_face.jpg",
    Path("/app/tests/fixtures/sample_face.jpg"),
    Path("tests/fixtures/sample_face.jpg"),
    Path("/app/HIRSCHBERG_METHOD/harschberg_data_detect/normal_harschberg/img_023e6caea47f18ee__normal_truc.jpg"),
    Path("HIRSCHBERG_METHOD/harschberg_data_detect/normal_harschberg/img_023e6caea47f18ee__normal_truc.jpg"),
    Path("../HIRSCHBERG_METHOD/harschberg_data_detect/normal_harschberg/img_023e6caea47f18ee__normal_truc.jpg"),
    Path("/app/HIRSCHBERG_METHOD/hirschberg_review_all_v1/normal/img_023e6caea47f18ee__normal_truc.jpg"),
    Path("HIRSCHBERG_METHOD/hirschberg_review_all_v1/normal/img_023e6caea47f18ee__normal_truc.jpg"),
    Path("../HIRSCHBERG_METHOD/hirschberg_review_all_v1/normal/img_023e6caea47f18ee__normal_truc.jpg"),
]


def get_sample_image_path() -> Path:
    """Trả về đường dẫn tới tệp ảnh benchmark thực tế tồn tại trên hệ thống."""
    for p in CANDIDATE_PATHS:
        if p.exists():
            return p
    return CANDIDATE_PATHS[0]
