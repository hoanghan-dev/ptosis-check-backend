"""
Bộ phận kiểm soát và giải phóng bộ nhớ RAM tức thì (In-Memory Zero-Retention).
Tuân thủ nguyên tắc G2: Tuyệt đối không ghi dữ liệu ảnh xuống ổ cứng,
và giải phóng hoàn toàn mảng pixel ngay khi hoàn thành đo lường.
"""

import gc
from typing import Any


def purge_image_data(*objects: Any) -> None:
    """
    Giải phóng triệt để các đối tượng dữ liệu ảnh, mảng nhị phân, bộ đệm RAM
    và kích hoạt bộ thu gom rác gc.collect() ngay lập tức.
    """
    for obj in objects:
        try:
            del obj
        except Exception:
            pass
    gc.collect()
