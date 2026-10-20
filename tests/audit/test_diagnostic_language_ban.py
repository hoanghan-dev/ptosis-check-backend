"""
Kiểm toán độc lập tuân thủ nguyên tắc G4 (Non-Diagnostic Safety).
Quét toàn diện 100% tệp mã nguồn Backend (Python) và Frontend (TypeScript / React).
Khẳng định không chứa bất kỳ từ ngữ nào thuộc danh mục cấm:
'chẩn đoán', 'bệnh nhân', 'bệnh', 'nhược cơ', 'liệt', 'horner', 'bell', 'myasthenia',
'diagnosis', 'patient', 'diagnose'.
"""

import os
import re

BANNED_WORDS = [
    "chẩn đoán",
    "bệnh nhân",
    "bệnh",
    "nhược cơ",
    "liệt",
    "horner",
    "bell",
    "myasthenia",
    "diagnosis",
    "patient",
    "diagnose",
]


def test_strict_diagnostic_language_ban_across_codebase():
    """Kiểm tra toàn bộ codebase không vi phạm điều khoản cấm ngôn ngữ chẩn đoán (FR-024, FR-025, G4)."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    backend_tests_dir = os.path.dirname(current_dir)
    backend_dir = os.path.dirname(backend_tests_dir)
    root_dir = os.path.dirname(backend_dir)

    # Các thư mục chứa mã nguồn của hệ thống (hỗ trợ cả repo độc lập và monorepo)
    scan_targets = [
        os.path.join(backend_dir, "app"),
        os.path.join(root_dir, "backend", "app"),
        os.path.join(root_dir, "frontend", "src"),
    ]

    # Các thư mục chạy trong Docker container nếu chạy từ /app
    container_targets = [
        "/app/app",
    ]

    active_targets = list(dict.fromkeys([t for t in scan_targets if os.path.exists(t)]))
    if not active_targets:
        active_targets = [t for t in container_targets if os.path.exists(t)]

    assert len(active_targets) > 0, "Không tìm thấy thư mục mã nguồn để thực hiện kiểm toán G4!"

    violations = []

    for target in active_targets:
        for root, _, files in os.walk(target):
            # Bỏ qua thư mục node_modules, build, .git, v.v.
            if any(ignored in root for ignored in ["node_modules", "dist", ".git", "__pycache__", ".venv"]):
                continue

            for f in files:
                if f.endswith((".py", ".ts", ".tsx", ".html")):
                    filepath = os.path.join(root, f)
                    with open(filepath, "r", encoding="utf-8", errors="ignore") as fh:
                        for line_num, line in enumerate(fh, 1):
                            for bw in BANNED_WORDS:
                                pattern = r"(?i)\b" + re.escape(bw) + r"\b"
                                if re.search(pattern, line):
                                    rel_path = os.path.relpath(filepath, root_dir) if os.path.exists(root_dir) else filepath
                                    violations.append((rel_path, line_num, bw, line.strip()))

    assert len(violations) == 0, f"Phát hiện {len(violations)} vi phạm quy tắc G4: {violations}"
