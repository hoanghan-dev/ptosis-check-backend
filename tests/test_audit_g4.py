"""
Kiểm toán tuân thủ G4: Tuyệt đối cấm từ ngữ chẩn đoán và thuật ngữ y tế cấm trong codebase.
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


def test_zero_diagnostic_language_in_codebase():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    workspace_dir = os.path.dirname(base_dir)

    targets = [
        os.path.join(workspace_dir, "backend", "app"),
        os.path.join(workspace_dir, "frontend", "src"),
    ]

    violations = []

    for target in targets:
        for root, _, files in os.walk(target):
            for f in files:
                if f.endswith((".py", ".ts", ".tsx", ".json", ".html")):
                    filepath = os.path.join(root, f)
                    with open(filepath, "r", encoding="utf-8") as fh:
                        for line_num, line in enumerate(fh, 1):
                            for bw in BANNED_WORDS:
                                pattern = r"(?i)\b" + re.escape(bw) + r"\b"
                                if re.search(pattern, line):
                                    rel_path = os.path.relpath(filepath, workspace_dir)
                                    violations.append((rel_path, line_num, bw, line.strip()))

    assert len(violations) == 0, f"Phát hiện vi phạm G4: {violations}"
