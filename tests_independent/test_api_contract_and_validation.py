"""
Bộ kiểm thử độc lập cho Giao ước API REST & Kiểm tra Hợp lệ Đầu vào (API Contract & Validation).
Nguồn chân lý (Oracle):
- docs/06-system-architecture.md: § 9 (Đặc tả RESTful API), § 10
- docs/02-requirements.md: FR-005, FR-010, SP-001, NFR-008
- Chuẩn HTTP Semantics (RFC 9110)

Mục tiêu chính:
Kiểm tra tính kiên cố của Gateway trước các payload hỏng, tệp rỗng, sai định dạng,
và kiểm tra xem cấu trúc lỗi trả về có đúng hợp đồng OpenAPI / Architecture Spec hay không.
"""

import io
import re
import pytest
from fastapi.testclient import TestClient

BANNED_DIAGNOSTIC_TERMS = [
    "chẩn đoán", "bệnh nhân", "bệnh", "nhược cơ", "liệt",
    "horner", "bell", "myasthenia", "diagnosis", "patient", "diagnose"
]


class TestApiContractAndValidation:
    """Kiểm thử tính hợp lệ của API Gateway và tính toàn vẹn của giao ước."""

    def test_get_thresholds_contract(self, api_client: TestClient):
        """
        [IT-020]
        Kiểm tra endpoint GET /api/v1/config/thresholds:
        - Phải trả về HTTP 200 OK.
        - Chứa đầy đủ các trường ngưỡng cấu hình bắt buộc.
        - Mọi giá trị phải là số dương hợp lệ.
        """
        response = api_client.get("/api/v1/config/thresholds")
        assert response.status_code == 200
        data = response.json()

        expected_numeric_fields = [
            "absolute_severe_mrd1_mm",
            "tier_a_min_mrd1_mm",
            "tier_b_min_mrd1_mm",
            "physiological_asymmetry_max_mm",
            "borderline_asymmetry_max_mm",
            "brow_asymmetry_threshold_px",
            "max_yaw_deg",
            "max_pitch_deg",
            "max_roll_deg",
            "min_eye_width_px",
            "min_iris_diameter_px",
            "min_brightness",
            "max_brightness",
            "max_glare_ratio",
            "min_eye_aperture_ratio",
            "laplacian_sharp_threshold",
            "assumed_corneal_diameter_mm",
            "nominal_tolerance_mm",
        ]

        for field in expected_numeric_fields:
            assert field in data, f"Thiếu trường cấu hình bắt buộc: {field}"
            assert isinstance(data[field], (int, float)), f"Trường {field} phải là số: {type(data[field])}"
            assert data[field] > 0, f"Trường {field} phải có giá trị dương: {data[field]}"

    def test_analyze_empty_file_returns_400(self, api_client: TestClient):
        """
        [IT-010 / BR-012]
        Tải lên file có dung lượng 0 byte (empty file):
        Hệ thống phải trả về HTTP 400 Bad Request, không được ném 500 Internal Server Error.
        """
        files = {"file": ("empty.jpg", io.BytesIO(b""), "image/jpeg")}
        response = api_client.post("/api/v1/screening/analyze", files=files)

        assert response.status_code == 400, (
            f"Kỳ vọng HTTP 400 cho tệp rỗng, thực tế nhận: {response.status_code} - {response.text}"
        )
        data = response.json()
        assert "detail" in data

    def test_analyze_corrupt_payload_returns_400(self, api_client: TestClient):
        """
        [IT-011 / BR-012]
        Gửi dữ liệu nhị phân rác giả định dạng JPEG:
        Hệ thống phải từ chối giải mã an toàn với HTTP 400 Bad Request mà không crash.
        """
        corrupt_bytes = b"NOT_A_VALID_JPEG_HEADER_CORRUPTED_BYTES_1234567890"
        files = {"file": ("corrupt.jpg", io.BytesIO(corrupt_bytes), "image/jpeg")}
        response = api_client.post("/api/v1/screening/analyze", files=files)

        assert response.status_code == 400, (
            f"Kỳ vọng HTTP 400 cho dữ liệu hỏng, thực tế: {response.status_code}"
        )

    def test_analyze_missing_multipart_file_field_returns_422(self, api_client: TestClient):
        """
        [IT-009]
        Gửi request multipart form nhưng không có trường 'file' (hoặc gửi nhầm tên 'photo'):
        FastAPI phải bắt lỗi validation và trả về HTTP 422.
        """
        # Gửi form rỗng
        response_empty = api_client.post("/api/v1/screening/analyze", data={})
        assert response_empty.status_code == 422

        # Gửi sai tên trường
        response_wrong_key = api_client.post(
            "/api/v1/screening/analyze",
            files={"photo": ("image.jpg", io.BytesIO(b"dummy"), "image/jpeg")},
        )
        assert response_wrong_key.status_code == 422

    def test_analyze_rejection_payload_structure_contract(self, api_client: TestClient, make_solid_image_bytes):
        """
        [IT-018 / BR-013]
        Kiểm tra cấu trúc JSON trả về khi bị từ chối chất lượng (HTTP 422).
        Theo Đặc tả Kiến trúc 06-Arch § 9.1:
        Khi ảnh bị từ chối, client kỳ vọng nhận cấu trúc JSON với các trường:
        - error_code: 'QUALITY_GATE_REJECTED'
        - rejection_reason: (chuỗi định danh lý do)
        - rejection_message_vi: (hướng dẫn tiếng Việt)
        - audit_details: (thông số kỹ thuật)

        Kiểm tra xem response của code có khớp với giao ước đặc tả hay bị bọc lồng sai lệch.
        """
        # Ảnh 50x50 màu đen (vừa quá nhỏ LOW_RESOLUTION, vừa quá tối TOO_DARK)
        tiny_dark_img = make_solid_image_bytes(brightness=10, width=50, height=50)
        files = {"file": ("tiny.jpg", io.BytesIO(tiny_dark_img), "image/jpeg")}
        response = api_client.post("/api/v1/screening/analyze", files=files)

        assert response.status_code == 422, f"Kỳ vọng HTTP 422, nhận {response.status_code}"
        data = response.json()

        # Kiểm tra Hợp đồng § 9.1:
        # Nếu data có key 'detail' và bên trong mới là error_code, đây là điểm bất nhất API Contract
        if "detail" in data and isinstance(data["detail"], dict):
            inner = data["detail"]
            assert inner.get("error_code") == "QUALITY_GATE_REJECTED", (
                f"error_code bên trong detail không đúng: {inner}"
            )
            assert "rejection_reason" in inner
            assert "rejection_message_vi" in inner
            # Khẳng định sự bất nhất giữa API Spec (root-level) và Code thực tế (nested in detail)
            assert "error_code" in data, (
                "LỖI HỢP ĐỒNG API (API CONTRACT MISMATCH): 06-Arch § 9.1 quy định error_code nằm ở ROOT "
                "của JSON response, nhưng code FastAPI đang bọc lồng trong key 'detail'. "
                "Client React không thể đọc trực tiếp response.error_code!"
            )
        else:
            assert data.get("error_code") == "QUALITY_GATE_REJECTED"
            assert "rejection_reason" in data
            assert "rejection_message_vi" in data

    def test_security_headers_present_on_all_responses(self, api_client: TestClient):
        """
        [IT-015 / SP-001]
        Kiểm tra tính hiện diện của các Security Headers trên response theo SP-001:
        - X-Content-Type-Options: nosniff
        - X-Frame-Options: DENY
        - X-XSS-Protection: 1; mode=block
        """
        response = api_client.get("/health")
        assert response.status_code == 200
        headers = response.headers

        assert headers.get("x-content-type-options") == "nosniff", "Thiếu header X-Content-Type-Options: nosniff"
        assert headers.get("x-frame-options") == "DENY", "Thiếu header X-Frame-Options: DENY"
        assert "1; mode=block" in headers.get("x-xss-protection", ""), "Thiếu header X-XSS-Protection"

    def test_zero_diagnostic_language_in_api_error_responses(self, api_client: TestClient, make_solid_image_bytes):
        """
        [IT-013 / BR-010]
        Khẳng định các thông điệp lỗi trả về không chứa từ ngữ thuộc danh mục cấm (G4).
        """
        # Test lỗi quá tối
        dark_img = make_solid_image_bytes(brightness=20, width=400, height=400)
        resp = api_client.post(
            "/api/v1/screening/analyze",
            files={"file": ("dark.jpg", io.BytesIO(dark_img), "image/jpeg")},
        )
        assert resp.status_code == 422
        body_text = resp.text.lower()
        for banned in BANNED_DIAGNOSTIC_TERMS:
            pattern = r"\b" + re.escape(banned) + r"\b"
            assert not re.search(pattern, body_text), (
                f"Phát hiện từ cấm '{banned}' trong thông điệp từ chối: {body_text}"
            )
