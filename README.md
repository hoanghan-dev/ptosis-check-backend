# PtosisCheck Backend API

Hệ thống API Sàng lọc Hình thái Mi mắt (Ptosis Morphology Screening Engine).

## Kiến trúc & Nguyên tắc cốt lõi
- **G1 (Client-Server Separation)**: Backend thuần Python/FastAPI chịu trách nhiệm Computer Vision pipeline & Triage engine.
- **G2 (Zero-Retention In-Memory)**: 100% xử lý ảnh trong bộ nhớ RAM tạm thời qua `io.BytesIO`, tuyệt đối không ghi bất kỳ tệp ảnh nào xuống đĩa cứng.
- **G4 (Non-Diagnostic Safety)**: Không trả về chẩn đoán y khoa hoặc bệnh danh, phân tầng nguy cơ theo Mức A/B/C dựa trên độ mở và bất đối xứng mi mắt.
- **G5 (Strict API Contract)**: Chuẩn hóa 100% định dạng theo `MeasurementPayload`.

## Khởi chạy nhanh bằng Docker Compose

```bash
# 1. Khởi động dịch vụ trong nền
docker compose up -d --build

# 2. Xem trạng thái container
docker compose ps

# 3. Xem log thời gian thực
docker compose logs -f backend

# 4. Chạy toàn bộ 71 kiểm thử đơn vị & tích hợp
docker compose exec backend pytest -v /app/tests

# 5. Dừng dịch vụ
docker compose down
```

## Các cổng kết nối
- **API Server**: `http://localhost:8000`
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`
- **Bảng ngưỡng lâm sàng**: `http://localhost:8000/api/v1/config/thresholds`

## Chạy trực tiếp trên máy host (Local Python)

Yêu cầu Python 3.10+:
```bash
python -m venv .venv
source .venv/bin/activate  # Hoặc .venv\Scripts\activate trên Windows
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
