# RemiCare PtosisCheck — Backend API Engine

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue?logo=python)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688?logo=fastapi)](https://fastapi.tiangolo.com/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8?logo=opencv)](https://opencv.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Face%20Mesh%20v2-FF6F00?logo=google)](https://developers.google.com/mediapipe)
[![Pytest](https://img.shields.io/badge/Tests-110%20Passing-brightgreen?logo=pytest)](https://pytest.org/)
[![Security & Privacy](https://img.shields.io/badge/Privacy-Zero--Retention%20In--Memory-success)]()

Hệ thống API backend hiệu năng cao phục vụ **Sàng lọc Hình thái Mi mắt (Ptosis Morphology Screening)**. Ứng dụng công nghệ Thị giác Máy tính (Computer Vision), mô hình điểm mốc giải phẫu khuôn mặt độ phân giải cao và thuật toán phân tầng ưu tiên tính bất đối xứng hai bên (**Asymmetry-First Triage Engine**).

> [!IMPORTANT]
> **THÔNG CÁO AN TOÀN PHI CHẨN ĐOÁN (NON-DIAGNOSTIC NOTICE):**
> Hệ thống được thiết kế phục vụ mục đích sàng lọc hình thái học ban đầu và hỗ trợ nhận biết sớm sự bất đối xứng mi mắt cho cộng đồng. Hệ thống **tuyệt đối không đưa ra kết luận y khoa** và **không thay thế quy trình thăm khám trực tiếp** của bác sĩ chuyên khoa mắt.

---

## Mục lục

1. [Kiến trúc & Đường ống xử lý ảnh (M1 - M8)](#kiến-trúc--đường-ống-xử-lý-ảnh-m1---m8)
2. [Phương pháp đo đạc hình thái học (MRD1, MRD2, PFD/PFH, ΔMRD1)](#phương-pháp-đo-đạc-hình-thái-học)
3. [Cây quyết định phân tầng sàng lọc (Mức A, B, C)](#cây-quyết-định-phân-tầng-sàng-lọc)
4. [Nguyên tắc cốt lõi & Bảo vệ quyền riêng tư](#nguyên-tắc-cốt-lõi--bảo-vệ-quyền-riêng-tư)
5. [Hướng dẫn cài đặt & Khởi chạy](#hướng-dẫn-cài-đặt--khởi-chạy)
6. [Tài liệu API & Endpoint](#tài-liệu-api--endpoint)
7. [Kiểm thử tự động & Kiểm toán an toàn](#kiểm-thử-tự-động--kiểm-toán-an-toàn)

---

## Kiến trúc & Đường ống xử lý ảnh (M1 - M8)

Hệ thống xử lý ảnh đơn nhất qua đường ống nối tiếp gồm 8 module chuyên biệt:

```
[ Ảnh tải lên ] 
       │
       ▼
 [ Module M1: Tiền xác thực ] ──► (Kiểm tra định dạng JPEG/PNG, dung lượng 50KB - 15MB)
       │
       ▼
 [ Module M2: Cổng kiểm định chất lượng (Quality Gate) ] 
       │   ├── Tư thế đầu: Yaw ≤ 10°, Pitch ≤ 10°, Roll ≤ 5°
       │   ├── Độ sắc nét: Laplacian Variance ≥ 50.0
       │   ├── Độ rọi & Ánh xạ: Min brightness ≥ 40, Max ≤ 220, Chói lóa ≤ 15%
       │   └── Khoảng cách & Mở mắt: Khẩu độ khe mi ≥ 0.15, Đường kính mống mắt ≥ 30px
       │
       ▼
 [ Module M3: Trích xuất mốc hình học & Mống mắt ] ──► (MediaPipe 478 landmarks + Iris tracking)
       │
       ▼
 [ Module M4: Định vị phản quang giác mạc Purkinje ] ──► (Tìm phản xạ R1; Fallback: Đồng tử / Tâm mống mắt)
       │
       ▼
 [ Module M5: Phân đoạn bờ sụn mi trên ] ──► (Lọc gradient Sobel-Y dọc + Nội suy Spline bậc 3)
       │
       ▼
 [ Module M6: Động cơ tính toán chỉ số hình thái ] ──► (MRD1, MRD2, PFD, ΔMRD1, ΔBrow, Dung sai ±0.5mm)
       │
       ▼
 [ Module M7: Động cơ phân tầng nguy cơ Asymmetry-First ] ──► (Cây quyết định 3 mức: TIER_A, TIER_B, TIER_C)
       │
       ▼
 [ Module M8: Tạo tọa độ lớp phủ trực quan ] ──► (Phản quang, đường viền mống mắt, bờ mi để vẽ Canvas)
       │
       ▼
[ MeasurementPayload JSON ]
```

---

## Phương pháp đo đạc hình thái học

### 1. Thước đo giải phẫu quy chuẩn (White-to-White Scale)

Để chuyển đổi từ tọa độ điểm ảnh (pixels) sang milimét (mm) thực tế mà không cần người dùng đặt vật chuẩn ngoại vi (thước kẻ hoặc đồng xu), hệ thống sử dụng **đường kính ngang mống mắt chuẩn** (Corneal Diameter / White-to-White) của người trưởng thành Đông Á:

$$
D_{\text{iris}} = 11.71\text{ mm}
$$

Hệ số quy đổi $\text{Scale}$ trên từng mắt được tính:

$$
\text{Scale}_{\text{OD}} = \frac{11.71}{D_{\text{iris, OD}}} \quad (\text{mm/px}), \qquad \text{Scale}_{\text{OS}} = \frac{11.71}{D_{\text{iris, OS}}} \quad (\text{mm/px})
$$

---

### 2. MRD1 (Margin Reflex Distance 1)
- **Định nghĩa:** Khoảng cách thẳng đứng từ tâm phản xạ giác mạc Purkinje ($R_1$) lên đến điểm đáy thấp nhất của bờ mi trên tại trục nhìn.
- **Điểm tham chiếu phản quang ($y_{\text{ref}}$):**
  1. *Cấp 1 (Chuẩn):* Phản xạ ánh sáng Purkinje thứ nhất ($R_1$).
  2. *Cấp 2 (Dự phòng 1):* Tâm đồng tử hình học (phát hiện qua Otsu Thresholding + RANSAC Ellipse Fitting).
  3. *Cấp 3 (Dự phòng 2):* Tâm mống mắt MediaPipe.
- **Công thức tính:** $MRD1_{\text{px}} = y_{\text{ref}} - y_{\text{upper}}$ và $MRD1_{\text{mm}} = MRD1_{\text{px}} \times \text{Scale}$.
  *(Lưu ý: Trong hệ trục tọa độ hình ảnh số, trục $Y$ hướng xuống, do đó viền mi trên có tọa độ $y$ nhỏ hơn tâm phản quang).*
- **Khoảng ước lượng danh định:** $MRD1_{\text{range}} = \left[ MRD1_{\text{mm}} - 0.5,\; MRD1_{\text{mm}} + 0.5 \right]\text{ mm}$ *(phản ánh dung sai giải phẫu học và góc chiếu sáng $\pm 0.5\text{ mm}$)*.

---

### 3. MRD2 (Margin Reflex Distance 2)
- **Định nghĩa:** Khoảng cách thẳng đứng từ tâm phản quang giác mạc ($R_1$) xuống đến viền bờ mi dưới tại trục nhìn.
- **Công thức tính:** $MRD2_{\text{mm}} = (y_{\text{lower}} - y_{\text{ref}}) \times \text{Scale}$.

---

### 4. PFD / PFH (Palpebral Fissure Height / Độ mở khe mi)
- **Định nghĩa:** Tổng độ cao khe mi theo phương thẳng đứng từ bờ mi dưới lên bờ mi trên:

$$
PFD = MRD1 + MRD2 = (y_{\text{lower}} - y_{\text{upper}}) \times \text{Scale}
$$

---

### 5. $\Delta MRD1$ (Chỉ số bất đối xứng hai bên)
- **Định nghĩa:** Độ chênh lệch tuyệt đối về độ cao mi trên giữa mắt trái (OS) và mắt phải (OD):

$$
\Delta MRD1 = \lvert MRD1_{\text{OS}} - MRD1_{\text{OD}} \rvert
$$

- **Bảng phân loại mức độ đối xứng:**
  | Khoảng giá trị $\Delta MRD1$ | Phân loại hình thái | Ý nghĩa lâm sàng |
  | :--- | :--- | :--- |
  | $\le 1.5\text{ mm}$ | `PHYSIOLOGICAL_SYMMETRY` | Đối xứng sinh lý bình thường |
  | $1.5\text{ mm} < \Delta MRD1 \le 2.0\text{ mm}$ | `BORDERLINE_ASYMMETRY` | Bất đối xứng ranh giới / cần theo dõi |
  | $> 2.0\text{ mm}$ | `SIGNIFICANT_ASYMMETRY` | **Bất đối xứng rõ rệt** $\rightarrow$ Kích hoạt cảnh báo Mức C |

---

### 6. $\Delta Brow$ (Độ chênh lệch cung mày / Co cơ trán bù trừ)
- Khi một bên mi bị sụp, người dùng thường có phản xạ tự nhiên co cơ trán bên đó để kéo mi mắt lên, làm cung mày bị nhướng cao bất thường:

$$
\Delta Brow = \lvert H_{\text{brow, OS}} - H_{\text{brow, OD}} \rvert \quad (\text{pixels})
$$

- Nếu $\Delta Brow > 8.0\text{ px}$, hệ thống sẽ kích hoạt cờ nghi ngờ có co cơ trán nâng mi bù trừ (`BROW_COMPENSATION_SUSPECTED`).

---

## Cây quyết định phân tầng sàng lọc

Cây quyết định hoạt động theo triết lý **Ưu tiên Bất đối xứng (Asymmetry-First)** với **Chốt chặn Ngoại lệ Che lấp Trục nhìn**:

```
[ Bắt đầu đánh giá ]
       │
       ├─► [NGOẠI LỆ 1] Bờ mi che lấp hoàn toàn phản quang? ──► CÓ ──► [ MỨC C (TIER_C) ]
       │
       ├─► [NGOẠI LỆ 2] min(MRD1) < 1.0 mm (bất kể ΔMRD1)? ──► CÓ ──► [ MỨC C (TIER_C) ]
       │
       ▼ (Không vi phạm ngoại lệ: min(MRD1) ≥ 1.0 mm)
       │
       ├─► [ƯU TIÊN 1] ΔMRD1 > 2.0 mm (Bất đối xứng rõ rệt)? ──► CÓ ──► [ MỨC C (TIER_C) ]
       │               (Lúc này mới xét chi tiết MRD1 2 mắt để phân định mức độ)
       │
       ├─► [ƯU TIÊN 2] 1.5 mm < ΔMRD1 ≤ 2.0 mm HOẶC ΔBrow > 8 px? ──► CÓ ──► [ MỨC B (TIER_B) ]
       │
       ▼
 [ MỨC A (TIER_A): ΔMRD1 ≤ 1.5 mm VÀ min(MRD1) ≥ 1.0 mm (Sinh lý bình thường) ]
```

### Bảng chi tiết 3 mức khuyến nghị:

| Mức phân tầng | Tên gọi chuẩn | Tiêu chí kích hoạt chính | Khuyến nghị hành động đại chúng |
| :---: | :--- | :--- | :--- |
| **Mức A**<br>`TIER_A` | **Không phát hiện bất thường** | $\Delta MRD1 \le 1.5\text{ mm}$ VÀ $\min(MRD1) \ge 1.0\text{ mm}$ VÀ $\Delta Brow \le 8\text{ px}$. | Chưa ghi nhận dấu hiệu cần can thiệp. Tiếp tục duy trì sinh hoạt lành mạnh và theo dõi định kỳ. |
| **Mức B**<br>`TIER_B` | **Dấu hiệu nghi ngờ / Ranh giới** | $1.5\text{ mm} < \Delta MRD1 \le 2.0\text{ mm}$; HOẶC $\Delta Brow > 8.0\text{ px}$. | Ghi nhận độ cao mi ở mức ranh giới hoặc có chênh lệch nhẹ. Khuyến nghị theo dõi thêm và cân nhắc thăm khám khi có dịp. |
| **Mức C**<br>`TIER_C` | **Khuyến nghị sớm thăm khám chuyên khoa** | **$\Delta MRD1 > 2.0\text{ mm}$**; HOẶC **ngoại lệ $\min(MRD1) < 1.0\text{ mm}$** (kể cả khi $\Delta \approx 0$ hay $\Delta < 2.0\text{ mm}$); HOẶC phản quang giác mạc bị che lấp. | Bờ mi che phủ đáng kể trục thị giác hoặc có sự bất đối xứng rõ rệt. Khuyến nghị sớm đến cơ sở y tế chuyên khoa mắt để được thăm khám trực tiếp. |

---

## Nguyên tắc cốt lõi & Bảo vệ quyền riêng tư

- **[G1] Client-Server Separation:** Tách biệt tuyệt đối Backend (FastAPI CV Engine) và Frontend (React SPA). Giao tiếp chuẩn hóa qua RESTful JSON.
- **[G2] Zero-Retention In-Memory:** Tệp ảnh tải lên được giải mã và xử lý 100% trong bộ nhớ đệm RAM (`io.BytesIO`). **Tuyệt đối không có lệnh ghi tệp ảnh nào xuống đĩa cứng hoặc cơ sở dữ liệu tạm thời**.
- **[G3] Asymmetry-First & Bilateral Fail-Safe:** Luôn đánh giá tương quan hai mắt trước khi kết luận đơn mắt; bảo vệ người có dáng mắt hẹp bẩm sinh không bị báo động giả.
- **[G4] Non-Diagnostic Safety Ban:** Tuyệt đối không dùng từ ngữ chẩn đoán bệnh học hoặc gán nhãn bệnh danh. Toàn bộ chuỗi văn bản được kiểm toán tự động qua test suite `test_diagnostic_language_ban.py`.
- **[G5] Strict API Contract:** 100% phản hồi chuẩn hóa theo schema Pydantic `MeasurementPayload`.

---

## Hướng dẫn cài đặt & Khởi chạy

### Yêu cầu tiên quyết
- **Python:** 3.10, 3.11 hoặc 3.12 (khuyến nghị 3.12).
- **Hệ điều hành:** Windows 10/11, Ubuntu 20.04+, hoặc macOS 12+.

---

### Phương án 1: Chạy trực tiếp trên máy host (Khuyến nghị cho phát triển)

#### 1. Điều hướng vào thư mục backend:
```bash
cd backend
```

#### 2. Khởi tạo và kích hoạt môi trường ảo:
- **Trên Windows (PowerShell):**
  ```powershell
  python -m venv .venv
  .\.venv\Scripts\Activate.ps1
  ```
- **Trên Linux / macOS:**
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  ```

#### 3. Cài đặt các thư viện phụ thuộc:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

#### 4. Khởi chạy Uvicorn Development Server:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
Sau khi khởi chạy thành công, API sẵn sàng lắng nghe tại `http://localhost:8000`.

---

### Phương án 2: Khởi chạy bằng Docker & Docker Compose

```bash
# 1. Xây dựng image và chạy container ở chế độ nền
docker compose up -d --build

# 2. Kiểm tra trạng thái hoạt động của container
docker compose ps

# 3. Xem log thời gian thực
docker compose logs -f backend

# 4. Dừng container khi hoàn thành
docker compose down
```

---

## Tài liệu API & Endpoint

Sau khi khởi chạy server, bạn có thể truy cập các tài liệu tương tác:
- **Interactive Swagger UI:** `http://localhost:8000/docs`
- **ReDoc Documentation:** `http://localhost:8000/redoc`
- **OpenAPI Schema:** `http://localhost:8000/openapi.json`

### Danh sách Endpoint chính:

| Phương thức | Đường dẫn | Chức năng | Định dạng Input / Output |
| :---: | :--- | :--- | :--- |
| `POST` | `/api/v1/screening/analyze` | Phân tích ảnh chụp khuôn mặt, trích xuất chỉ số hình thái và phân tầng Mức A/B/C. | `multipart/form-data` (`file`) $\rightarrow$ `MeasurementPayload` (JSON) |
| `GET` | `/api/v1/config/thresholds` | Trả về bảng các tham số ngưỡng cấu hình lâm sàng hiện hành. | Không $\rightarrow$ `JSON` (Bảng ngưỡng) |
| `GET` | `/health` | Kiểm tra trạng thái hoạt động của dịch vụ (Health Check). | Không $\rightarrow$ `{"status": "ok"}` |

---

## Kiểm thử tự động & Kiểm toán an toàn

Toàn bộ backend được bảo vệ bởi bộ kiểm thử tự động toàn diện với **110 bài test**:

```bash
# Kích hoạt môi trường ảo nếu chưa kích hoạt
# Chạy toàn bộ 110 bài test đơn vị, tích hợp và luồng phân tầng
pytest tests/ -v -p no:cacheprovider

# Chạy kiểm toán danh mục từ ngữ cấm theo tiêu chuẩn phi chẩn đoán G4
pytest tests/audit/test_diagnostic_language_ban.py -v

# Chạy kiểm toán nguyên tắc Zero-Retention (không ghi đĩa)
pytest tests/audit/test_zero_retention_audit.py -v
```

---

## Bản quyền & Giấy phép
Dự án được xây dựng phục vụ nghiên cứu và phát triển sàng lọc y tế số vì cộng đồng. Xem chi tiết điều khoản sử dụng trong tài liệu tổng thể của dự án.
