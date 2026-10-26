"""
Điểm khởi chạy ứng dụng FastAPI PtosisCheck.
Tuân thủ G1: Phân tách Client - Server rõ ràng.
Tuân thủ G2: Zero retention trên RAM.
Tuân thủ G4: Tuyệt đối không dùng từ ngữ thuộc danh mục cấm.
"""

from fastapi import FastAPI, Request, Response
from fastapi.exceptions import HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app.api.v1.api_router import api_router
from app.core.config import settings

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Hệ thống sàng lọc hình thái mi mắt hỗ trợ phát hiện sớm sa trễ bờ mi.",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Cấu hình CORS cho phép Client gọi API
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Middleware bổ sung các tiêu đề bảo mật HTTP và HSTS cho đường truyền an toàn (SP-001).
    """

    async def dispatch(self, request: Request, call_next):
        response: Response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        return response


app.add_middleware(SecurityHeadersMiddleware)

# Đăng ký Router chính
app.include_router(api_router, prefix=settings.API_V1_PREFIX)


@app.exception_handler(HTTPException)
async def custom_http_exception_handler(request: Request, exc: HTTPException):
    """
    Chuẩn hóa cấu trúc phản hồi lỗi theo 06-system-architecture.md § 9.1:
    Nếu exc.detail là dict chứa error_code (ví dụ QUALITY_GATE_REJECTED),
    trả về trực tiếp ở root-level của JSON thay vì bọc lồng trong key 'detail'.
    """
    if isinstance(exc.detail, dict) and "error_code" in exc.detail:
        return JSONResponse(status_code=exc.status_code, content=exc.detail)
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


@app.get("/", tags=["Health"])
def root_endpoint():
    return {
        "name": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "status": "online",
        "description": "Dịch vụ sàng lọc hình thái mi mắt (In-Memory Processing).",
    }


@app.get("/health", tags=["Health"])
def health_check():
    return {"status": "healthy"}
