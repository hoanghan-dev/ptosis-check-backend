"""
Điểm khởi chạy ứng dụng FastAPI PtosisCheck.
Tuân thủ G1: Phân tách Client - Server rõ ràng.
Tuân thủ G2: Zero retention trên RAM.
Tuân thủ G4: Tuyệt đối không dùng từ ngữ thuộc danh mục cấm.
"""

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
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
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Middleware bổ sung các tiêu đề bảo mật HTTP và HSTS cho đường truyền an toàn (SP-001).
    """

    async def dispatch(self, request: Request, call_next):
        response: Response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        return response


app.add_middleware(SecurityHeadersMiddleware)

# Đăng ký Router chính
app.include_router(api_router, prefix=settings.API_V1_PREFIX)


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
