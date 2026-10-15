"""
Router tổng hợp các API v1 cho hệ thống PtosisCheck.
"""

from fastapi import APIRouter
from app.api.v1.endpoints import screening, config

api_router = APIRouter()

api_router.include_router(screening.router, prefix="/screening", tags=["Screening"])
api_router.include_router(config.router, prefix="/config", tags=["Configuration"])
