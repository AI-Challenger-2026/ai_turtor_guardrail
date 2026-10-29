"""GET /api/health — 서버 상태 (liveness)."""

from __future__ import annotations

from fastapi import APIRouter, Request

from app import __version__
from app.schemas import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    return HealthResponse(status="ok", version=__version__, mock_mode=request.app.state.settings.mock_mode)
