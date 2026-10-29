"""GET /api/config — 실제 적용 중인 설정과 모듈 상태 (enabled / disabled / not_implemented)."""

from __future__ import annotations

from fastapi import APIRouter, Request

from app.schemas import ConfigResponse

router = APIRouter(tags=["config"])


@router.get("/config", response_model=ConfigResponse)
def get_config(request: Request) -> ConfigResponse:
    return request.app.state.service.config_snapshot()
