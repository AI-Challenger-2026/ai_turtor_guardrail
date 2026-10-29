"""엘리스 ML API 제공자 자리 (Elice ML API provider — placeholder).

대회 개발 환경(엘리스 Helpy Code, ML API)이 공개되면 여기를 구현한다.
연결 지점 (integration points):
  1. ELICE_API_BASE, ELICE_API_KEY 환경변수로 엔드포인트와 키를 받는다 (settings.py 에 이미 정의됨).
  2. _complete() 에서 HTTP 요청을 보내고 RawCompletion(text, input_tokens, output_tokens, stop_reason)을 채운다.
  3. 인증 실패/429/타임아웃을 app.errors 의 LLMAuthError/LLMRateLimitError/LLMTimeoutError 로 변환한다.
  4. GENERATOR_PROVIDER=elice 또는 JUDGE_PROVIDER=elice 로 선택한다 (factory.py).
현재는 선택되면 "미구현" 오류(HTTP 501)를 명확히 반환한다.
"""

from __future__ import annotations

from app.errors import LLMNotImplementedError
from app.llm.base import LLMProvider, ProviderOptions, RawCompletion


class EliceProvider(LLMProvider):
    name = "elice"

    def __init__(self, model: str, options: ProviderOptions | None = None, *, api_key: str = "", api_base: str = "") -> None:
        super().__init__(model, options)
        self.api_base = api_base
        self._has_key = bool(api_key)

    def _complete(self, system, messages, temperature, max_tokens, json_mode, task) -> RawCompletion:  # noqa: ANN001
        raise LLMNotImplementedError(
            "엘리스 ML API 연동은 아직 구현되지 않았습니다. GENERATOR_PROVIDER/JUDGE_PROVIDER 를 anthropic 으로 바꾸거나 "
            "MOCK_MODE=true 로 실행해 주세요.",
            detail="EliceProvider._complete is not implemented yet",
        )
