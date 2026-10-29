"""애플리케이션 예외 계층 (application error hierarchy).

모든 예외는 사용자용 한국어 메시지(user_message)와 HTTP 상태 코드(http_status), 기계용 코드(code)를 가진다.
API 계층은 이 정보만 사용해 오류 응답을 만들고, 내부 상세(detail)는 로그에만 남긴다.
"""

from __future__ import annotations


class AppError(Exception):
    code: str = "internal_error"
    http_status: int = 500
    user_message: str = "알 수 없는 오류가 발생했습니다. 잠시 후 다시 시도해 주세요."
    retryable: bool = False

    def __init__(
        self,
        user_message: str | None = None,
        *,
        detail: str | None = None,
        code: str | None = None,
        http_status: int | None = None,
    ) -> None:
        if user_message:
            self.user_message = user_message
        if code:
            self.code = code
        if http_status:
            self.http_status = http_status
        self.detail = detail
        super().__init__(detail or self.user_message)


class InvalidRequestError(AppError):
    code = "invalid_request"
    http_status = 400
    user_message = "요청 형식이 올바르지 않습니다."


class ConfigError(AppError):
    code = "config_error"
    http_status = 500
    user_message = "서버 설정 오류가 있습니다. 관리자에게 문의해 주세요."


# ---------------------------------------------------------------------------
# LLM 호출 오류 (LLM provider errors)
# ---------------------------------------------------------------------------


class LLMError(AppError):
    code = "llm_error"
    http_status = 502
    user_message = "AI 서비스 호출 중 오류가 발생했습니다. 잠시 후 다시 시도해 주세요."


class LLMAuthError(LLMError):
    code = "llm_auth_error"
    http_status = 502
    user_message = "AI 서비스 인증에 실패했습니다. 서버의 API 키 설정을 확인해 주세요."


class LLMRateLimitError(LLMError):
    code = "llm_rate_limited"
    http_status = 429
    retryable = True
    user_message = "AI 서비스 사용량 한도를 초과했습니다. 잠시 후 다시 시도해 주세요."


class LLMTimeoutError(LLMError):
    code = "llm_timeout"
    http_status = 504
    retryable = True
    user_message = "AI 응답 시간이 초과되었습니다. 잠시 후 다시 시도해 주세요."


class LLMConnectionError(LLMError):
    code = "llm_connection_error"
    http_status = 503
    retryable = True
    user_message = "AI 서비스에 연결할 수 없습니다. 네트워크 상태를 확인하고 다시 시도해 주세요."


class LLMAPIError(LLMError):
    code = "llm_api_error"
    http_status = 502
    user_message = "AI 서비스에서 오류가 발생했습니다. 잠시 후 다시 시도해 주세요."

    def __init__(self, user_message: str | None = None, *, retryable: bool = False, **kwargs) -> None:
        super().__init__(user_message, **kwargs)
        self.retryable = retryable


class LLMNotImplementedError(LLMError):
    code = "provider_not_implemented"
    http_status = 501
    user_message = "선택한 AI 제공자는 아직 구현되지 않았습니다."


class LLMConfigError(LLMError):
    code = "llm_config_error"
    http_status = 500
    user_message = "AI 제공자 설정이 올바르지 않습니다. 서버 환경변수를 확인해 주세요."


class LLMJSONError(LLMError):
    code = "llm_bad_json"
    http_status = 502
    user_message = "AI 판정 결과를 해석하지 못했습니다. 다시 시도해 주세요."


class LLMRefusalError(LLMError):
    """모델 자체 안전 장치가 응답을 거절한 경우 (stop_reason == "refusal")."""

    code = "llm_refusal"
    http_status = 200
    user_message = "AI 모델이 이 요청에 대한 응답을 거절했습니다."

    def __init__(self, category: str | None = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.category = category or "unspecified"
