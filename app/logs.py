"""Metadata-only logging.

Allowed fields: time, endpoint, status code, latency, input length, error kind.
Never log request/answer text, LLM output, verdicts, or exception messages
(a pydantic ValidationError message embeds the input). Exceptions are logged by class name.

Each request gets its own metadata dict in a ContextVar (set by the route). Tasks created inside
the request copy the context, so a stage that falls back anywhere in the request can mark it.
The dict lives only for that request; nothing is carried between requests.
"""
import contextvars
import logging
import sys

_logger = logging.getLogger("app")
_REQUEST = contextvars.ContextVar("request_info", default=None)


def setup() -> None:
    if not _logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
        _logger.addHandler(handler)
    _logger.setLevel(logging.INFO)
    _logger.propagate = False
    # Third-party loggers can dump request bodies at debug level (e.g. OPENAI_LOG=debug).
    for name in ("openai", "httpx", "httpcore"):
        logging.getLogger(name).setLevel(logging.WARNING)
    # uvicorn's access log writes the client address and the raw request path for every request,
    # which our allowlist does not include; request_line already records endpoint, status and
    # latency. uvicorn configures logging before it imports the app, so this stays in effect.
    access = logging.getLogger("uvicorn.access")
    access.disabled = True
    access.setLevel(logging.WARNING)


def _emit(fields: dict) -> None:
    _logger.info(" ".join(f"{k}={v}" for k, v in fields.items() if v is not None))


def start_request() -> dict:
    info: dict = {}
    _REQUEST.set(info)
    return info


def degrade(stage: str, exc: BaseException = None) -> None:
    """Mark that a stage failed and fell back. Stage names only (e.g. "classify"); never a verdict."""
    info = _REQUEST.get()
    if info is None:
        return
    info["degraded"] = f"{info['degraded']},{stage}" if info.get("degraded") else stage
    if exc is not None and "error" not in info:
        info["error"] = type(exc).__name__


def set_outcome(outcome: str) -> None:
    info = _REQUEST.get()
    if info is not None:
        info["outcome"] = outcome


def request_line(*, endpoint: str, status, latency_ms: int, body_bytes: int,
                 input_chars=None, n_msgs=None, outcome=None, degraded=None, error=None) -> None:
    _emit({"event": "request", "endpoint": endpoint, "status": status, "latency_ms": latency_ms,
           "body_bytes": body_bytes, "input_chars": input_chars, "n_msgs": n_msgs,
           "outcome": outcome, "degraded": degraded, "error": error})


def startup_line(*, kb_sections: int, kb_chunks: int, latency_ms: int) -> None:
    # Index sizes only; never any knowledge-base text.
    _emit({"event": "startup", "kb_sections": kb_sections, "kb_chunks": kb_chunks, "latency_ms": latency_ms})


def index_line(*, kb_en_chunks: int, kb_chunks: int, latency_ms: int, error=None) -> None:
    # English index built at start-up: how many chunks got a translation; never any text.
    _emit({"event": "index", "kb_en_chunks": kb_en_chunks, "kb_chunks": kb_chunks, "latency_ms": latency_ms,
           "error": error})


def loop_exception(loop, context) -> None:
    """Event-loop exception handler: class name only. asyncio's default handler prints the message
    and a traceback, and an exception message can carry API error text."""
    exc = context.get("exception")
    _emit({"event": "asyncio", "error": type(exc).__name__ if exc is not None else "none"})


def llm_line(*, kind: str, status: str, latency_ms: int, attempt: int, hedge: int = 0, error=None) -> None:
    # One line per API call. attempt = retry round (1, 2); hedge=1 marks the duplicate sent when
    # the first call of a round is slow.
    _emit({"event": "llm", "kind": kind, "status": status, "latency_ms": latency_ms,
           "attempt": attempt, "hedge": hedge or None, "error": error})
