"""Metadata-only logging.

Fields: time, endpoint, status, latency, sizes, counts, stage names and error class names.
Request text, answers, model output and exception messages are never logged.
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
    # Library loggers can print request bodies at debug level.
    for name in ("openai", "httpx", "httpcore"):
        logging.getLogger(name).setLevel(logging.WARNING)
    # The access log would record client addresses and raw paths.
    access = logging.getLogger("uvicorn.access")
    access.disabled = True
    access.setLevel(logging.WARNING)


def _emit(fields: dict) -> None:
    _logger.info(" ".join(f"{k}={v}" for k, v in fields.items() if v is not None))


# Per-request dict in a ContextVar; nothing is kept between requests.
def start_request() -> dict:
    info: dict = {}
    _REQUEST.set(info)
    return info


def degrade(stage: str, exc: BaseException = None) -> None:
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
    _emit({"event": "startup", "kb_sections": kb_sections, "kb_chunks": kb_chunks, "latency_ms": latency_ms})


def index_line(*, kb_en_chunks: int, kb_chunks: int, latency_ms: int, error=None) -> None:
    _emit({"event": "index", "kb_en_chunks": kb_en_chunks, "kb_chunks": kb_chunks, "latency_ms": latency_ms,
           "error": error})


# Unhandled task exceptions: class name only.
def loop_exception(loop, context) -> None:
    exc = context.get("exception")
    _emit({"event": "asyncio", "error": type(exc).__name__ if exc is not None else "none"})


def llm_line(*, kind: str, status: str, latency_ms: int, attempt: int, hedge: int = 0, error=None) -> None:
    _emit({"event": "llm", "kind": kind, "status": status, "latency_ms": latency_ms,
           "attempt": attempt, "hedge": hedge or None, "error": error})
