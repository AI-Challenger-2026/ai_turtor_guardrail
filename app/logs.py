"""Metadata-only logging.

Allowed fields: time, endpoint, status code, latency, input length, error kind.
Never log request/answer text, LLM output, verdict reasons, or exception messages
(a pydantic ValidationError message embeds the input). Exceptions are logged by class name.
"""
import logging
import sys

_logger = logging.getLogger("app")
_KNOWN_ENDPOINTS = ("/tutor", "/guard")


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


def endpoint_label(path: str) -> str:
    # The path comes from the client; only log it if it is one of ours.
    return path if path in _KNOWN_ENDPOINTS else "other"


def _emit(fields: dict) -> None:
    _logger.info(" ".join(f"{k}={v}" for k, v in fields.items() if v is not None))


def request_line(*, endpoint: str, status, latency_ms: int, body_bytes: int,
                 input_chars=None, n_msgs=None, outcome=None, degraded=None, error=None) -> None:
    # `degraded` names a stage that failed and fell back (e.g. "classify"); never a verdict.
    _emit({"event": "request", "endpoint": endpoint, "status": status, "latency_ms": latency_ms,
           "body_bytes": body_bytes, "input_chars": input_chars, "n_msgs": n_msgs,
           "outcome": outcome, "degraded": degraded, "error": error})


def startup_line(*, kb_sections: int, kb_chunks: int, embeddings: str, latency_ms: int) -> None:
    # Index sizes and whether embeddings are on; never any knowledge-base text.
    _emit({"event": "startup", "kb_sections": kb_sections, "kb_chunks": kb_chunks,
           "embeddings": embeddings, "latency_ms": latency_ms})


def llm_line(*, kind: str, status: str, latency_ms: int, attempt: int, error=None) -> None:
    _emit({"event": "llm", "kind": kind, "status": status, "latency_ms": latency_ms,
           "attempt": attempt, "error": error})
