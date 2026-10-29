"""스모크 테스트 (smoke test) — 백엔드 서버를 띄운 상태에서 실행한다.

  (터미널 1) uvicorn app.main:app --port 8000
  (터미널 2) python scripts/smoke_test.py [--base-url http://localhost:8000] [--grounding kb] [--profile default]

순서: health 확인 → Mock 모드 채팅 2턴(후속 질문 포함) → 잘못된 요청의 오류 처리 확인.
표준 라이브러리(urllib)만 사용한다.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from typing import Any


def _configure_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        except (AttributeError, ValueError):
            pass


def request(base: str, method: str, path: str, body: Any = None, raw: bytes | None = None) -> tuple[int, Any]:
    data = raw if raw is not None else (json.dumps(body).encode("utf-8") if body is not None else None)
    req = urllib.request.Request(base + path, data=data, method=method)
    if data is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as err:
        text = err.read().decode("utf-8")
        try:
            return err.code, json.loads(text)
        except json.JSONDecodeError:
            return err.code, text


class Checker:
    def __init__(self) -> None:
        self.failures = 0

    def check(self, cond: bool, label: str, extra: str = "") -> None:
        mark = "PASS" if cond else "FAIL"
        if not cond:
            self.failures += 1
        print(f"[{mark}] {label}" + (f" — {extra}" if extra else ""))


def main() -> int:
    _configure_stdio()
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--grounding", default=None, choices=[None, "general", "kb"])
    parser.add_argument("--profile", default=None)
    parser.add_argument("--allow-real", action="store_true", help="MOCK_MODE=false 서버에서도 실행 (실제 API 비용 발생)")
    args = parser.parse_args()
    base = args.base_url.rstrip("/")
    c = Checker()

    # 1) health
    try:
        status, body = request(base, "GET", "/api/health")
    except urllib.error.URLError as err:
        print(f"[FAIL] 서버에 연결할 수 없습니다: {err}. 먼저 uvicorn app.main:app --port 8000 을 실행하세요.")
        return 1
    c.check(status == 200 and body.get("status") == "ok", "GET /api/health", json.dumps(body, ensure_ascii=False))
    if not body.get("mock_mode") and not args.allow_real:
        print("서버가 MOCK_MODE=false 입니다. 실제 API 호출 비용을 피하려고 중단합니다 (--allow-real 로 강제 실행).")
        return 1

    status, cfg = request(base, "GET", "/api/config")
    c.check(status == 200 and "guards" in cfg, "GET /api/config",
            f"mock={cfg.get('mock_mode')} grounding={cfg.get('grounding_mode')} profile={cfg.get('guards', {}).get('profile')}")

    # 2) 채팅 2턴 (후속 질문 포함)
    opts: dict[str, Any] = {}
    if args.grounding:
        opts["grounding_mode"] = args.grounding
    if args.profile:
        opts["guard_profile"] = args.profile
    q1 = "리스트와 튜플의 차이가 뭐야?"
    status, r1 = request(base, "POST", "/api/chat", {"messages": [{"role": "user", "content": q1}], **opts})
    c.check(status == 200 and bool(r1.get("answer")), "POST /api/chat 1턴",
            f"provider={r1.get('provider')} status={r1.get('status')} citations={len(r1.get('citations', []))}")
    if status == 200:
        print("   답변:", r1["answer"][:160].replace("\n", " "))
        messages = [
            {"role": "user", "content": q1},
            {"role": "assistant", "content": r1["answer"]},
            {"role": "user", "content": "그럼 튜플은 언제 써?"},
        ]
        status, r2 = request(base, "POST", "/api/chat", {"messages": messages, **opts})
        c.check(status == 200 and bool(r2.get("answer")), "POST /api/chat 2턴 (후속 질문)",
                f"rewrite_applied={r2.get('meta', {}).get('rewrite_applied')} rewritten={r2.get('meta', {}).get('rewritten_query')}")
        if status == 200:
            print("   답변:", r2["answer"][:160].replace("\n", " "))

    # 3) 잘못된 요청의 오류 처리
    status, body = request(base, "POST", "/api/chat", {"messages": []})
    c.check(status == 422 and "error" in body, "빈 messages → 422", json.dumps(body, ensure_ascii=False)[:120])
    status, body = request(base, "POST", "/api/chat", raw=b"{broken json")
    c.check(status == 422 and "error" in body, "깨진 JSON → 422")
    long_msg = "가" * (int(cfg.get("limits", {}).get("max_message_chars", 2000)) + 1)
    status, body = request(base, "POST", "/api/chat", {"messages": [{"role": "user", "content": long_msg}]})
    c.check(status == 400 and body.get("error", {}).get("code") == "message_too_long", "너무 긴 메시지 → 400")
    status, body = request(base, "POST", "/api/chat", {"messages": [{"role": "user", "content": "q"}], "guard_profile": "no_such"})
    c.check(status == 400 and "error" in body, "없는 가드 프로필 → 400")
    status, body = request(base, "GET", "/api/health")
    c.check(status == 200, "오류 이후에도 서버 정상 응답")

    print(f"\n결과: {'모두 통과' if c.failures == 0 else f'{c.failures}개 실패'}")
    return 0 if c.failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
