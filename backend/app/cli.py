"""터미널 대화 CLI (command-line chat) — TutorService.handle() 을 그대로 호출한다.

사용법 (usage, backend/ 폴더에서):
  python -m app.cli                      # 대화형 (interactive)
  python -m app.cli --debug              # 턴마다 중간 결과 출력
  python -m app.cli --grounding kb --profile default --mode hint
  python -m app.cli --once "리스트가 뭐야?"  # 한 번만 묻고 종료

대화 중 명령: /new (새 대화), /mode <direct|hint|adaptive>, /grounding <general|kb>, /profile <이름>, /exit
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from app.errors import AppError
from app.schemas import TurnResult
from app.service import HandleOptions, TutorService
from app.settings import load_settings


def _configure_stdio() -> None:
    # 콘솔이 아닌 파이프/파일 출력에서는 Windows 기본 cp949 로 한글·특수문자가 깨지거나 예외가 나므로 UTF-8 로 바꾼다.
    for stream in (sys.stdout, sys.stderr):
        try:
            if not stream.isatty():
                stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
            else:
                stream.reconfigure(errors="replace")  # type: ignore[attr-defined]
        except (AttributeError, ValueError):
            pass


def _print_debug(result: TurnResult) -> None:
    t = result.trace
    print("\n----- [debug] 턴 추적 (turn trace) -----")
    print(f"trace_id={t.trace_id} status={t.status} latency={result.latency_ms}ms")
    print(f"모드: answer_mode={t.answer_mode} grounding={t.grounding_mode} guard_profile={t.guard_profile}")
    print(f"질문 유형={t.question_type} 학습자 수준={t.learner_level} 이력 메시지={t.history_messages} (잘림 {t.truncated_messages})")
    if t.rewrite:
        print(f"질의 재작성: applied={t.rewrite.applied} reason={t.rewrite.reason}")
        print(f"  원문: {t.rewrite.original}")
        print(f"  재작성: {t.rewrite.rewritten}")
    if t.retrieved or t.retrieval_query:
        print(f"검색 질의: {t.retrieval_query} (threshold={t.retrieval_threshold})")
        for r in t.retrieved:
            mark = "O" if r.passed_threshold else "x"
            print(f"  [{mark}] #{r.rank} {r.score:.3f} {r.chunk_id} | {r.unit} > {r.section}")
    gt = result.guard_trace
    if gt:
        print(f"가드 프로필: {gt.profile} final_action={gt.final_action} regenerations={gt.regenerations}")
        for g in gt.input + gt.output:
            print(f"  [{g.stage}#{g.attempt}] {g.name}: {g.decision} ({g.category}, score={g.score:.2f}, {g.latency_ms}ms) {g.reason}")
    else:
        print("가드레일: 미적용 (guard_profile=none)")
    for i, raw in enumerate(t.raw_answers):
        print(f"원본 답변 #{i}: {raw[:300]}{'…' if len(raw) > 300 else ''}")
    u = result.usage
    print(f"LLM 호출 {u.llm_calls}회, 토큰 in={u.input_tokens} out={u.output_tokens}, 예상 비용 ${u.estimated_cost_usd:.5f}")
    for c in u.calls:
        print(f"  - {c.task} [{c.role}] {c.provider}/{c.model} {c.latency_ms}ms ok={c.ok} temp_applied={c.temperature_applied}")
    for note in t.notes:
        print(f"  note: {note}")
    print("---------------------------------------\n")


def _print_result(result: TurnResult) -> None:
    tag = {"blocked": "[차단됨] ", "fallback": "[안전 대체 답변] ", "no_evidence": "[자료에서 근거를 찾지 못함] ", "revised": "[재생성됨] "}.get(result.status, "")
    print(f"\n튜터 ({result.provider}/{result.model})> {tag}{result.answer}")
    for c in result.citations:
        print(f"  [{c.index}] {c.doc_title} > {c.unit} > {c.section} ({c.chunk_id})")
    print()


def run(argv: list[str] | None = None) -> int:
    _configure_stdio()
    parser = argparse.ArgumentParser(description="AI 튜터 CLI (TutorService.handle)")
    parser.add_argument("--debug", action="store_true", help="턴마다 중간 결과 출력")
    parser.add_argument("--mode", choices=["direct", "hint", "adaptive"], default=None, help="응답 모드")
    parser.add_argument("--grounding", choices=["general", "kb"], default=None, help="근거 모드")
    parser.add_argument("--profile", default=None, help="가드 프로필 (none, default, member_a ...)")
    parser.add_argument("--once", default=None, help="질문 하나만 처리하고 종료")
    parser.add_argument("--json", action="store_true", help="--once 결과를 JSON 으로 출력")
    args = parser.parse_args(argv)

    service = TutorService(load_settings())
    opts: dict[str, Any] = {"answer_mode": args.mode, "grounding_mode": args.grounding, "guard_profile": args.profile}
    history: list[dict[str, str]] = []

    def ask(question: str) -> TurnResult | None:
        messages = history + [{"role": "user", "content": question}]
        try:
            result = service.handle(messages, HandleOptions(**opts, source="cli"))
        except AppError as err:
            print(f"\n[오류] {err.user_message} (code={err.code})\n")
            return None
        return result

    if args.once:
        result = ask(args.once)
        if result is None:
            return 1
        if args.json:
            print(json.dumps(result.model_dump(), ensure_ascii=False, indent=2, default=str))
        else:
            _print_result(result)
            if args.debug:
                _print_debug(result)
        return 0

    s = service.settings
    print("AI 튜터 CLI — /new 새 대화, /mode, /grounding, /profile, /exit")
    print(f"MOCK_MODE={s.mock_mode} generator={service.generator.name}/{service.generator.model} "
          f"grounding={args.grounding or s.grounding_mode} profile={args.profile or s.guard_profile}\n")
    while True:
        try:
            line = input("나> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line.startswith("/"):
            cmd, _, value = line.partition(" ")
            value = value.strip()
            if cmd in ("/exit", "/quit"):
                break
            if cmd == "/new":
                history.clear()
                print("(새 대화를 시작합니다. 이전 문맥이 초기화되었습니다.)\n")
            elif cmd == "/mode" and value:
                opts["answer_mode"] = value
                print(f"(응답 모드: {value})\n")
            elif cmd == "/grounding" and value:
                opts["grounding_mode"] = value
                print(f"(근거 모드: {value})\n")
            elif cmd == "/profile" and value:
                opts["guard_profile"] = value
                print(f"(가드 프로필: {value})\n")
            else:
                print("(알 수 없는 명령입니다)\n")
            continue
        result = ask(line)
        if result is None:
            continue
        _print_result(result)
        if args.debug:
            _print_debug(result)
        history.append({"role": "user", "content": line})
        history.append({"role": "assistant", "content": result.answer})
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
