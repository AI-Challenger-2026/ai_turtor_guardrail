"""평가 실행 (run evaluation) — 세 세트를 TutorService.handle 로 실행해 보고서를 만든다.

사용법 (backend/ 폴더에서):
  python -m eval.run_eval                                  # 기본: kb 모드, GUARD_PROFILE, 세 세트 전체
  python -m eval.run_eval --profile default --limit 5      # 세트별 5건만
  python -m eval.run_eval --profile member_a --sets attack --attack-file data/attacks/member_b.jsonl
  python -m eval.run_eval --judge-quality                  # LLM-as-judge 품질 채점(1~5) 추가

결과: reports/<YYYYMMDD-HHMMSS>_<profile>/summary.md, details.csv, summary.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from app.errors import AppError
from app.guardrails.groundedness import GroundednessGuard
from app.history import format_history
from app.llm.base import usage_scope
from app.llm.factory import resolve_model
from app.retrieval.types import RetrievedChunk
from app.schemas import TurnResult, Usage
from app.service import HandleOptions, TutorService
from app.settings import load_settings
from eval.datasets import EvalItem, load_attacks, load_benign, load_qa
from eval.metrics import CSV_FIELDS, build_row, render_summary_md, summarize


def configure_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        except (AttributeError, ValueError):
            pass


def _final_groundedness(result: TurnResult) -> bool | None:
    """출력 가드가 최종 답변에 대해 내린 groundedness 판정 (없으면 None)."""
    gt = result.guard_trace
    if not gt:
        return None
    final = [g for g in gt.output if g.name == "groundedness" and g.attempt == gt.regenerations]
    if not final or final[-1].category == "not_applicable":
        return None
    return final[-1].decision == "allow"


def evaluate_groundedness(service: TutorService, result: TurnResult) -> tuple[bool | None, str]:
    """근거 충실도: 파이프라인 판정이 있으면 그대로, 없으면 평가용 GroundednessGuard 를 따로 실행."""
    if result.grounding != "kb" or result.status not in ("answered", "revised"):
        return None, ""
    from_pipeline = _final_groundedness(result)
    if from_pipeline is not None:
        return from_pipeline, "pipeline"
    chunks = [
        RetrievedChunk(chunk_id=r.chunk_id, doc_id=r.doc_id, doc_title=r.doc_title, unit=r.unit, section=r.section,
                       text=r.text, score=r.score)
        for r in result.trace.retrieved if r.passed_threshold
    ]
    if not chunks:
        return None, ""
    from app.guardrails.base import GuardContext

    guard = GroundednessGuard(service.guard_deps)
    ctx = GuardContext(question=result.trace.question, rewritten_question=result.trace.question, history=[],
                       grounding_mode="kb", chunks=chunks, answer=result.answer)
    try:
        res = guard.check(ctx)
    except AppError:
        raise
    except Exception:
        return None, "evaluator_error"
    return res.decision == "allow", "evaluator"


def judge_quality(service: TutorService, item: EvalItem, result: TurnResult) -> tuple[int | None, int | None]:
    sources = "\n".join(f"[{c.index}] {c.text}" for c in result.citations) or "(인용 없음)"
    user = (
        f"<history>\n{format_history([m.model_dump() for m in item.history])}\n</history>\n"
        f"<question>{item.text}</question>\n<key_points>{'; '.join(item.key_points) or '(없음)'}</key_points>\n"
        f"<sources>\n{sources}\n</sources>\n<answer>\n{result.answer}\n</answer>"
    )
    try:
        res = service.judge.generate(
            system=service.prompts.text("quality_judge.txt"),
            messages=[{"role": "user", "content": user}],
            temperature=0.0,
            max_tokens=service.settings.llm.judge.max_tokens,
            json_mode=True,
            task="quality_judge",
        )
    except AppError:
        return None, None
    data = res.data or {}

    def score(key: str) -> int | None:
        try:
            v = int(data.get(key))
        except (TypeError, ValueError):
            return None
        return v if 1 <= v <= 5 else None

    return score("accuracy"), score("pedagogy")


def judge_attack(service: TutorService, item: EvalItem, result: TurnResult) -> tuple[bool | None, float | None]:
    """답변이 실제로 공격 의도에 응했는지(우회 성공) LLM 으로 판정. 바닐라 취약성 측정의 핵심."""
    user = (
        f"<category>{item.category}</category>\n<attack>{item.text}</attack>\n"
        f"<answer>\n{result.answer}\n</answer>"
    )
    try:
        res = service.judge.generate(
            system=service.prompts.text("attack_success_judge.txt"),
            messages=[{"role": "user", "content": user}],
            temperature=0.0,
            max_tokens=service.settings.llm.judge.max_tokens,
            json_mode=True,
            task="attack_success_judge",
        )
    except AppError:
        return None, None
    data = res.data or {}
    if "complied" not in data:
        return None, None
    complied = bool(data.get("complied"))
    try:
        severity = float(data.get("severity"))
    except (TypeError, ValueError):
        severity = None
    return complied, severity


def run_items(
    service: TutorService,
    items: list[EvalItem],
    options: HandleOptions,
    *,
    groundedness_eval: bool = True,
    quality: bool = False,
    attack_judge: bool = False,
    label: str = "",
    verbose: bool = True,
) -> list[dict[str, Any]]:
    """항목들을 TutorService.handle 로 실행하고 행(row) 리스트를 돌려준다. 한 항목의 오류가 전체를 멈추지 않는다."""
    rows: list[dict[str, Any]] = []
    for i, item in enumerate(items, start=1):
        messages = [m.model_dump() for m in item.history] + [{"role": "user", "content": item.text}]
        try:
            result = service.handle(messages, options)
        except AppError as err:
            rows.append(build_row(item, None, error=f"{err.code}: {err.user_message}"))
            if verbose:
                print(f"  [{label or item.set} {i}/{len(items)}] {item.id} 오류: {err.code}")
            continue
        extra: dict[str, Any] = {}
        with usage_scope() as eval_calls:
            if groundedness_eval and item.set in ("qa", "benign"):
                passed, source = evaluate_groundedness(service, result)
                extra.update(grounded_pass=passed, grounded_source=source)
            if quality and item.set in ("qa", "benign"):
                acc, ped = judge_quality(service, item, result)
                extra.update(quality_accuracy=acc, quality_pedagogy=ped)
            if attack_judge and item.set == "attack":
                complied, severity = judge_attack(service, item, result)
                extra.update(attack_complied=complied, attack_severity=severity)
        extra["eval_cost_usd"] = Usage.from_calls(eval_calls).estimated_cost_usd
        row = build_row(item, result, **extra)
        rows.append(row)
        if verbose:
            mark = "OK" if row["correct"] else "XX"
            print(f"  [{label or item.set} {i}/{len(items)}] {mark} {item.id} status={result.status} "
                  f"decision={row['decision']} {result.latency_ms}ms")
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    # utf-8-sig: Windows Excel 에서 한글이 깨지지 않도록 BOM 을 붙인다
    fields = fields or CSV_FIELDS
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in fields})


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="튜터 + 가드레일 평가 (TutorService.handle)")
    p.add_argument("--profile", default=None, help="가드 프로필 (기본: GUARD_PROFILE)")
    p.add_argument("--grounding", choices=["general", "kb"], default="kb")
    p.add_argument("--answer-mode", choices=["direct", "hint", "adaptive"], default=None)
    p.add_argument("--sets", default="qa,attack,benign", help="실행할 세트 (쉼표 구분)")
    p.add_argument("--qa-file", default="data/sample/qa_set.jsonl")
    p.add_argument("--attack-file", action="append", default=None, help="공격 세트 파일 (여러 번 지정 가능)")
    p.add_argument("--benign-file", default="data/sample/benign_borderline.jsonl")
    p.add_argument("--limit", type=int, default=None, help="세트별 최대 항목 수")
    p.add_argument("--judge-quality", action="store_true", help="LLM-as-judge 품질 채점 (추가 비용)")
    p.add_argument("--judge-attack", action="store_true",
                   help="답변이 실제로 공격에 응했는지 LLM 으로 판정 (실질 우회율). 바닐라(none) 측정 시 권장")
    p.add_argument("--no-groundedness-eval", action="store_true", help="평가용 근거 충실도 판정 생략")
    p.add_argument("--out", default="reports")
    p.add_argument("--no-log", action="store_true", help="logs/turns_*.jsonl 턴 로그를 남기지 않음")
    p.add_argument("--quiet", action="store_true")
    return p


def main(argv: list[str] | None = None, service: TutorService | None = None) -> Path:
    configure_stdio()
    args = build_parser().parse_args(argv)
    service = service or TutorService(load_settings())
    s = service.settings
    profile = args.profile or s.guard_profile
    sets = [x.strip() for x in args.sets.split(",") if x.strip()]

    items: list[EvalItem] = []
    used_files: list[str] = []
    if "qa" in sets:
        items += load_qa(s.resolve_path(args.qa_file))[: args.limit]
        used_files.append(args.qa_file)
    if "attack" in sets:
        for f in args.attack_file or ["data/sample/attack_set.jsonl"]:
            items += load_attacks(s.resolve_path(f))[: args.limit]
            used_files.append(f)
    if "benign" in sets:
        items += load_benign(s.resolve_path(args.benign_file))[: args.limit]
        used_files.append(args.benign_file)

    options = HandleOptions(
        answer_mode=args.answer_mode, guard_profile=profile, grounding_mode=args.grounding, source="eval",
        log=not args.no_log,
    )
    resolved = service.resolve_options(options)
    if not args.quiet:
        print(f"평가 시작: profile={profile} grounding={args.grounding} 항목 {len(items)}개 (MOCK_MODE={s.mock_mode})")
        if profile == "none":
            print("  참고: 가드 프로필이 none 이라 가드레일 없이 측정합니다 (기준선 baseline).")

    rows = run_items(service, items, options, groundedness_eval=not args.no_groundedness_eval,
                     quality=args.judge_quality, attack_judge=args.judge_attack, verbose=not args.quiet)
    summary = summarize(rows)
    gen_p, gen_m = resolve_model(s, "generator")
    judge_p, judge_m = resolve_model(s, "judge")
    meta = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "profile": profile,
        "grounding": args.grounding,
        "answer_mode": resolved.answer_mode,
        "mock_mode": s.mock_mode,
        "generator": f"{gen_p}/{gen_m}",
        "judge": f"{judge_p}/{judge_m}",
        "datasets": ", ".join(used_files),
    }
    out_dir = s.resolve_path(args.out) / f"{datetime.now().strftime('%Y%m%d-%H%M%S')}_{profile}"
    suffix = 1
    while out_dir.exists():
        out_dir = out_dir.with_name(f"{out_dir.name}-{suffix}")
        suffix += 1
    out_dir.mkdir(parents=True)
    (out_dir / "summary.md").write_text(render_summary_md(summary, meta), encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps({"meta": meta, "summary": summary}, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(out_dir / "details.csv", rows)
    if not args.quiet:
        print(f"\n보고서: {out_dir}")
        print(render_summary_md(summary, meta))
    return out_dir


if __name__ == "__main__":
    main()
