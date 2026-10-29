"""교차 공격 평가 (cross attack) — 팀원 4명의 가드 프로필 × 팀원 4명의 공격 세트.

사용법 (backend/ 폴더에서):
  python -m eval.cross_attack                         # member_a~d 프로필 × data/attacks/member_*.jsonl
  python -m eval.cross_attack --include-sample        # 샘플 공격 세트도 열로 추가
  python -m eval.cross_attack --profiles default,member_a --limit 10

출력: 차단율 행렬(행 = 방어 프로필, 열 = 공격 작성자)과 각 프로필의 오차단율(정상·경계 질문 block 비율)
      reports/<시각>_cross_attack/summary.md, matrix.csv, details.csv
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
from typing import Any

from app.service import HandleOptions, TutorService
from app.settings import load_settings
from eval.datasets import EvalItem, load_attacks, load_benign, load_qa
from eval.metrics import CSV_FIELDS, pct
from eval.run_eval import configure_stdio, run_items, write_csv


def _cell(blocked: int, n: int) -> str:
    return "-" if n == 0 else f"{blocked / n * 100:.0f}% ({blocked}/{n})"


def main(argv: list[str] | None = None, service: TutorService | None = None) -> Path:
    configure_stdio()
    p = argparse.ArgumentParser(description="교차 공격 평가")
    p.add_argument("--profiles", default="member_a,member_b,member_c,member_d")
    p.add_argument("--attack-dir", default="data/attacks")
    p.add_argument("--include-sample", action="store_true", help="data/sample/attack_set.jsonl 을 'sample' 열로 추가")
    p.add_argument("--benign-files", default="data/sample/benign_borderline.jsonl,data/sample/qa_set.jsonl",
                   help="오차단율 측정용 정상 질문 파일 (쉼표 구분)")
    p.add_argument("--grounding", choices=["general", "kb"], default="kb")
    p.add_argument("--limit", type=int, default=None, help="세트별 최대 항목 수")
    p.add_argument("--out", default="reports")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)

    service = service or TutorService(load_settings())
    s = service.settings
    profiles = [x.strip() for x in args.profiles.split(",") if x.strip()]
    unknown = [x for x in profiles if x not in service.available_profiles()]
    if unknown:
        raise SystemExit(f"알 수 없는 프로필: {unknown} (사용 가능: {service.available_profiles()})")

    attack_sets: dict[str, list[EvalItem]] = {}
    for f in sorted(s.resolve_path(args.attack_dir).glob("*.jsonl")):
        attack_sets[f.stem] = load_attacks(f)[: args.limit]
    if args.include_sample:
        attack_sets["sample"] = load_attacks(s.resolve_path("data/sample/attack_set.jsonl"))[: args.limit]
    benign: list[EvalItem] = []
    for f in [x.strip() for x in args.benign_files.split(",") if x.strip()]:
        path = s.resolve_path(f)
        benign += (load_qa(path) if "qa_set" in path.name else load_benign(path))[: args.limit]

    if not args.quiet:
        sizes = ", ".join(f"{k}={len(v)}" for k, v in attack_sets.items())
        print(f"교차 공격: 프로필 {profiles} × 공격 세트 [{sizes}], 정상 {len(benign)}건 (MOCK_MODE={s.mock_mode})")
        if all(len(v) == 0 for v in attack_sets.values()):
            print("  참고: 공격 세트가 모두 비어 있습니다. data/attacks/member_*.jsonl 을 채우거나 --include-sample 을 쓰세요.")

    matrix: dict[str, dict[str, tuple[int, int]]] = {}
    over_block: dict[str, tuple[int, int]] = {}
    all_rows: list[dict[str, Any]] = []
    for prof in profiles:
        opts = HandleOptions(guard_profile=prof, grounding_mode=args.grounding, source="eval")
        matrix[prof] = {}
        for author, items in attack_sets.items():
            rows = run_items(service, items, opts, groundedness_eval=False, label=f"{prof}<-{author}", verbose=not args.quiet)
            valid = [r for r in rows if r["decision"] != "error"]
            matrix[prof][author] = (sum(1 for r in valid if r["decision"] == "block"), len(valid))
            all_rows += [{**r, "profile": prof} for r in rows]
        rows = run_items(service, benign, opts, groundedness_eval=False, label=f"{prof} benign", verbose=not args.quiet)
        valid = [r for r in rows if r["decision"] != "error"]
        over_block[prof] = (sum(1 for r in valid if r["decision"] == "block"), len(valid))
        all_rows += [{**r, "profile": prof} for r in rows]

    authors = list(attack_sets)
    lines = ["# 교차 공격 결과 (cross attack)", ""]
    if s.mock_mode:
        lines += ["> **MOCK_MODE 결과** — 실제 LLM 판정이 아니므로 파이프라인 동작 확인용입니다.", ""]
    lines += [
        f"- 실행 시각: {datetime.now().isoformat(timespec='seconds')}, 근거 모드: {args.grounding}",
        "- 셀 = 차단율 (차단/전체). 행 = 방어 프로필, 열 = 공격 작성자. `*` = 자기 공격 세트(대각선).", "",
        "| 방어 \\ 공격 | " + " | ".join(authors) + " | 오차단율 |",
        "| --- | " + " | ".join("---" for _ in authors) + " | --- |",
    ]
    for prof in profiles:
        cells = []
        for a in authors:
            blocked, n = matrix[prof][a]
            cells.append(_cell(blocked, n) + ("*" if a == prof and n else ""))
        ob_b, ob_n = over_block[prof]
        lines.append(f"| {prof} | " + " | ".join(cells) + f" | {pct(None if ob_n == 0 else ob_b / ob_n)} ({ob_b}/{ob_n}) |")
    table = "\n".join(lines) + "\n"

    out_dir = s.resolve_path(args.out) / f"{datetime.now().strftime('%Y%m%d-%H%M%S')}_cross_attack"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.md").write_text(table, encoding="utf-8")
    with (out_dir / "matrix.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["defense_profile"] + [f"block_rate_vs_{a}" for a in authors] + ["over_block_rate"])
        for prof in profiles:
            row = [prof]
            for a in authors:
                blocked, n = matrix[prof][a]
                row.append("" if n == 0 else round(blocked / n, 4))
            ob_b, ob_n = over_block[prof]
            row.append("" if ob_n == 0 else round(ob_b / ob_n, 4))
            w.writerow(row)
    write_csv(out_dir / "details.csv", all_rows, fields=["profile"] + CSV_FIELDS)
    if not args.quiet:
        print(f"\n보고서: {out_dir}\n")
        print(table)
    return out_dir


if __name__ == "__main__":
    main()
