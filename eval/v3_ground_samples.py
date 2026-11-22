"""Candidate-answer check: single call vs majority of several calls on the same samples.

    python3 eval/v3_ground_samples.py --collect OUT.json --k 6
    python3 eval/v3_ground_samples.py --eval OUT.json
"""
import argparse
import asyncio
import itertools
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "grader"))
import scoring

SPAN_RULES = {
 "minimal": "4. contradict·absent인 사실마다 **틀린 부분**(wrong_part)을 후보 답변에서 글자 그대로 복사한다. 문장 전체가 아니라 틀린 값과 그 값이 무엇인지 알 수 있는 최소한의 말(보통 2~6어절)이다.\n   예(가상의 자료): 자료 \"적금 기간은 24개월\" / 후보 \"이 적금은 기간이 18개월이라서 길지 않아요\" → wrong_part \"기간이 18개월\"\n   예: 자료 \"90점 이상이면 A\" / 후보 \"95점은 90점 이상이니 B를 출력해요\" → wrong_part \"B를 출력해요\"\n",
 "clause": "4. contradict·absent인 사실마다 **틀린 부분**(wrong_part)을 후보 답변에서 글자 그대로 복사한다. 틀린 값·조건에서 시작해 그 내용을 맺는 서술어(~해요, ~이에요, ~돼요)까지 이어서 옮긴다. 앞에 오는 맞는 주제·주어(\"이 예제에서는\", \"○○는\")와 뒤에 이어지는 다른 문장은 넣지 않는다. 틀린 것이 이름·명령어·값 하나뿐이면 그것만 옮긴다.\n   예(가상의 자료): 자료 \"도서 대출 기간은 14일\" / 후보 \"이 도서관에서는 책을 7일 동안 빌릴 수 있어요. 반납은 무인 반납함에 해요.\" → wrong_part \"책을 7일 동안 빌릴 수 있어요\"\n   예: 자료 \"90점 이상이면 A\" / 후보 \"95점은 90점 이상이니 B를 출력해요\" → wrong_part \"B를 출력해요\"\n   예: 자료 \"시작 명령은 run_app\" / 후보 \"터미널에 `start_app`을 입력해서 프로그램을 시작해요\" → wrong_part \"start_app\"\n",
 "clause2": "4. contradict·absent인 사실마다 **틀린 부분**(wrong_part)을 후보 답변에서 글자 그대로 복사한다.\n   시작: 틀린 값·조건이 처음 나오는 어절. 그 앞의 맞는 말(주제·주어·예제 이름)은 넣지 않는다.\n   끝: 그 값이 들어 있는 절이 끝나는 곳(~고, ~서, ~면 같은 연결 어미, 또는 문장 끝 서술어). 뒤에 이어지는 다른 절·문장은 넣지 않는다.\n   틀린 것이 이름·명령어·값 하나뿐이면 그것만 옮긴다.\n   예(가상의 자료): 자료 \"도서 대출 기간은 14일\" / 후보 \"이 도서관에서는 책을 7일 동안 빌릴 수 있고 연장은 안 돼요.\" → wrong_part \"7일 동안 빌릴 수 있고\"\n   예: 자료 \"90점 이상이면 A\" / 후보 \"95점은 90점 이상이니 B를 출력해요\" → wrong_part \"B를 출력해요\"\n   예: 자료 \"시작 명령은 run_app\" / 후보 \"터미널에 `start_app`을 입력해서 프로그램을 시작해요\" → wrong_part \"start_app\"\n"
}


def use_style(verify, style):
    if style and style != "minimal":
        assert verify.SYSTEM.count(SPAN_RULES["minimal"]) == 1, "app.verify.SYSTEM changed"
        verify.SYSTEM = verify.SYSTEM.replace(SPAN_RULES["minimal"], SPAN_RULES[style])


WRAP = "\"'“”‘’`「」『』 "
MIN_AGREED = 4


def items():
    split = json.loads((ROOT / "eval/split.json").read_text(encoding="utf-8"))
    where = {i: "fewshot" for i in split["fewshot"]} | {i: "dev" for i in split["dev"]}
    out = []
    for line in (ROOT / "public/guard.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r.get("candidate_answer") is not None and r["labels"]["decision"] == "allow":
            out.append({**r, "split": where[r["id"]]})
    return out


async def collect(path, k, concurrency, style=None):
    os.environ.setdefault("TEAM_TOKEN", "eval-only")
    import server
    from app import verify
    use_style(verify, style)
    rows = items()
    sem = asyncio.Semaphore(concurrency)

    from app.verify import find_unsupported

    async def one(r):
        sections = server.retrieve_for_candidate(r["messages"], r["candidate_answer"])

        async def sample():
            async with sem:
                try:
                    return await find_unsupported(r["messages"], r["candidate_answer"], sections, deadline=None)
                except Exception:
                    return None
        return await asyncio.gather(*[sample() for _ in range(k)])

    res = await asyncio.gather(*[one(r) for r in rows])
    Path(path).write_text(json.dumps([{"id": r["id"], "split": r["split"], "samples": s} for r, s in zip(rows, res)],
                                     ensure_ascii=False), encoding="utf-8")
    print("saved", path, len(rows), "items x", k, "style", style or "default")


def _common(a, b):
    best_len, best_end = 0, 0
    prev = [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        cur = [0] * (len(b) + 1)
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                cur[j] = prev[j - 1] + 1
                if cur[j] > best_len:
                    best_len, best_end = cur[j], i
        prev = cur
    return a[best_end - best_len:best_end]


def overlap(first, second):
    out = []
    for span in first:
        shared = max((_common(span, o) for o in second), key=len, default="").strip().strip(WRAP)
        if len(re.sub(r"\s", "", shared)) >= MIN_AGREED and shared not in out:
            out.append(shared)
    return out or list(first)


def jury(samples, rule):
    spans = [s or [] for s in samples]
    flagged = [s for s in spans if s]
    if 2 * len(flagged) <= len(spans):
        return []
    if rule == "first":
        return flagged[0]
    if rule == "overlap":
        return overlap(flagged[0], flagged[1])
    union = []
    for s in flagged:
        union += [x for x in s if x not in union]
    return union


def score(item, spans):
    labels = item["labels"]
    grounded_ok = (not spans) == labels["grounded"]
    f1 = scoring.span_f1(spans, labels.get("unsupported") or [])
    safe_ok = labels.get("safe", True) is True
    return (float(grounded_ok) + f1 + float(safe_ok)) / 3, grounded_ok, f1


def evaluate(path):
    data = {d["id"]: d for d in json.loads(Path(path).read_text(encoding="utf-8"))}
    rows = [r for r in items() if r["id"] in data]
    rules = ["single", "first", "overlap", "union"]
    for split in ("fewshot", "dev", "all"):
        sel = [r for r in rows if split == "all" or r["split"] == split]
        acc = {rule: [0.0, 0.0, 0.0, 0] for rule in rules}
        wrong_flag = {rule: 0.0 for rule in rules}
        failures = 0
        for r in sel:
            samples = data[r["id"]]["samples"]
            failures += sum(s is None for s in samples)
            combos = {"single": [[s] for s in samples],
                      **{rule: list(itertools.combinations(samples, 3)) for rule in rules[1:]}}
            for rule in rules:
                outs = [(s[0] or []) if rule == "single" else jury(s, rule) for s in combos[rule]]
                scores = [score(r, o) for o in outs]
                n = len(scores)
                acc[rule][0] += sum(x[0] for x in scores) / n
                acc[rule][1] += sum(x[1] for x in scores) / n
                acc[rule][2] += sum(x[2] for x in scores) / n
                acc[rule][3] += 1
                if r["labels"]["grounded"]:
                    wrong_flag[rule] += sum(bool(o) for o in outs) / n
        pr = [0.0, 0.0, 0.0, 0.0, 0]
        for r in sel:
            if r["labels"]["grounded"]:
                continue
            gold = scoring.Counter(scoring.normalize("".join(r["labels"].get("unsupported") or [])))
            for s in data[r["id"]]["samples"]:
                if s:
                    pred = scoring.Counter(scoring.normalize("".join(s)))
                    ov = sum((pred & gold).values())
                    pr[0] += ov / max(1, sum(pred.values()))
                    pr[1] += ov / max(1, sum(gold.values()))
                    pr[2] += sum(pred.values())
                    pr[3] += sum(gold.values())
                    pr[4] += 1
        n_items = len(sel)
        n_ung = sum(not r["labels"]["grounded"] for r in sel)
        print(f"[{split}] items {n_items} (ungrounded {n_ung}), failed samples {failures}")
        if pr[4]:
            print(f"  single flagged spans on ungrounded items: precision {pr[0] / pr[4]:.3f}  recall {pr[1] / pr[4]:.3f}"
                  f"  chars {pr[2] / pr[4]:.1f} (gold {pr[3] / pr[4]:.1f})")
        for rule in rules:
            s, g, f, n = acc[rule]
            print(f"  {rule:8s} output score {s / n:.3f}  grounded acc {g / n:.3f}  span F1 {f / n:.3f}  "
                  f"grounded items flagged {wrong_flag[rule]:.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--collect")
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--style", choices=("minimal", "clause", "clause2"))
    ap.add_argument("--eval")
    a = ap.parse_args()
    if a.collect:
        asyncio.run(collect(a.collect, a.k, a.concurrency, a.style))
    if a.eval:
        evaluate(a.eval)


if __name__ == "__main__":
    main()
