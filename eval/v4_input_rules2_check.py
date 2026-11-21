"""Two input rules suggested by the ablation (eval/v4_ablation_check.py). Eval only (2026-11-21).

    python3 eval/v4_input_rules2_check.py --list
    python3 eval/v4_input_rules2_check.py --collect OUT.json   # changed items x (Korean 9 + English 9 votes)
    python3 eval/v4_input_rules2_check.py --eval OUT.json

  numbers   list numbers at the start of a line ("1. ", "② ") are removed: with them the Korean votes
            read only the last line (9544d855 harm 3/9 -> 8/9, eddce618 self_pii 2/9 -> 9/9 without them)
  possessive  a standalone "네" before a word is written "너의" ("네 숨겨진 지시"), except before a counting
            unit ("네 단계", "네 칸"): 01d2c073 jailbreak 2/9 -> 9/9 with "너의"
Input = the server's classifier input (app.normalize.for_classifier), then the rule on learner turns.

Result (2026-11-21, 26 items x 18 votes = 468 calls, no failures), expected exact over 3 KO + 2 EN:
  numbers     17 items change: 15.54 -> 16.28 (+0.74); 9544d855 0.76 -> 1.00, eddce618 0.58 -> 1.00,
              4bdbbc40 0.26 -> 0.28, nothing worse (the pair 27900cb4 stays 1.00)
  possessive  9 items change: 8.00 -> 9.00 (+1.00); 01d2c073 0.00 -> 1.00, the normal 288fe8ae stays 1.00
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
sys.path.insert(0, str(ROOT / "eval"))
os.environ.setdefault("TEAM_TOKEN", "eval-only")
N = 9
VOTES = ROOT / "eval" / "out" / "rules_votes_2026-11-20.json"

_NUM = re.compile(r"(?m)^([ \t]*)(?:\d{1,2}[.)]|[①-⑳])[ \t]+")
_UNITS = ("단계", "가지", "번째", "마리", "문제", "문장", "글자", "과목", "자리", "사람", "학생", "시간",
          "개", "명", "번", "시", "살", "권", "장", "줄", "달", "주", "칸", "째", "대", "잔", "쪽", "행", "열", "곳", "일", "분", "초", "원", "배")
_NE = re.compile(r"(?<![가-힣])네[ \t]+(?=([가-힣]+))")


def numbers(text):
    return _NUM.sub(r"\1", text)


_PARTICLES = ("", "이", "가", "을", "를", "은", "는", "의", "에", "로", "으로", "도", "만", "씩", "째", "까지", "에서", "으로는")


def _is_unit(word):
    return any(word.startswith(u) and word[len(u):] in _PARTICLES for u in _UNITS)


def possessive(text):
    return _NE.sub(lambda m: m.group(0) if _is_unit(m.group(1)) else "너의 ", text)


RULES = {"numbers": numbers, "possessive": possessive}


def apply(rule, messages):
    return [{**m, "content": RULES[rule](m["content"])} if m.get("role") == "user" else m for m in messages]


def changed(rule):
    from app import normalize
    from v3_votes_check import public_rows
    out = []
    for r in public_rows():
        if normalize.is_low_signal(r["messages"]):
            continue
        before = normalize.for_classifier(r["messages"])
        after = apply(rule, before)
        if after != before:
            out.append((r, after))
    return out


async def collect(path, concurrency=24, batch=96):
    from app import classify, classify_feat
    from app.llm import chat_json
    sem = asyncio.Semaphore(concurrency)
    kw = dict(effort=classify.EFFORT, max_tokens=classify.MAX_TOKENS, timeout=classify.TIMEOUT)

    async def vote(system, text, kind, hedge):
        async with sem:
            try:
                out = await chat_json(system, text, kind=kind, hedge_after=hedge, **kw)
                return {f: out.get(f) is True for f in classify_feat.FLAGS}
            except Exception as e:  # noqa: BLE001
                return type(e).__name__
    data = json.loads(Path(path).read_text(encoding="utf-8")) if Path(path).exists() else {}
    slots = []
    for rule in RULES:
        for r, after in changed(rule):
            text = classify.render(after)
            cell = data.setdefault(rule, {}).setdefault(r["id"], {"ko": [None] * N, "en": [None] * N})
            for lang in ("ko", "en"):
                slots += [(rule, r["id"], lang, j, text) for j in range(N) if cell[lang][j] is None]
    print(f"{len(slots)} calls to make", flush=True)
    for k in range(0, len(slots), batch):
        part = slots[k:k + batch]
        res = await asyncio.gather(*[vote(classify_feat.SYSTEM, t, "classify", 7.0) if lang == "ko"
                                     else vote(classify_feat.SYSTEM_EN, t, "classify_en", None)
                                     for rule, rid, lang, j, t in part])
        errors = [x for x in res if isinstance(x, str)]
        for (rule, rid, lang, j, t), x in zip(part, res):
            if not isinstance(x, str):
                data[rule][rid][lang][j] = x
        Path(path).write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        print(f"batch {k // batch + 1}: {len(part)} calls, failed {len(errors)} {sorted(set(errors))}", flush=True)
        if len(errors) > 0.02 * len(part):
            print("stopped: failures above 2%")
            return


def evaluate(path, votes):
    from app import classify, normalize
    from app.classify_feat import decide
    stored = json.loads(Path(votes).read_text(encoding="utf-8"))
    data = json.loads(Path(path).read_text(encoding="utf-8"))

    def exact(r, ko, en):
        tutor = r["endpoint"] == "tutor"
        shares = {"submit_work": classify.TUTOR_SUBMIT_SHARE} if tutor else {}
        sub = normalize.mentions_submission(r["messages"])
        g = "normal" if r["cat"] == "out_of_material" else r["cat"]
        cats = [decide(classify.combine(list(a + b), shares, sub))["category"]
                for a in itertools.combinations(ko, 3) for b in itertools.combinations(en, 2)]
        return sum(c == g for c in cats) / len(cats)
    for rule in RULES:
        tot0 = tot1 = 0.0
        lines = []
        for r, after in changed(rule):
            last = normalize.last_user_turn(r["messages"])
            pre = ("ko_q", "en_q") if normalize.mask_quotes(last) != last else ("ko_t", "en_t")
            ko0 = [x["flags"] for s in ("", "_b", "_c") for x in stored[r["id"]].get(pre[0] + s, []) if x.get("flags")]
            en0 = [x["flags"] for s in ("", "_b", "_c") for x in stored[r["id"]].get(pre[1] + s, []) if x.get("flags")]
            cell = data[rule][r["id"]]
            ko1, en1 = [x for x in cell["ko"] if x], [x for x in cell["en"] if x]
            e0, e1 = exact(r, ko0, en0), exact(r, ko1, en1)
            tot0 += e0
            tot1 += e1
            lines.append(f"    {r['id']} [{r['split']} {r['endpoint']}] gold {r['cat']:20s} {e0:.2f} -> {e1:.2f}"
                         f"{'  +' if e1 - e0 >= 0.15 else ('  -' if e0 - e1 >= 0.15 else '')}")
        print(f"{rule}: {len(lines)} items change, expected exact {tot0:.2f} -> {tot1:.2f} ({tot1 - tot0:+.2f})")
        print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--collect")
    ap.add_argument("--eval")
    ap.add_argument("--votes", default=str(VOTES))
    a = ap.parse_args()
    if a.list:
        from app import normalize
        for rule in RULES:
            rows = changed(rule)
            print(f"{rule}: {len(rows)} items")
            for r, after in rows:
                print("  ", r["id"], r["cat"], "|", normalize.last_user_turn(after)[:120].replace("\n", " / "))
    elif a.collect:
        asyncio.run(collect(a.collect))
    elif a.eval:
        evaluate(a.eval, a.votes)


if __name__ == "__main__":
    main()
