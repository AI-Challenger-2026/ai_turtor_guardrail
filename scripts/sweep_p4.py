"""P4 오프라인 스윕(DEVLOG 18장). LLM 호출은 '수집' 단계에서 한 번만 하고, 임계값·구간 후처리는 캐시 위에서 훑는다.

  python3 scripts/sweep_p4.py collect CACHE.json   # 분류 5표 x 400문항 + 검증기 원본 출력 x 67문항 (약 2,100회 호출)
  python3 scripts/sweep_p4.py sweep CACHE.json     # 호출 없음

(a) 분류 투표 임계값: 5표 중 몇 표 이상이면 신호를 true로 볼지(막는 신호 / 그 외 신호 따로).
    기준 = 앞 3표 중 2표 이상(현재 서버와 같음). 지표: 가드 block F1, 범주 정확도(normal 제외), 튜터 action 정확도.
    짝/홀 문항으로 나눠 한쪽에서 고른 값이 다른 쪽에서도 좋은지 본다.
(b) 근거 없는 구간 후처리: 검증기가 준 wrong_part 주변을 몇 어절 넓히면 구간 F1이 오르는지.
"""
import asyncio
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "grader"))
import scoring  # noqa: E402
from app import classify_feat, verify  # noqa: E402
from app.kb import KB  # noqa: E402
from app.llm import chat_json  # noqa: E402
from app.policy import BLOCK_CATEGORIES, guard_decision, tutor_action  # noqa: E402

BLOCK_FLAGS = ("jailbreak", "harm", "hate", "submit_work", "off_topic")
OTHER_FLAGS = tuple(f for f in classify_feat.FLAGS if f not in BLOCK_FLAGS)
VOTES = 5


def load(name):
    return [json.loads(l) for l in (ROOT / "public" / f"{name}.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]


async def collect(path):
    from app.classify import render as crender
    sem = asyncio.Semaphore(32)
    rows = [("tutor", r) for r in load("tutor")] + [("guard", r) for r in load("guard")]

    async def vote(user):
        async with sem:
            try:
                return await chat_json(classify_feat.SYSTEM, user, effort="low", max_tokens=2000, timeout=25)
            except Exception:  # noqa: BLE001
                return None

    votes = await asyncio.gather(*[asyncio.gather(*[vote(crender(r["messages"])) for _ in range(VOTES)]) for _, r in rows])
    kb = KB()
    cands = [r for r in load("guard") if "candidate_answer" in r and r["labels"]["category"] not in BLOCK_CATEGORIES]

    async def ver(r):
        cand = r["candidate_answer"]
        q = "\n".join(m["content"] for m in r["messages"][-4:]) + "\n" + cand + "\n" + cand
        secs = [kb.secs[s] for s, _ in kb.rank(q)[:4]]
        ctx = "\n\n".join(f"<섹션 id=\"{s['id']}\" 제목=\"{s['title']}\">\n{s['text']}\n</섹션>" for s in secs)
        user = f"[자료]\n{ctx}\n\n[대화]\n{verify.render(r['messages'])}\n\n[후보 답변]\n{cand}"
        async with sem:
            try:
                return await chat_json(verify.SYSTEM, user, effort="low", max_tokens=2000, timeout=25)
            except Exception:  # noqa: BLE001
                return None

    vers = await asyncio.gather(*[ver(r) for r in cands])
    out = {"votes": [{"ep": ep, "id": r["id"], "votes": v} for (ep, r), v in zip(rows, votes)],
           "verify": [{"id": r["id"], "out": v} for r, v in zip(cands, vers)]}
    Path(path).write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print("saved", path, "votes failed:", sum(x is None for row in votes for x in row), "verify failed:", sum(v is None for v in vers))


def decide_from(votes, tb, to, first=None):
    vs = [v for v in (votes[:first] if first else votes) if isinstance(v, dict)]
    if not vs:
        return {"category": "normal", "harmful_type": "none"}
    flags = {k: sum(v.get(k) is True for v in vs) >= (tb if k in BLOCK_FLAGS else to) for k in classify_feat.FLAGS}
    return classify_feat.decide(flags)


def metrics(items, tb, to, first=None):
    g = [(r, decide_from(c["votes"], tb, to, first)) for r, c in items if c["ep"] == "guard"]
    t = [(r, decide_from(c["votes"], tb, to, first)) for r, c in items if c["ep"] == "tutor"]
    tp = sum(r["labels"]["decision"] == "block" and guard_decision(p["category"]) == "block" for r, p in g)
    fp = sum(r["labels"]["decision"] == "allow" and guard_decision(p["category"]) == "block" for r, p in g)
    fn = sum(r["labels"]["decision"] == "block" and guard_decision(p["category"]) == "allow" for r, p in g)
    nn = [(r, p) for r, p in g if r["labels"]["category"] != "normal"]
    return dict(f1=2 * tp / max(1, 2 * tp + fp + fn), fp=fp, fn=fn,
                cat=sum(r["labels"]["category"] == p["category"] for r, p in nn) / max(1, len(nn)),
                act=sum(r["labels"]["action"] == tutor_action(p["category"], p["harmful_type"]) for r, p in t) / max(1, len(t)))


def expand(cand, wp, k):
    i = cand.find(wp)
    if i < 0 or k == 0:
        return wp
    left = list(re.finditer(r"\S+", cand[:i]))
    right = list(re.finditer(r"\S+", cand[i + len(wp):]))
    s = left[-k].start() if len(left) >= k else 0
    e = i + len(wp) + (right[k - 1].end() if len(right) >= k else (len(cand) - i - len(wp)))
    return cand[s:e]


def sweep(path):
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    src = {("tutor", r["id"]): r for r in load("tutor")} | {("guard", r["id"]): r for r in load("guard")}
    items = [(src[(c["ep"], c["id"])], c) for c in d["votes"]]
    print(f"분류 투표 수집: {len(items)}문항 x {VOTES}표")
    base = metrics(items, 2, 2, first=3)
    print(f"기준(앞 3표 중 2표 이상) block F1 {base['f1']:.3f} (FP {base['fp']}, FN {base['fn']}) 범주 {base['cat']:.3f} 튜터 action {base['act']:.3f}")
    print("\n막는 신호 임계값(tb) / 그 외 신호 임계값(to), 5표 중:")
    print("  tb to | F1(전체) FP FN | 범주 | 튜터action | F1 짝 / 홀")
    ev = [x for i, x in enumerate(items) if i % 2 == 0]
    od = [x for i, x in enumerate(items) if i % 2 == 1]
    for tb in (1, 2, 3, 4):
        for to in (3,):
            m = metrics(items, tb, to); a = metrics(ev, tb, to); b = metrics(od, tb, to)
            print(f"  {tb}  {to} | {m['f1']:.3f}  {m['fp']:2d} {m['fn']:2d} | {m['cat']:.3f} | {m['act']:.3f} | {a['f1']:.3f} / {b['f1']:.3f}")
    for to in (2, 4):
        m = metrics(items, 3, to); a = metrics(ev, 3, to); b = metrics(od, 3, to)
        print(f"  3  {to} | {m['f1']:.3f}  {m['fp']:2d} {m['fn']:2d} | {m['cat']:.3f} | {m['act']:.3f} | {a['f1']:.3f} / {b['f1']:.3f}")

    print("\n구간 후처리(검증기 원본 출력 위, 구간 F1은 근거 없는 답변 중 탐지한 것 기준)")
    cands = {r["id"]: r for r in load("guard")}
    for k in (0, 1, 2, 3):
        f1s, nflag, fpn = [], 0, 0
        for v in d["verify"]:
            if not isinstance(v["out"], dict):
                continue
            r = cands[v["id"]]; cand = r["candidate_answer"]; spans = []
            for c in v["out"].get("claims") or []:
                if not isinstance(c, dict) or c.get("verdict") not in ("contradict", "absent"):
                    continue
                wp, txt = c.get("wrong_part"), c.get("text")
                if isinstance(wp, str) and wp.strip() and verify._norm(wp) in verify._norm(cand):
                    s = expand(cand, wp.strip(), k)
                else:
                    s = txt
                if isinstance(s, str) and s.strip() and verify._norm(s) in verify._norm(cand) and not verify._is_non_claim(cand, s):
                    spans.append(s.strip())
            gold = r["labels"]["unsupported"]
            if gold and spans:
                f1s.append(scoring.span_f1(spans, gold))
            fpn += (not gold) and bool(spans)
            nflag += bool(gold and spans)
        print(f"  wrong_part ±{k}어절: 탐지 {nflag}건, 구간 F1 {sum(f1s) / max(1, len(f1s)):.3f}, 정상 답변 오탐 {fpn}건")


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] not in ("collect", "sweep"):
        raise SystemExit(__doc__)
    asyncio.run(collect(sys.argv[2])) if sys.argv[1] == "collect" else sweep(sys.argv[2])
