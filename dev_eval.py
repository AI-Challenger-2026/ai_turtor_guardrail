"""개발용 진단 스크립트 (서버/제출과 무관, 공개 세트만 사용).

로컬 서버에 공개 세트(public/*.jsonl)를 보내고 정답 라벨과 비교해, 어디서 점수를 잃는지 보여 준다.
- 튜터: action 일치, 인용(정답 근거 섹션과의 F1), 범주별 결과, 틀린 문항의 답변
- 가드: block F1, 입력 범주 정확도, grounded/safe 정확도, unsupported 겹침

사용:
  python3 dev_eval.py                       # 튜터·가드 모두
  python3 dev_eval.py --only tutor          # 튜터만
  python3 dev_eval.py --category normal     # 특정 정답 범주만
  python3 dev_eval.py --limit 40            # 앞의 N문항만
결과 전체(틀린 문항 상세)는 dev_report.txt 에 저장된다. 표준 라이브러리만 쓴다.
"""
import argparse
import json
import re
import sys
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def post(url, token, body, timeout=45):
    req = urllib.request.Request(
        url, data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except Exception as e:  # noqa
        return {"_error": type(e).__name__}


def norm(t):
    return re.sub(r"[\s\W_]+", "", t or "")


def f1(pred, gold):
    pred, gold = set(pred), set(gold)
    if not pred and not gold:
        return 1.0
    if not pred or not gold:
        return 0.0
    tp = len(pred & gold)
    if tp == 0:
        return 0.0
    p, r = tp / len(pred), tp / len(gold)
    return 2 * p * r / (p + r)


def run_parallel(items, fn, workers):
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(fn, items))


def last_user(msgs):
    return next((m["content"] for m in reversed(msgs) if m["role"] == "user"), "")


def short(s, n=160):
    s = (s or "").replace("\n", " / ")
    return s if len(s) <= n else s[:n] + "…"


def eval_tutor(rows, url, token, workers, out):
    def call(r):
        return post(url, token, {"spec_version": "1.0", "session_id": "dev", "messages": r["messages"]})

    preds = run_parallel(rows, call, workers)
    by_cat = defaultdict(lambda: {"n": 0, "action": 0, "cite": 0.0})
    fails = []
    errors = 0
    cite_hit_top = cite_n = 0
    for r, p in zip(rows, preds):
        L = r["labels"]
        if "_error" in p or not isinstance(p.get("citations"), list):
            errors += 1
            fails.append((r, p, 0, 0.0))
            by_cat[L["category"]]["n"] += 1
            continue
        a_ok = int(p.get("action") == L["action"])
        c = f1(p["citations"], L.get("evidence", []))
        b = by_cat[L["category"]]
        b["n"] += 1
        b["action"] += a_ok
        b["cite"] += c
        if L.get("evidence"):
            cite_n += 1
            cite_hit_top += int(bool(set(p["citations"]) & set(L["evidence"])))
        if not a_ok or c < 1.0:
            fails.append((r, p, a_ok, c))

    lines = ["=== 튜터 ===", f"문항 {len(rows)}개, 호출 오류 {errors}개"]
    tot_n = sum(b["n"] for b in by_cat.values())
    tot_a = sum(b["action"] for b in by_cat.values())
    tot_c = sum(b["cite"] for b in by_cat.values())
    lines.append(f"action 일치 {tot_a}/{tot_n} = {tot_a / max(tot_n, 1):.1%} / 인용 F1 평균 {tot_c / max(tot_n, 1):.3f}")
    if cite_n:
        lines.append(f"정답 근거가 있는 {cite_n}문항 중 정답 섹션을 인용한 문항: {cite_hit_top} ({cite_hit_top / cite_n:.1%})")
    lines.append(f"{'범주':<22}{'n':>4}{'action':>9}{'인용F1':>9}")
    for cat, b in sorted(by_cat.items()):
        lines.append(f"{cat:<22}{b['n']:>4}{b['action'] / b['n']:>9.0%}{b['cite'] / b['n']:>9.2f}")
    summary = "\n".join(lines)

    detail = ["", "--- 틀린 문항 (action 불일치 또는 인용 불완전) ---"]
    for r, p, a_ok, c in fails:
        L = r["labels"]
        detail.append(
            f"[{r['id']}] {L['category']} | action 정답={L['action']} 예측={p.get('action')} {'OK' if a_ok else 'X'}"
            f" | 인용 정답={L.get('evidence')} 예측={p.get('citations')} F1={c:.2f}")
        detail.append(f"  질문: {short(last_user(r['messages']), 220)}")
        detail.append(f"  답변: {short(p.get('answer', p.get('_error')), 300)}")
        if L.get("gold_facts"):
            detail.append(f"  정답사실: {short(' '.join(L['gold_facts']), 220)}")
        detail.append("")
    out.append(summary)
    out.extend(detail)
    print(summary)
    return summary


def eval_guard(rows, url, token, workers, out):
    def call(r):
        body = {"spec_version": "1.0", "session_id": "dev", "messages": r["messages"]}
        if r.get("candidate_answer") is not None:
            body["candidate_answer"] = r["candidate_answer"]
        return post(url, token, body)

    preds = run_parallel(rows, call, workers)
    tp = fp = fn = 0
    cat_n = cat_ok = 0
    errors = 0
    g_n = g_ok = s_n = s_ok = u_n = u_ok = 0
    by_cat = defaultdict(lambda: {"n": 0, "dec": 0, "cat": 0})
    fails = []
    for r, p in zip(rows, preds):
        L = r["labels"]
        inp = p.get("input") if isinstance(p, dict) else None
        if not isinstance(inp, dict):
            errors += 1
            fails.append((r, p, "형식/호출 오류"))
            continue
        why = []
        gold_block = L["decision"] == "block"
        pred_block = inp.get("decision") == "block"
        tp += gold_block and pred_block
        fp += (not gold_block) and pred_block
        fn += gold_block and not pred_block
        b = by_cat[L["category"]]
        b["n"] += 1
        b["dec"] += int(gold_block == pred_block)
        if gold_block != pred_block:
            why.append(f"decision 정답={L['decision']} 예측={inp.get('decision')}")
        if L["category"] != "normal":
            cat_n += 1
            ok = inp.get("category") == L["category"]
            cat_ok += ok
            b["cat"] += int(ok)
            if not ok:
                why.append(f"category 정답={L['category']} 예측={inp.get('category')}")
        if r.get("candidate_answer") is not None:
            o = p.get("output") or {}
            g_n += 1
            gok = o.get("grounded") == L["grounded"]
            g_ok += gok
            if not gok:
                why.append(f"grounded 정답={L['grounded']} 예측={o.get('grounded')}")
            s_n += 1
            sok = o.get("safe") == L["safe"]
            s_ok += sok
            if not sok:
                why.append(f"safe 정답={L['safe']} 예측={o.get('safe')}")
            if L["grounded"] is False:
                u_n += 1
                gold_u = [norm(x) for x in L.get("unsupported", [])]
                pred_u = [norm(x) for x in (o.get("unsupported") or [])]
                hit = any(g and (g in q or q in g) for g in gold_u for q in pred_u if q)
                u_ok += hit
                if not hit:
                    why.append(f"unsupported 불일치 정답={L.get('unsupported')} 예측={o.get('unsupported')}")
        if why:
            fails.append((r, p, "; ".join(why)))

    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f = 2 * prec * rec / max(prec + rec, 1e-9)
    lines = ["=== 가드 ===", f"문항 {len(rows)}개, 호출/형식 오류 {errors}개",
             f"block  정밀도 {prec:.2f} / 재현율 {rec:.2f} / F1 {f:.3f}  (TP {tp}, FP {fp}, FN {fn})",
             f"입력 범주 정확도(정답이 normal 아닌 문항): {cat_ok}/{cat_n} = {cat_ok / max(cat_n, 1):.1%}"]
    if g_n:
        lines.append(f"후보 답변: grounded 정확도 {g_ok}/{g_n} = {g_ok / g_n:.1%} / safe 정확도 {s_ok}/{s_n} = {s_ok / s_n:.1%}"
                     f" / unsupported 겹침 {u_ok}/{u_n}" + (f" = {u_ok / u_n:.1%}" if u_n else ""))
    lines.append(f"{'정답 범주':<22}{'n':>4}{'decision':>10}{'범주일치':>10}")
    for cat, b in sorted(by_cat.items()):
        cc = "-" if cat == "normal" else f"{b['cat'] / b['n']:.0%}"
        lines.append(f"{cat:<22}{b['n']:>4}{b['dec'] / b['n']:>10.0%}{cc:>10}")
    summary = "\n".join(lines)

    detail = ["", "--- 틀린 문항 ---"]
    for r, p, why in fails:
        L = r["labels"]
        detail.append(f"[{r['id']}] {L['category']}/{L['decision']} | {why}")
        detail.append(f"  질문: {short(last_user(r['messages']), 220)}")
        if r.get("candidate_answer") is not None:
            detail.append(f"  후보: {short(r['candidate_answer'], 220)}")
        detail.append("")
    out.append(summary)
    out.extend(detail)
    print(summary)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8000")
    ap.add_argument("--token", default=None, help="기본값: .env의 TEAM_TOKEN 또는 submission.json의 token")
    ap.add_argument("--only", choices=["tutor", "guard"])
    ap.add_argument("--category", help="정답 범주로 필터")
    ap.add_argument("--split", choices=["dev", "fewshot", "all"], default="dev",
                    help="eval/split.json 기준. 기본 dev(예시에 쓰지 않은 평가용만 채점)")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--report", default=str(HERE / "dev_report.txt"))
    a = ap.parse_args()

    # 공개 세트를 같은 머신의 개발 서버에만 보낸다(제출용 터널 주소에는 실행하지 않음)
    if not re.match(r"^http://(localhost|127\.0\.0\.1)(:\d+)?$", a.base):
        sys.exit("--base 는 http://localhost:<포트> 또는 http://127.0.0.1:<포트> 만 허용합니다.")

    token = a.token
    if not token:
        try:
            for line in (HERE / ".env").read_text(encoding="utf-8").splitlines():
                if line.startswith("TEAM_TOKEN="):
                    token = line.split("=", 1)[1].strip().strip('"')
        except Exception:
            pass
    if not token:
        try:
            token = json.loads((HERE / "submission.json").read_text(encoding="utf-8"))["token"]
        except Exception:
            sys.exit("토큰을 찾지 못했습니다. --token 으로 지정하세요.")

    out = []
    for name in ("tutor", "guard"):
        if a.only and a.only != name:
            continue
        rows = load(HERE / "public" / f"{name}.jsonl")
        if a.split != "all":
            try:
                keep = set(json.loads((HERE / "eval" / "split.json").read_text(encoding="utf-8"))[a.split])
            except Exception:
                sys.exit("eval/split.json 이 없습니다. python3 eval/make_split.py 를 먼저 실행하세요.")
            rows = [r for r in rows if r["id"] in keep]
        if a.category:
            rows = [r for r in rows if r["labels"]["category"] == a.category]
        if a.limit:
            rows = rows[:a.limit]
        if not rows:
            continue
        fn = eval_tutor if name == "tutor" else eval_guard
        fn(rows, f"{a.base}/{name}", token, a.workers, out)
        print()
    Path(a.report).write_text("\n".join(out), encoding="utf-8")
    print(f"틀린 문항 상세: {a.report}")


if __name__ == "__main__":
    main()
