"""Which item types lower a sampled scoring round.

    python3 eval/v4_low_rounds.py --run RESULTS.json --base http://localhost:8001
    python3 eval/v4_low_rounds.py --rounds RESULTS.json
"""
import argparse
import json
import os
import random
import re
import secrets
import sys
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "grader"))
sys.path.insert(0, str(ROOT / "harness"))
import scoring

HANGUL = re.compile(r"[가-힣]")
LATIN = re.compile(r"[A-Za-z]")


def public():
    out = []
    for ep in ("tutor", "guard"):
        for line in (ROOT / f"public/{ep}.jsonl").read_text(encoding="utf-8").splitlines():
            if line.strip():
                out.append(json.loads(line))
    return out


def english(p):
    users = [m["content"] for m in p["messages"] if m["role"] == "user"]
    last = users[-1] if users else ""
    return bool(LATIN.search(last)) and not HANGUL.search(last)


def run(path, base, token, workers):
    from judge import make_judge
    from score_local import as_scoring_input, load_kb_dir
    env = {k: os.environ.get(f"OPENAI_{k.upper()}") for k in ("base_url", "model", "api_key")}
    judge = make_judge(**env)
    kb = load_kb_dir(ROOT / "kb")
    opener = urllib.request.build_opener()

    def call(endpoint, payload):
        req = urllib.request.Request(f"{base}/{endpoint}", data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                                     method="POST", headers={"Content-Type": "application/json",
                                                             "Authorization": f"Bearer {token}"})
        with opener.open(req, timeout=30) as resp:
            return json.loads(resp.read(256 * 1024 + 1))

    def one(p):
        item, v = as_scoring_input(p)
        payload = {"spec_version": "1.0", "session_id": "s_" + secrets.token_hex(8), "messages": v["messages"]}
        if v["candidate_answer"]:
            payload["candidate_answer"] = v["candidate_answer"]
        try:
            resp = call(p["endpoint"], payload)
        except Exception:
            resp = None
        if p["endpoint"] == "tutor":
            r = scoring.score_tutor(item, v, resp, judge, kb)
        else:
            r = scoring.score_guard(item, v, resp)
        action = resp.get("action") if isinstance(resp, dict) and p["endpoint"] == "tutor" else None
        return {"id": p["id"], "endpoint": p["endpoint"], "category": p["labels"]["category"],
                "gold_action": p["labels"].get("action"), "english": english(p), "action": action, "r": r}

    with ThreadPoolExecutor(workers) as ex:
        res = list(ex.map(one, public()))
    Path(path).write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")
    tutor = [d["r"] for d in res if d["endpoint"] == "tutor"]
    guard = [d["r"] for d in res if d["endpoint"] == "guard"]
    agg = scoring.aggregate(tutor, guard, {"tutor": 0.5, "guard": 0.5})
    print(f"saved {path}: total {agg['total']} (tutor {agg['tutor']}, guard {agg['guard']}), invalid {agg['invalid']}, "
          f"judge errors {agg['judge_errors']}")


def item_type(d):
    if d["endpoint"] == "tutor":
        if d["english"]:
            return "tutor: English question"
        if d["action"] in ("refuse", "redirect") and d["gold_action"] not in ("refuse", "redirect"):
            return "tutor: refused though not an attack"
    return f"{d['endpoint']}: {d['category']}"


def item_loss(d):
    if d["endpoint"] == "tutor":
        return 1 - scoring.tutor_points(d["r"]) if not d["r"].get("judge_error") else 0.0
    return 1 - scoring.guard_points(d["r"])


def rounds(path, n, seed):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    tutor = [d for d in data if d["endpoint"] == "tutor"]
    guard = [d for d in data if d["endpoint"] == "guard"]
    rng = random.Random(seed)
    sims = []
    for _ in range(n):
        ts = [rng.choice(tutor) for _ in range(300)]
        gs = [rng.choice(guard) for _ in range(300)]
        agg = scoring.aggregate([d["r"] for d in ts], [d["r"] for d in gs], {"tutor": 0.5, "guard": 0.5})
        counts, loss = Counter(), Counter()
        for d in ts + gs:
            t = item_type(d)
            counts[t] += 1
            loss[t] += item_loss(d) * 100 / 300 / 2
        sims.append((agg["total"], agg["tutor"], agg["guard"], counts, loss))
    sims.sort(key=lambda s: s[0])
    tot = [s[0] for s in sims]
    mean = sum(tot) / n
    sd = (sum((x - mean) ** 2 for x in tot) / (n - 1)) ** 0.5
    print(f"{n} imitated rounds: total mean {mean:.2f}, sd {sd:.2f}, p10 {tot[n // 10]:.2f}, p50 {tot[n // 2]:.2f}, "
          f"p90 {tot[9 * n // 10]:.2f}, min {tot[0]:.2f}, max {tot[-1]:.2f}")
    low = sims[: n // 10]
    print(f"lowest 10%: tutor {sum(s[1] for s in low) / len(low):.2f} (all {sum(s[1] for s in sims) / n:.2f}), "
          f"guard {sum(s[2] for s in low) / len(low):.2f} (all {sum(s[2] for s in sims) / n:.2f})")

    types = sorted({item_type(d) for d in data})
    stats = {}
    for t in types:
        members = [d for d in data if item_type(d) == t]
        pts = sum(1 - item_loss(d) for d in members) / len(members)
        c_all = sum(s[3][t] for s in sims) / n
        c_low = sum(s[3][t] for s in low) / len(low)
        l_all = sum(s[4][t] for s in sims) / n
        l_low = sum(s[4][t] for s in low) / len(low)
        stats[t] = (len(members), pts, c_all, c_low, l_all, l_low)
    print("\nitem type (public items, mean item points) | items per round: all -> lowest 10% | points lost: all -> lowest 10%")
    for t, (m, pts, c_all, c_low, l_all, l_low) in sorted(stats.items(), key=lambda kv: -(kv[1][5] - kv[1][4])):
        print(f"  {t:38s} n {m:3d}  pts {pts * 100:5.1f} | {c_all:5.1f} -> {c_low:5.1f} | {l_all:4.2f} -> {l_low:4.2f}  "
              f"(+{l_low - l_all:.2f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run")
    ap.add_argument("--base", default="http://localhost:8001")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--rounds")
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=20261120)
    a = ap.parse_args()
    if a.run:
        run(a.run, a.base, os.environ["TEAM_TOKEN"], a.workers)
    elif a.rounds:
        rounds(a.rounds, a.n, a.seed)


if __name__ == "__main__":
    main()
