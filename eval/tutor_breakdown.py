"""Tutor score broken down by grader item: action, citation F1, gold facts, faithfulness.

Uses the local grader's own scoring (grader/scoring.py) and judge (grader/judge.py, OPENAI_*
environment, the same as harness/score_local.py), so the per-item points match score_local.
Prints per-category means of each item and how much each item costs the category's points,
then the lowest-scoring items (ids and item values only).

--dump writes the answers with the gold facts to a local file for offline inspection of the
public items evaluated (development use only; the server never logs answers).

  python3 eval/tutor_breakdown.py --url http://localhost:8001/tutor --data public/tutor.jsonl \
      --ids @/tmp/tune_ids.txt --categories normal,pii_disclosure,answer_seeking,out_of_material
"""
import argparse
import json
import os
import secrets
import sys
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "grader"))
sys.path.insert(0, str(ROOT / "harness"))
import scoring  # noqa: E402
from judge import make_judge  # noqa: E402
from score_local import as_scoring_input, load_kb_dir  # noqa: E402

PARTS = ("action", "citation", "facts", "faithful")


def read_items(path, ids_arg, cats_arg):
    items = [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]
    if ids_arg:
        raw = Path(ids_arg[1:]).read_text(encoding="utf-8") if ids_arg.startswith("@") else ids_arg
        keep = {x.strip() for x in raw.replace(",", "\n").splitlines() if x.strip()}
        items = [it for it in items if it["id"] in keep]
    if cats_arg:
        keep = set(cats_arg.split(","))
        items = [it for it in items if it["labels"]["category"] in keep]
    return items


def parts_of(r):
    """The grader's per-item parts (None when the part does not apply)."""
    cit = None
    if r["citation_p"] is not None:
        p, rc = r["citation_p"], r["citation_r"]
        cit = 2 * p * rc / (p + rc) if p + rc else 0.0
    return {"action": float(r["action_correct"]) if r["valid"] else 0.0, "citation": cit,
            "facts": r["facts"], "faithful": None if r["faithful"] is None else float(r["faithful"])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8001/tutor")
    ap.add_argument("--token", default=os.environ.get("TEAM_TOKEN", ""))
    ap.add_argument("--data", default=str(ROOT / "public/tutor.jsonl"))
    ap.add_argument("--ids", default="")
    ap.add_argument("--categories", default="")
    ap.add_argument("--kb", default=str(ROOT / "kb"))
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--dump", default="")
    a = ap.parse_args()

    env = {k: os.environ.get(f"OPENAI_{k.upper()}") for k in ("base_url", "model", "api_key")}
    if not all(env.values()):
        sys.exit("OPENAI_* settings are missing (load .env first)")
    judge = make_judge(**env)
    kb = load_kb_dir(a.kb)
    items = read_items(a.data, a.ids, a.categories)
    opener = urllib.request.build_opener()

    def one(p):
        item, v = as_scoring_input(p)
        payload = {"spec_version": "1.0", "session_id": "s_" + secrets.token_hex(8), "messages": v["messages"]}
        req = urllib.request.Request(a.url, data=json.dumps(payload, ensure_ascii=False).encode("utf-8"), method="POST",
                                     headers={"Content-Type": "application/json", "Authorization": f"Bearer {a.token}"})
        try:
            with opener.open(req, timeout=30) as resp:
                out = json.loads(resp.read(256 * 1024 + 1))
        except Exception:  # noqa: BLE001
            out = None
        r = scoring.score_tutor(item, v, out, judge, kb)
        return {"id": p["id"], "cat": p["labels"]["category"], "gold_action": p["labels"]["action"],
                "pred_action": out.get("action") if isinstance(out, dict) else None,
                "cited": out.get("citations") if isinstance(out, dict) else None,
                "evidence": p["labels"].get("evidence") or [], "points": scoring.tutor_points(r),
                "complied": r["complied"], "judge_error": r["judge_error"], "parts": parts_of(r),
                "answer": out.get("answer") if isinstance(out, dict) else None,
                "gold_facts": p["labels"].get("gold_facts") or []}

    with ThreadPoolExecutor(a.workers) as ex:
        rows = list(ex.map(one, items))

    by = defaultdict(list)
    for r in rows:
        by[r["cat"]].append(r)
    print(f"{len(rows)} items; judge errors {sum(r['judge_error'] for r in rows)}")
    print(f"{'category':16s} {'n':>3s} {'points':>7s}  " + "  ".join(f"{p:>9s}" for p in PARTS) + "   loss by part (points)")
    for cat, rs in sorted(by.items()):
        means, losses = [], []
        for part in PARTS:
            vals = [r["parts"][part] for r in rs if r["parts"][part] is not None]
            means.append(f"{sum(vals) / len(vals):9.3f}" if vals else f"{'-':>9s}")
            # Each item's points are the mean of its applicable parts, so a part costs (1 - value) / k.
            lost = sum((1 - r["parts"][part]) / sum(v is not None for v in r["parts"].values())
                       for r in rs if r["parts"][part] is not None)
            losses.append(f"{part} {100 * lost / len(rs):.1f}")
        pts = 100 * sum(r["points"] for r in rs) / len(rs)
        print(f"{cat:16s} {len(rs):3d} {pts:7.2f}  " + "  ".join(means) + "   " + ", ".join(losses))
    print("lowest-scoring items:")
    for r in sorted(rows, key=lambda r: r["points"])[:20]:
        if r["points"] >= 1.0:
            break
        pv = " ".join(f"{k}={'-' if v is None else round(v, 2)}" for k, v in r["parts"].items())
        print(f"  {r['id']}  {r['cat']:15s} points {100 * r['points']:5.1f}  action {r['pred_action']}/{r['gold_action']}  "
              f"{pv}  cited {r['cited']} evidence {r['evidence']}")
    if a.dump:
        Path(a.dump).write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
