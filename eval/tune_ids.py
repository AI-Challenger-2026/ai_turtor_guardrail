"""Print the tuning-set ids: fewshot-split items that are not used as few-shot examples.

Prompt/few-shot changes are tuned on this set so the dev split stays a clean check.
Usage: python3 eval/guard_report.py --data public/guard.jsonl public/tutor.jsonl --ids @<(python3 eval/tune_ids.py)
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    split = json.loads((ROOT / "eval/split.json").read_text(encoding="utf-8"))
    shots = {s["id"] for s in json.loads((ROOT / "app/fewshot_ids.json").read_text(encoding="utf-8"))["shots"]}
    for item_id in split["fewshot"]:
        if item_id not in shots:
            print(item_id)
