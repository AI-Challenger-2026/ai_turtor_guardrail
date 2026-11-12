"""Print the tuning-set ids: fewshot-split items that are not used as few-shot examples.

Prompt/few-shot changes are tuned on this set so the dev split stays a clean check.
The final-v3 server has no public few-shot examples, so without app/fewshot_ids.json this is the
whole fewshot split. (final-v3 itself was tuned on all 400 public items by its author.)
Usage: python3 eval/guard_report.py --data public/guard.jsonl public/tutor.jsonl --ids @<(python3 eval/tune_ids.py)
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    split = json.loads((ROOT / "eval/split.json").read_text(encoding="utf-8"))
    shots_file = ROOT / "app/fewshot_ids.json"
    shots = ({s["id"] for s in json.loads(shots_file.read_text(encoding="utf-8"))["shots"]}
             if shots_file.exists() else set())
    for item_id in split["fewshot"]:
        if item_id not in shots:
            print(item_id)
