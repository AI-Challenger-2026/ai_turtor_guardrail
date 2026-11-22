"""Print the ids of the tuning half of the public set.

    python3 eval/tune_ids.py > /tmp/tune_ids.txt
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
