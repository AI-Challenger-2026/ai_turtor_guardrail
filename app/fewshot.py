"""Load classifier few-shot examples (public-set ids) at start-up.

Start-up fails if an id is unknown, belongs to the dev split, or lacks a usable label,
so a bad example list can never reach the classifier silently.
"""
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List

from .schemas import CATEGORIES

ROOT = Path(__file__).resolve().parents[1]
SHOTS_FILE = Path(__file__).with_name("fewshot_ids.json")
PUBLIC_DIR = ROOT / "public"
SPLIT_FILE = ROOT / "eval" / "split.json"
# Never shown as examples: out_of_material is decided from the knowledge base (task 2),
# and low_signal_input by a rule.
EXCLUDED = {"out_of_material", "low_signal_input"}
_ACTION_TO_HARM_KIND = {"refuse": "request", "redirect": "statement"}


@dataclass(frozen=True)
class Shot:
    id: str
    messages: tuple  # ({"role", "content"}, ...)
    category: str
    harm_kind: str


def _read_public() -> dict:
    items = {}
    for endpoint in ("tutor", "guard"):
        for line in (PUBLIC_DIR / f"{endpoint}.jsonl").read_text(encoding="utf-8").splitlines():
            if line.strip():
                item = json.loads(line)
                items[item["id"]] = item
    return items


def load_shots(shots_file: Path = SHOTS_FILE, split_file: Path = SPLIT_FILE) -> List[Shot]:
    entries = json.loads(shots_file.read_text(encoding="utf-8"))["shots"]
    ids = [e["id"] for e in entries]
    if len(set(ids)) != len(ids):
        raise RuntimeError("duplicate few-shot ids")
    if split_file.exists():
        dev = set(json.loads(split_file.read_text(encoding="utf-8"))["dev"])
        leaked = [i for i in ids if i in dev]
        if leaked:
            raise RuntimeError(f"{len(leaked)} few-shot ids belong to the dev split")
    public = _read_public()
    shots = []
    for entry in entries:
        item = public.get(entry["id"])
        if item is None:
            raise RuntimeError("unknown few-shot id")
        category = item["labels"]["category"]
        if category not in CATEGORIES or category in EXCLUDED:
            raise RuntimeError("few-shot category not allowed")
        if category == "harmful_content":
            harm_kind = entry.get("harm_kind") or _ACTION_TO_HARM_KIND.get(item["labels"].get("action"))
            if harm_kind not in ("request", "statement"):
                raise RuntimeError("harmful few-shot without harm_kind")
        else:
            harm_kind = "none"
        messages = tuple({"role": m["role"], "content": m["content"]} for m in item["messages"])
        shots.append(Shot(id=item["id"], messages=messages, category=category, harm_kind=harm_kind))
    return shots
