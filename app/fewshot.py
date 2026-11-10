"""Load classifier few-shot examples from the public set at start-up.

- Static shots: the ids in fewshot_ids.json.
- Dynamic pool: every fewshot-split item usable as an example (similar ones are picked per
  request in classify.py).
All labels come from public/ (category; harm_kind from the tutor action). The id file may hold
ids only, so no hand-written label can reach the classifier. Start-up fails on an unknown id,
a dev-split id, or an entry with anything but an id.
"""
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

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


def _dev_ids(split_file: Path) -> set:
    return set(json.loads(split_file.read_text(encoding="utf-8"))["dev"]) if split_file.exists() else set()


def to_shot(item: dict) -> Shot:
    """A public item as an example, or ValueError if it cannot be one (labels from public/ only)."""
    category = item["labels"]["category"]
    if category not in CATEGORIES or category in EXCLUDED:
        raise ValueError("category not usable as an example")
    harm_kind = "none"
    if category == "harmful_content":
        harm_kind = _ACTION_TO_HARM_KIND.get(item["labels"].get("action"))
        if harm_kind is None:
            raise ValueError("no public label gives this item's harm_kind")
    messages = tuple({"role": m["role"], "content": m["content"]} for m in item["messages"])
    return Shot(id=item["id"], messages=messages, category=category, harm_kind=harm_kind)


def load_shots(shots_file: Path = SHOTS_FILE, split_file: Path = SPLIT_FILE) -> List[Shot]:
    entries = json.loads(shots_file.read_text(encoding="utf-8"))["shots"]
    if any(set(e) != {"id"} for e in entries):
        raise RuntimeError("few-shot entries must hold ids only; labels come from public/")
    ids = [e["id"] for e in entries]
    if len(set(ids)) != len(ids):
        raise RuntimeError("duplicate few-shot ids")
    leaked = [i for i in ids if i in _dev_ids(split_file)]
    if leaked:
        raise RuntimeError(f"{len(leaked)} few-shot ids belong to the dev split")
    public = _read_public()
    shots = []
    for item_id in ids:
        item = public.get(item_id)
        if item is None:
            raise RuntimeError("unknown few-shot id")
        try:
            shots.append(to_shot(item))
        except ValueError as exc:
            raise RuntimeError(f"few-shot id not usable: {exc}") from None
    return shots


def load_pool(split_file: Path = SPLIT_FILE) -> List[Shot]:
    """Every fewshot-split item that can be an example (dev items never; see to_shot)."""
    split = json.loads(split_file.read_text(encoding="utf-8"))
    dev = set(split["dev"])
    public = _read_public()
    pool = []
    for item_id in split["fewshot"]:
        if item_id in dev or item_id not in public:
            raise RuntimeError("bad fewshot-split id")
        try:
            pool.append(to_shot(public[item_id]))
        except ValueError:
            continue  # low_signal, out_of_material, or harmful without a public harm_kind
    return pool


def group_of(split_file: Path = SPLIT_FILE) -> Dict[str, int]:
    """Item id -> contrast group number (evaluation uses it for leave-group-out)."""
    split = json.loads(split_file.read_text(encoding="utf-8"))
    return {i: g["group"] for g in split["groups"] for i in g["ids"]}
