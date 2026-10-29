"""평가 데이터셋 로더 (dataset loaders) — data/SCHEMA.md 의 스키마를 Pydantic 으로 검증한다.

공식 데이터 형식이 다르면 이 파일에서 변환 함수를 추가해 같은 EvalItem 으로 맞춘다.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

from app.schemas import ChatMessage


class QAItem(BaseModel):
    id: str
    history: list[ChatMessage] = Field(default_factory=list)
    question: str
    q_type: Literal["concept", "exercise", "confusion", "followup"]
    expected_units: list[str] = Field(default_factory=list)
    key_points: list[str] = Field(default_factory=list)


class AttackItem(BaseModel):
    id: str
    history: list[ChatMessage] = Field(default_factory=list)
    prompt: str
    category: Literal["out_of_scope", "harmful_sensitive", "jailbreak", "disguised_attack"]
    expected_decision: Literal["block"] = "block"
    author: str = "unknown"


class BenignItem(BaseModel):
    id: str
    history: list[ChatMessage] = Field(default_factory=list)
    prompt: str
    note: str = ""
    expected_units: list[str] = Field(default_factory=list)
    expected_decision: Literal["allow"] = "allow"


class EvalItem(BaseModel):
    """세 세트를 하나의 형태로 맞춘 평가 단위."""

    set: Literal["qa", "attack", "benign"]
    id: str
    history: list[ChatMessage]
    text: str
    expected_decision: Literal["allow", "block"]
    category: str  # qa: q_type, attack: category, benign: "benign"
    expected_units: list[str] = Field(default_factory=list)
    key_points: list[str] = Field(default_factory=list)
    author: str = ""


def read_jsonl(path: Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"데이터 파일이 없습니다: {path}")
    rows = []
    with path.open("r", encoding="utf-8-sig") as f:  # BOM 이 있어도 읽는다
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno} JSON 형식 오류: {exc}") from exc
    return rows


def _validate(model: type[BaseModel], rows: list[dict], path: Path) -> list:
    out = []
    for i, row in enumerate(rows, start=1):
        try:
            out.append(model(**row))
        except ValidationError as exc:
            raise ValueError(f"{path} {i}번째 항목 스키마 오류: {exc}") from exc
    return out


def load_qa(path: Path) -> list[EvalItem]:
    items: list[QAItem] = _validate(QAItem, read_jsonl(path), path)
    return [
        EvalItem(set="qa", id=x.id, history=x.history, text=x.question, expected_decision="allow", category=x.q_type,
                 expected_units=x.expected_units, key_points=x.key_points)
        for x in items
    ]


def load_attacks(path: Path) -> list[EvalItem]:
    items: list[AttackItem] = _validate(AttackItem, read_jsonl(path), path)
    return [
        EvalItem(set="attack", id=x.id, history=x.history, text=x.prompt, expected_decision="block", category=x.category,
                 author=x.author)
        for x in items
    ]


def load_benign(path: Path) -> list[EvalItem]:
    items: list[BenignItem] = _validate(BenignItem, read_jsonl(path), path)
    return [
        EvalItem(set="benign", id=x.id, history=x.history, text=x.prompt, expected_decision="allow", category="benign",
                 expected_units=x.expected_units)
        for x in items
    ]
