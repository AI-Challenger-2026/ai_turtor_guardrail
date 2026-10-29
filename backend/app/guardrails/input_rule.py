"""InputRuleGuard (규칙, input) — 정규식으로 명백한 탈옥 패턴과 개인정보를 잡는다.

규칙은 configs/input_rules.yaml 에 있다 (params.rules_file 로 다른 파일 지정 가능).
LLM 호출이 없어 빠르고 공짜이므로 입력 단계에서 가장 먼저 실행된다.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from app.guardrails.base import Guard, GuardContext, GuardDeps
from app.guardrails.registry import register_guard
from app.schemas import GuardResult


@dataclass
class _Rule:
    id: str
    category: str
    patterns: list[re.Pattern[str]]


def load_rules(path: Path) -> list[_Rule]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    rules = []
    for item in data.get("rules", []):
        rules.append(
            _Rule(
                id=str(item["id"]),
                category=str(item.get("category", "jailbreak")),
                patterns=[re.compile(p, re.IGNORECASE) for p in item.get("patterns", [])],
            )
        )
    return rules


@register_guard
class InputRuleGuard(Guard):
    name = "input_rule"
    stage = "input"
    kind = "rule"
    description = "정규식 기반 탈옥·개인정보 탐지"

    def __init__(self, deps: GuardDeps, **params) -> None:
        super().__init__(deps, **params)
        rules_file = params.get("rules_file", "input_rules.yaml")
        path = Path(rules_file)
        if not path.is_absolute():
            path = deps.configs_dir / path
        self.rules = load_rules(path)
        self.disabled = set(params.get("disabled_rules", []))

    def check(self, ctx: GuardContext) -> GuardResult:
        texts = [ctx.question]
        if ctx.rewritten_question and ctx.rewritten_question != ctx.question:
            texts.append(ctx.rewritten_question)
        for rule in self.rules:
            if rule.id in self.disabled:
                continue
            for pattern in rule.patterns:
                for text in texts:
                    m = pattern.search(text)
                    if m:
                        snippet = m.group(0)
                        if rule.category == "pii":
                            snippet = snippet[:3] + "***"  # 판정 기록에도 개인정보를 그대로 남기지 않는다
                        return self.result(
                            "block", rule.category, f"규칙 '{rule.id}' 에 매칭: '{snippet[:40]}'", 1.0, rule=rule.id
                        )
        return self.result("allow", "none", "매칭된 규칙 없음", 0.0)
