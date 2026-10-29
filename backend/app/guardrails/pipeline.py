"""가드 파이프라인 (guard pipeline) — 프로필 YAML 에 적힌 가드를 순서대로 실행한다.

실행 규칙:
- 같은 단계 안에서 규칙(rule) 가드를 LLM 가드보다 먼저 실행한다 (enforce_rule_first, 비용 관리).
- block 이 나오면 그 단계의 나머지 가드는 생략한다.
- revise 는 다음 가드를 계속 실행한다 (안전 가드가 block 할 수 있고, 수정 지시를 한 번에 모으기 위해).
- 가드 내부 오류: 인프라 오류(인증·429·타임아웃·연결·미구현)는 그대로 올려 사용자에게 오류를 보여 주고,
  그 밖의 오류(JSON 해석 실패, 코드 버그)는 on_error 정책(allow | block)으로 처리한다.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field

from app.errors import (
    InvalidRequestError,
    LLMAPIError,
    LLMAuthError,
    LLMConfigError,
    LLMConnectionError,
    LLMNotImplementedError,
    LLMRateLimitError,
    LLMRefusalError,
    LLMTimeoutError,
)
from app.guardrails.base import Guard, GuardContext, GuardDeps
from app.guardrails.registry import get_guard_class
from app.logging_utils import get_logger
from app.schemas import GuardInfo, GuardResult
from app.settings import GuardsSection

log = get_logger("guardrails.pipeline")

_INFRA_ERRORS = (
    LLMAuthError,
    LLMRateLimitError,
    LLMTimeoutError,
    LLMConnectionError,
    LLMNotImplementedError,
    LLMConfigError,
    LLMAPIError,
)


class GuardSpec(BaseModel):
    guard: str
    params: dict[str, Any] = Field(default_factory=dict)
    on_error: Literal["allow", "block"] | None = None


class PolicyConfig(BaseModel):
    max_regenerations: int = Field(default=1, ge=0, le=3)


class GuardProfile(BaseModel):
    name: str
    description: str = ""
    input: list[GuardSpec] = Field(default_factory=list)
    output: list[GuardSpec] = Field(default_factory=list)
    policy: PolicyConfig = Field(default_factory=PolicyConfig)


def list_profiles(configs_dir: Path) -> list[str]:
    """사용 가능한 프로필 이름: "none" + configs/guards_<name>.yaml."""
    names = sorted(p.stem[len("guards_"):] for p in Path(configs_dir).glob("guards_*.yaml"))
    ordered = [n for n in ("default",) if n in names] + [n for n in names if n != "default"]
    return ["none"] + ordered


def load_profile(configs_dir: Path, name: str) -> GuardProfile | None:
    """프로필 YAML 을 읽는다. "none" 이면 None (가드 미적용)."""
    if name == "none":
        return None
    path = Path(configs_dir) / f"guards_{name}.yaml"
    if not path.exists():
        raise InvalidRequestError(f"알 수 없는 가드 프로필입니다: {name}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    data.setdefault("name", name)
    return GuardProfile(**data)


@dataclass
class StageResult:
    stage: str
    results: list[GuardResult] = field(default_factory=list)
    decision: Literal["allow", "block", "revise"] = "allow"

    @property
    def blocking(self) -> GuardResult | None:
        return next((r for r in self.results if r.decision == "block"), None)

    @property
    def revisions(self) -> list[GuardResult]:
        return [r for r in self.results if r.decision == "revise"]


@dataclass
class _Entry:
    guard: Guard
    on_error: Literal["allow", "block"]


class GuardPipeline:
    def __init__(self, profile: GuardProfile, deps: GuardDeps, config: GuardsSection) -> None:
        self.profile = profile
        self.config = config
        self.input = self._build(profile.input, deps, "input")
        self.output = self._build(profile.output, deps, "output")

    def _build(self, specs: list[GuardSpec], deps: GuardDeps, stage: str) -> list[_Entry]:
        entries: list[_Entry] = []
        for spec in specs:
            try:
                cls = get_guard_class(spec.guard)
            except KeyError as exc:
                raise InvalidRequestError(
                    f"가드 프로필 '{self.profile.name}' 에 등록되지 않은 가드가 있습니다: {spec.guard}", detail=str(exc)
                ) from exc
            if cls.stage != stage:
                raise InvalidRequestError(
                    f"가드 '{spec.guard}' 는 {cls.stage} 단계용인데 프로필 '{self.profile.name}' 의 {stage} 에 들어 있습니다."
                )
            entries.append(_Entry(cls(deps, **spec.params), spec.on_error or self.config.default_on_error))
        if self.config.enforce_rule_first:
            entries.sort(key=lambda e: 0 if e.guard.kind == "rule" else 1)  # 안정 정렬: 같은 종류 안에서는 순서 유지
        return entries

    def describe(self) -> tuple[list[GuardInfo], list[GuardInfo]]:
        def info(entries: list[_Entry]) -> list[GuardInfo]:
            return [GuardInfo(name=e.guard.name, stage=e.guard.stage, kind=e.guard.kind) for e in entries]

        return info(self.input), info(self.output)

    def run_input(self, ctx: GuardContext) -> StageResult:
        return self._run("input", self.input, ctx)

    def run_output(self, ctx: GuardContext) -> StageResult:
        return self._run("output", self.output, ctx)

    def _run(self, stage: str, entries: list[_Entry], ctx: GuardContext) -> StageResult:
        out = StageResult(stage=stage)
        for entry in entries:
            guard = entry.guard
            started = time.perf_counter()
            try:
                res = guard.check(ctx)
            except _INFRA_ERRORS:
                raise
            except LLMRefusalError as err:
                # 판정 모델이 분류 자체를 거절 → 위험 신호로 보고 차단한다
                res = guard.result("block", "judge_refusal", f"판정 모델이 판정을 거절함 (category={err.category})", 1.0)
            except Exception as exc:
                log.warning("가드 오류 (guard error) %s: %s", guard.name, exc)
                res = guard.result(
                    entry.on_error, "guard_error", f"가드 실행 오류 → on_error={entry.on_error}: {type(exc).__name__}: {exc}"[:300],
                    1.0 if entry.on_error == "block" else 0.0,
                )
            res.name = guard.name
            res.stage = guard.stage
            res.latency_ms = int((time.perf_counter() - started) * 1000)
            res.attempt = ctx.attempt
            out.results.append(res)
            if res.decision == "block":
                out.decision = "block"
                break
            if res.decision == "revise":
                out.decision = "revise"
        return out
