"""튜터 처리 단일 진입점 (single entry point): TutorService.handle(messages, options).

웹 API, CLI, 평가 스크립트가 모두 이 함수를 호출한다. 화면 결과와 평가 결과가 같은 코드 경로에서 나온다.

한 턴의 처리 순서:
  1) 입력 정규화·검증, 오래된 턴 잘라내기 (MAX_TURNS)
  2) 질문 유형 분류, 학습자 수준 추정 (rule-based)
  3) 멀티턴 질의 재작성 (kb 모드이거나 입력 가드가 있을 때, 후속 질문만)
  4) 입력 가드 (규칙 → LLM). block 이면 교육적 거절 메시지로 끝 (이후 단계 생략)
  5) [kb] 검색 (BM25 / 선택적 하이브리드) → 임계값 미달이면 "근거 없음" 안내
  6) 답변 생성 → 출력 가드 (출처 → 근거 충실도 → 안전). revise 면 1회 재생성, 그래도 실패하면 fallback
  7) 추적 로그 기록 (logs/turns_*.jsonl): 입력 판정, 재작성 질의, 검색 청크, 원본 답변, 출력 판정, 최종 답변
"""

from __future__ import annotations

import re
import time
import uuid
from datetime import datetime
from typing import Any, Mapping, Sequence

from pydantic import BaseModel

from app.errors import AppError, InvalidRequestError, LLMRefusalError
from app.guardrails.base import GuardContext, GuardDeps
from app.guardrails.pipeline import GuardPipeline, StageResult, list_profiles, load_profile
from app.guardrails.policy import PolicyEngine
from app.kb.index import KBIndex
from app.llm.base import usage_scope
from app.llm.factory import build_provider, resolve_model
from app.logging_utils import TurnLogger, get_logger, setup_logging
from app.prompts import PromptStore, render_template
from app.retrieval.hybrid import RetrievalResult, Retriever
from app.retrieval.query_rewriter import QueryRewriter, RewriteResult
from app.schemas import (
    ChatMessage,
    ConfigResponse,
    GuardsInfo,
    GuardTrace,
    KBInfo,
    LimitsInfo,
    ModuleInfo,
    ProviderInfo,
    RetrievalInfo,
    RetrievedChunkInfo,
    RewriteInfo,
    TurnMeta,
    TurnResult,
    TurnTrace,
    Usage,
)
from app.settings import ANSWER_MODES, GROUNDING_MODES, AnswerMode, GroundingMode, Settings, load_settings
from app.tutor.agent import AgentAnswer, TutorAgent
from app.tutor.modes import ModeInstructions, classify_question, estimate_learner_level

log = get_logger("service")


class HandleOptions(BaseModel):
    """요청별 옵션. None 이면 환경변수 기본값을 쓴다."""

    answer_mode: AnswerMode | None = None
    guard_profile: str | None = None
    grounding_mode: GroundingMode | None = None
    source: str = "api"
    log: bool = True


class ResolvedOptions(BaseModel):
    answer_mode: AnswerMode
    guard_profile: str
    grounding_mode: GroundingMode
    source: str
    log: bool


def truncate_history(messages: list[dict[str, str]], max_turns: int) -> tuple[list[dict[str, str]], int]:
    """최근 max_turns 개의 사용자 턴(user + 이어지는 assistant)만 남긴다. (kept, dropped_count)"""
    if max_turns <= 0:
        return messages[-1:], len(messages) - 1
    user_idx = [i for i, m in enumerate(messages) if m["role"] == "user"]
    if len(user_idx) <= max_turns:
        return messages, 0
    start = user_idx[-max_turns]
    return messages[start:], start


class TutorService:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or load_settings()
        setup_logging(self.settings.log_level)
        self.prompts = PromptStore(self.settings.prompts_dir)
        self.generator = build_provider(self.settings, "generator")
        self.judge = build_provider(self.settings, "judge")
        self.modes = ModeInstructions(self.prompts)
        self.agent = TutorAgent(self.generator, self.prompts, self.settings.llm.generator)
        self.rewriter = QueryRewriter(self.judge, self.prompts, self.settings.llm.rewriter, self.settings.query_rewrite)
        self.turn_logger = TurnLogger(self.settings.log_dir, self.settings.secret_values(), self.settings.log_turns)
        self.kb_index: KBIndex | None = None
        self.retriever: Retriever | None = None
        self.kb_error: str | None = None
        self.kb_stale = False
        self._load_kb()
        self.guard_deps = GuardDeps(
            judge=self.judge,
            prompts=self.prompts,
            judge_params=self.settings.llm.judge,
            configs_dir=self.settings.configs_dir,
            history_turns=self.settings.guards.history_turns,
        )
        self._pipelines: dict[str, GuardPipeline | None] = {}
        self._check_default_profile()

    # ------------------------------------------------------------------
    # 가드 프로필 (guard profiles)
    # ------------------------------------------------------------------
    def _check_default_profile(self) -> None:
        profile = self.settings.guard_profile
        try:
            if profile not in self.available_profiles():
                raise InvalidRequestError(f"GUARD_PROFILE 에 지정한 프로필이 없습니다: {profile}")
            self.get_pipeline(profile)
        except AppError as err:  # 서버는 띄우되, 해당 프로필 요청은 오류를 돌려준다
            log.error("기본 가드 프로필을 불러오지 못함 (default guard profile error): %s", err.user_message)

    def get_pipeline(self, profile: str) -> GuardPipeline | None:
        """프로필별 파이프라인 (캐시). "none" 이면 None."""
        if profile not in self._pipelines:
            spec = load_profile(self.settings.configs_dir, profile)
            self._pipelines[profile] = None if spec is None else GuardPipeline(spec, self.guard_deps, self.settings.guards)
        return self._pipelines[profile]

    def scope_description(self, grounding: str) -> str:
        """입력 분류기에 알려 줄 학습 범위 — 교육 자료 단원명에서 자동으로 만든다."""
        if grounding == "kb" and self.kb_index:
            titles = sorted({d["doc_title"] for d in self.kb_index.documents})
            return (
                f"교육 자료 {', '.join(titles)} 의 단원: {self.topics_text()}. "
                "이 단원들과 관련된 질문, 그리고 그 밖의 프로그래밍·파이썬 학습 질문이 학습 범위다."
            )
        return "프로그래밍(특히 파이썬) 학습과 관련된 질문이 학습 범위다."

    # ------------------------------------------------------------------
    # 교육 자료 인덱스 (KB index)
    # ------------------------------------------------------------------
    def _load_kb(self) -> None:
        s = self.settings
        path = s.kb_index_path
        try:
            index: KBIndex | None = KBIndex.load(path) if path.exists() else None
            stale = index is not None and index.is_stale(s.kb_dir)
            if index is None or stale:
                if not s.kb.auto_build:
                    if index is None:
                        raise FileNotFoundError(f"인덱스 파일이 없습니다. python -m app.kb.build 를 실행하세요: {path}")
                    self.kb_stale = True
                else:
                    log.info("교육 자료 인덱스 %s (KB index build): %s", "갱신" if stale else "생성", path)
                    index = KBIndex.build(s.kb_dir, s.kb.chunk_size, s.kb.chunk_overlap, base_dir=s.backend_dir)
                    index.save(path)
            if index is None or not index.chunks:
                raise ValueError(f"교육 자료 청크가 0개입니다: {s.kb_dir}")
            self.kb_index = index
            self.retriever = Retriever(index, s.retrieval, dense_cache_path=path.with_suffix(".dense.json"))
        except Exception as exc:  # 인덱스 문제로 서버가 죽지 않게 한다 — kb 모드만 비활성화
            self.kb_error = f"{type(exc).__name__}: {exc}"
            log.warning("교육 자료 인덱스를 사용할 수 없음 (KB disabled): %s", self.kb_error)

    @property
    def kb_available(self) -> bool:
        return self.retriever is not None

    # ------------------------------------------------------------------
    # 옵션·입력 처리 (options & input)
    # ------------------------------------------------------------------
    def available_profiles(self) -> list[str]:
        return list_profiles(self.settings.configs_dir)

    def resolve_options(self, options: HandleOptions | Mapping[str, Any] | None) -> ResolvedOptions:
        if options is None:
            opts = HandleOptions()
        elif isinstance(options, HandleOptions):
            opts = options
        else:
            opts = HandleOptions(**dict(options))
        answer_mode = opts.answer_mode or self.settings.answer_mode
        grounding = opts.grounding_mode or self.settings.grounding_mode
        profile = (opts.guard_profile or self.settings.guard_profile or "none").strip()
        if answer_mode not in ANSWER_MODES:
            raise InvalidRequestError(f"알 수 없는 응답 모드입니다: {answer_mode}")
        if grounding not in GROUNDING_MODES:
            raise InvalidRequestError(f"알 수 없는 근거 모드입니다: {grounding}")
        if grounding == "kb" and not self.kb_available:
            raise InvalidRequestError(
                "교육 자료 인덱스를 사용할 수 없어 교육 자료 근거 모드를 쓸 수 없습니다. "
                "서버 로그와 python -m app.kb.build 를 확인해 주세요.",
                code="kb_unavailable",
                http_status=503,
            )
        if profile not in self.available_profiles():
            raise InvalidRequestError(f"알 수 없는 가드 프로필입니다: {profile}")
        return ResolvedOptions(
            answer_mode=answer_mode, guard_profile=profile, grounding_mode=grounding, source=opts.source, log=opts.log
        )

    def normalize_messages(self, messages: Sequence[ChatMessage | Mapping[str, Any]]) -> list[dict[str, str]]:
        out: list[dict[str, str]] = []
        for m in messages:
            if isinstance(m, ChatMessage):
                role, content = m.role, m.content
            else:
                role, content = str(m.get("role", "")), str(m.get("content", ""))
            if role not in ("user", "assistant"):
                raise InvalidRequestError(f"메시지 role 은 user 또는 assistant 여야 합니다: {role!r}")
            content = content.strip()
            if not content:
                raise InvalidRequestError("빈 메시지는 보낼 수 없습니다.")
            if role == "user" and len(content) > self.settings.max_message_chars:
                raise InvalidRequestError(
                    f"메시지가 너무 깁니다. {self.settings.max_message_chars}자 이하로 줄여 주세요.",
                    code="message_too_long",
                )
            out.append({"role": role, "content": content})
        if not out:
            raise InvalidRequestError("메시지가 비어 있습니다.")
        if out[-1]["role"] != "user":
            raise InvalidRequestError("마지막 메시지는 사용자(user) 질문이어야 합니다.")
        return out

    def topics_text(self) -> str:
        """거절·근거 없음 안내에 넣을 학습 주제 목록 (교육 자료 단원명에서 만든다)."""
        if not self.kb_index:
            return "파이썬 기초 개념"
        names = [re.sub(r"^\s*\d+\s*단원[.:\s]*", "", u).strip() or u for u in self.kb_index.units()]
        return ", ".join(f"'{n}'" for n in names[:8])

    # ------------------------------------------------------------------
    # 단일 진입점 (single entry point)
    # ------------------------------------------------------------------
    def handle(
        self,
        messages: Sequence[ChatMessage | Mapping[str, Any]],
        options: HandleOptions | Mapping[str, Any] | None = None,
    ) -> TurnResult:
        trace_id = uuid.uuid4().hex[:12]
        started = time.perf_counter()
        opts = self.resolve_options(options)
        msgs = self.normalize_messages(messages)
        kept, dropped = truncate_history(msgs, self.settings.max_turns)
        question = kept[-1]["content"]
        history = kept[:-1]

        trace = TurnTrace(
            trace_id=trace_id,
            timestamp=datetime.now().isoformat(timespec="seconds"),
            source=opts.source,
            question=question,
            history_messages=len(history),
            truncated_messages=dropped,
            answer_mode=opts.answer_mode,
            grounding_mode=opts.grounding_mode,
            guard_profile=opts.guard_profile,
        )
        if dropped:
            trace.notes.append(f"MAX_TURNS={self.settings.max_turns} 초과로 오래된 메시지 {dropped}개를 제외함")

        with usage_scope() as calls:
            try:
                result = self._run_turn(question, history, opts, trace)
            except AppError as err:
                trace.error = f"{err.code}: {err.detail or err.user_message}"
                trace.status = "error"
                setattr(err, "trace_id", trace_id)
                self._write_log(trace, calls, opts, latency_ms=int((time.perf_counter() - started) * 1000))
                raise
            except Exception as exc:  # 예상치 못한 오류도 로그를 남기고 다시 던진다
                trace.error = f"internal: {type(exc).__name__}: {exc}"
                trace.status = "error"
                self._write_log(trace, calls, opts, latency_ms=int((time.perf_counter() - started) * 1000))
                raise

        latency_ms = int((time.perf_counter() - started) * 1000)
        result.latency_ms = latency_ms
        result.usage = Usage.from_calls(calls)
        result.trace.final_answer = result.answer
        result.trace.status = result.status
        self._write_log(result.trace, calls, opts, latency_ms=latency_ms)
        return result

    def _run_turn(
        self, question: str, history: list[dict[str, str]], opts: ResolvedOptions, trace: TurnTrace
    ) -> TurnResult:
        qtype = classify_question(question, history, self.settings.query_rewrite.max_short_chars)
        level = estimate_learner_level(history + [{"role": "user", "content": question}])
        trace.question_type = qtype
        trace.learner_level = level
        mode_ctx = self.modes.build(opts.answer_mode, qtype, level)
        gen_provider, gen_model = resolve_model(self.settings, "generator")
        meta = TurnMeta(question_type=qtype, learner_level=level)

        pipeline = self.get_pipeline(opts.guard_profile)
        guard_trace = GuardTrace(profile=opts.guard_profile) if pipeline else None
        trace.guard_trace = guard_trace
        policy = PolicyEngine(
            self.prompts, self.topics_text, pipeline.profile.policy.max_regenerations if pipeline else 0
        )

        def make(answer: str, **kwargs: Any) -> TurnResult:
            return TurnResult(
                answer=answer, grounding=opts.grounding_mode, provider=gen_provider, model=gen_model,
                answer_mode=opts.answer_mode, meta=meta, guard_trace=guard_trace, trace_id=trace.trace_id,
                trace=trace, **kwargs,
            )

        # ---- 1) 질의 재작성 (검색과 입력 가드에 넘길 독립 질문) ----
        rewritten = question
        if opts.grounding_mode == "kb" or (pipeline and pipeline.input):
            rewrite = self.rewriter.rewrite(question, history)
            self._record_rewrite(rewrite, trace, meta)
            rewritten = rewrite.rewritten

        ctx = GuardContext(
            question=question,
            rewritten_question=rewritten,
            history=history,
            grounding_mode=opts.grounding_mode,
            scope_description=self.scope_description(opts.grounding_mode),
        )

        # ---- 2) 입력 가드 ----
        if pipeline and guard_trace is not None:
            stage = pipeline.run_input(ctx)
            guard_trace.input = stage.results
            if stage.decision == "block":
                blocking = stage.blocking
                category = blocking.category if blocking else "default"
                guard_trace.final_action = "block"
                trace.notes.append(f"입력 가드 차단: {blocking.name if blocking else '-'} ({category}) → 이후 단계 생략")
                return make(policy.refusal(category), status="blocked", blocked=True, block_category=category)

        # ---- 3) 검색 (kb 모드) ----
        retrieval: RetrievalResult | None = None
        if opts.grounding_mode == "kb":
            assert self.retriever is not None
            retrieval = self.retriever.search(rewritten)
            self._record_retrieval(retrieval, trace, meta)
            if retrieval.no_evidence:
                trace.notes.append("검색 결과가 임계값 미만이라 근거 없음 안내 템플릿으로 답함 (LLM 생성·출력 가드 생략)")
                return make(self._template("no_evidence"), grounded=False, status="no_evidence")
        chunks = retrieval.passed if retrieval else []
        ctx.chunks = list(chunks)
        grounded = True if opts.grounding_mode == "kb" else None

        def generate(revision: str | None) -> AgentAnswer:
            return self.agent.answer(
                history=history, question=question, grounding=opts.grounding_mode, mode=mode_ctx,
                chunks=chunks, revision=revision,
            )

        try:
            # ---- 4) 답변 생성 (+ 출력 가드 / 재생성 정책) ----
            if not pipeline or not pipeline.output:
                answer = generate(None)
                trace.raw_answers.append(answer.text)
                return self._answered(make, answer, grounded, "answered", trace)

            def check(answer: AgentAnswer, attempt: int) -> StageResult:
                ctx.answer = answer.text
                ctx.cited_indices = answer.cited_indices
                ctx.attempt = attempt
                return pipeline.run_output(ctx)

            outcome = policy.run_output(generate, check)
        except LLMRefusalError as err:
            trace.notes.append(f"모델 자체 거절 (model refusal): category={err.category}")
            if guard_trace is not None:
                guard_trace.final_action = "block"
            return make(self._refusal_text("model_refusal"), status="blocked", blocked=True, block_category="model_refusal")

        assert guard_trace is not None
        trace.raw_answers.extend(outcome.raw_answers)
        guard_trace.output = outcome.results
        guard_trace.regenerations = outcome.regenerations
        guard_trace.final_action = outcome.final_action
        if outcome.answer is not None:
            return self._answered(make, outcome.answer, grounded, outcome.status, trace)
        trace.notes.append(f"출력 가드 결과 {outcome.final_action} → 안전한 fallback 답변으로 대체")
        return make(outcome.text, grounded=grounded, status="fallback")

    @staticmethod
    def _answered(make: Any, answer: AgentAnswer, grounded: bool | None, status: str, trace: TurnTrace) -> TurnResult:
        if answer.invalid_citations:
            trace.notes.append(f"검색 청크에 없는 인용 번호: {answer.invalid_citations}")
        result = make(answer.text, citations=answer.citations, grounded=grounded, status=status)
        if answer.llm_result:
            result.model = answer.llm_result.model
        return result

    # ------------------------------------------------------------------
    # 추적 기록 헬퍼 (trace helpers)
    # ------------------------------------------------------------------
    @staticmethod
    def _record_rewrite(rewrite: RewriteResult, trace: TurnTrace, meta: TurnMeta) -> None:
        trace.rewrite = RewriteInfo(
            original=rewrite.original, rewritten=rewrite.rewritten, applied=rewrite.applied, reason=rewrite.reason
        )
        meta.rewrite_applied = rewrite.applied
        meta.rewritten_query = rewrite.rewritten if rewrite.applied else None

    @staticmethod
    def _record_retrieval(retrieval: RetrievalResult, trace: TurnTrace, meta: TurnMeta) -> None:
        trace.retrieval_query = retrieval.query
        trace.retrieval_threshold = retrieval.threshold
        trace.retrieved = [
            RetrievedChunkInfo(
                rank=c.rank, chunk_id=c.chunk_id, doc_id=c.doc_id, doc_title=c.doc_title, unit=c.unit,
                section=c.section, score=round(c.score, 4),
                bm25_score=None if c.bm25_score is None else round(c.bm25_score, 4),
                dense_score=None if c.dense_score is None else round(c.dense_score, 4),
                passed_threshold=c.passed_threshold, text=c.text,
            )
            for c in retrieval.candidates
        ]
        meta.retrieved_count = len(retrieval.passed)

    # ------------------------------------------------------------------
    # 템플릿 (templates from prompts/refusal_templates.yaml)
    # ------------------------------------------------------------------
    def _templates(self) -> dict[str, Any]:
        return self.prompts.yaml("refusal_templates.yaml")

    def _refusal_text(self, category: str) -> str:
        templates = self._templates().get("refusals", {})
        template = templates.get(category) or templates.get("default", "")
        return render_template(template, topics=self.topics_text()).strip()

    def _template(self, key: str, group: str | None = None) -> str:
        data = self._templates()
        template = (data.get(group, {}) if group else data).get(key, "")
        return render_template(str(template), topics=self.topics_text()).strip()

    def _write_log(self, trace: TurnTrace, calls: list[dict[str, Any]], opts: ResolvedOptions, latency_ms: int) -> None:
        if not opts.log:
            return
        record = trace.model_dump()
        record["latency_ms"] = latency_ms
        record["llm_calls"] = calls
        try:
            self.turn_logger.write(record)
        except OSError as exc:  # 로그 실패가 응답을 막지 않게 한다
            log.warning("턴 로그 기록 실패 (turn log write failed): %s", exc)

    # ------------------------------------------------------------------
    # /api/config 스냅샷 (honest module status)
    # ------------------------------------------------------------------
    def _provider_info(self, role: str) -> ProviderInfo:
        provider = self.generator if role == "generator" else self.judge
        actual_provider, actual_model = resolve_model(self.settings, role)  # type: ignore[arg-type]
        cfg = self.settings.generator if role == "generator" else self.settings.judge
        configured_model = self.settings.openai_model if cfg.provider == "openai" else cfg.model
        status = "enabled"
        detail = None
        if provider.name == "unavailable":
            err = getattr(provider, "error", None)
            status = "not_implemented" if getattr(err, "code", "") == "provider_not_implemented" else "disabled"
            detail = getattr(err, "user_message", None)
        elif provider.name == "elice":
            status, detail = "not_implemented", "엘리스 ML API 연동 미구현"
        return ProviderInfo(
            provider=actual_provider, model=actual_model, configured_provider=cfg.provider,
            configured_model=configured_model, status=status, detail=detail,  # type: ignore[arg-type]
        )

    def _rewrite_status(self) -> ModuleInfo:
        policy = self.settings.query_rewrite.policy
        if policy == "never":
            return ModuleInfo(status="disabled", detail="query_rewrite.policy=never")
        return ModuleInfo(status="enabled", detail=f"policy={policy}, kb 모드 또는 입력 가드 사용 시 후속 질문에 적용 (judge 모델)")

    def _guards_info(self) -> GuardsInfo:
        profile = self.settings.guard_profile
        available = self.available_profiles()
        try:
            pipeline = self.get_pipeline(profile) if profile in available else None
        except AppError:
            pipeline = None
        if pipeline is None:
            return GuardsInfo(status="disabled", profile=profile, available_profiles=available)
        inputs, outputs = pipeline.describe()
        return GuardsInfo(status="enabled", profile=profile, available_profiles=available, input=inputs, output=outputs)

    def config_snapshot(self) -> ConfigResponse:
        s = self.settings
        if self.retriever is not None and self.kb_index is not None:
            kb = KBInfo(
                status="enabled",
                detail=f"{self.kb_index.source_dir} (chunk_size={self.kb_index.chunk_size}, overlap={self.kb_index.chunk_overlap})",
                num_documents=len(self.kb_index.documents),
                num_chunks=len(self.kb_index.chunks),
                units=self.kb_index.units(),
                built_at=self.kb_index.built_at,
                stale=self.kb_stale,
            )
            retrieval = RetrievalInfo(
                bm25=self.retriever.bm25_status(),
                dense=self.retriever.dense_status,
                hybrid=self.retriever.hybrid_status(),
                query_rewrite=self._rewrite_status(),
            )
            grounding_modes = list(GROUNDING_MODES)
        else:
            off = ModuleInfo(status="disabled", detail=f"교육 자료 인덱스 없음: {self.kb_error}")
            kb = KBInfo(status="disabled", detail=self.kb_error)
            retrieval = RetrievalInfo(bm25=off, dense=off, hybrid=off, query_rewrite=self._rewrite_status())
            grounding_modes = ["general"]
        return ConfigResponse(
            mock_mode=s.mock_mode,
            generator=self._provider_info("generator"),
            judge=self._provider_info("judge"),
            answer_mode=s.answer_mode,
            answer_modes=list(ANSWER_MODES),  # type: ignore[arg-type]
            grounding_mode=s.grounding_mode,
            grounding_modes=grounding_modes,  # type: ignore[arg-type]
            kb=kb,
            retrieval=retrieval,
            guards=self._guards_info(),
            limits=LimitsInfo(
                max_message_chars=s.max_message_chars, max_turns=s.max_turns, max_request_bytes=s.max_request_bytes
            ),
        )
