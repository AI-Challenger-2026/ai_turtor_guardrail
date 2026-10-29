from __future__ import annotations

import pytest

from app.kb.index import KBIndex
from app.llm.base import usage_scope
from app.llm.mock_provider import MockProvider
from app.prompts import PromptStore
from app.retrieval.bm25 import BM25Index
from app.retrieval.hybrid import Retriever
from app.retrieval.query_rewriter import QueryRewriter
from app.retrieval.tokenizer import SimpleTokenizer, get_tokenizer
from app.settings import BACKEND_DIR, DenseSection, LLMParams, RetrievalSection, RewriteSection

SAMPLE_KB = BACKEND_DIR / "data" / "sample" / "kb"


@pytest.fixture(scope="module")
def index() -> KBIndex:
    return KBIndex.build(SAMPLE_KB, 600, 100, base_dir=BACKEND_DIR)


@pytest.mark.parametrize("tokenizer", ["auto", "simple"])
@pytest.mark.parametrize(
    "query,unit",
    [
        ("리스트와 튜플의 차이가 뭐야?", "2단원. 리스트와 튜플"),
        ("print랑 return 차이", "4단원. 함수"),
        ("변수 이름 규칙 알려줘", "1단원. 변수와 자료형"),
        ("while 무한 루프", "3단원. 조건문과 반복문"),
    ],
)
def test_bm25_finds_expected_unit(index, tokenizer, query, unit):
    r = Retriever(index, RetrievalSection(tokenizer=tokenizer))
    res = r.search(query)
    assert res.passed, f"{query} 는 근거를 찾아야 한다"
    assert unit in [c.unit for c in res.passed]


@pytest.mark.parametrize("query", ["오늘 날씨 어때?", "주식 추천해줘", "대통령 선거 누가 이길까"])
def test_out_of_scope_has_no_evidence(index, query):
    r = Retriever(index, RetrievalSection(tokenizer="auto"))
    assert r.search(query).no_evidence


def test_simple_tokenizer_bigrams():
    toks = SimpleTokenizer().tokenize("리스트에서 append 사용")
    assert "리스트에서" in toks and "리스" in toks and "append" in toks


def test_tokenizer_fallback_to_simple_when_requested():
    assert get_tokenizer("simple").name == "simple"


def test_bm25_normalized_scores_in_range():
    bm = BM25Index([["리스트", "튜플"], ["함수", "return"], ["리스트", "리스트", "append"]])
    raw, norm = bm.normalized_scores(["리스트"])
    assert raw[1] == 0 and norm[1] == 0
    assert all(0.0 <= x <= 1.0 for x in norm)
    assert norm[2] >= norm[0] > 0


def test_hybrid_rrf_with_fake_encoder(index):
    # 가짜 임베더: "튜플" 포함 여부로 2차원 벡터 생성 — 무거운 모델 없이 RRF 결합을 검증한다
    def encoder(texts):
        return [[1.0 if "튜플" in t else 0.0, 1.0] for t in texts]

    cfg = RetrievalSection(tokenizer="auto", dense=DenseSection(enabled=True, model="fake-model", min_score=0.99, cache=False))
    r = Retriever(index, cfg, dense_encoder=encoder)
    assert r.dense_status.status == "enabled" and r.hybrid_status().status == "enabled"
    res = r.search("튜플은 언제 써?")
    assert res.mode == "hybrid" and res.passed
    assert res.candidates[0].dense_score is not None and res.candidates[0].bm25_score is not None
    assert "2단원. 리스트와 튜플" == res.candidates[0].unit


def test_dense_enabled_without_library_reports_disabled(index, monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("sentence_transformers"):
            raise ImportError("no module")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    cfg = RetrievalSection(dense=DenseSection(enabled=True))
    r = Retriever(index, cfg)
    assert r.dense is None and r.dense_status.status == "disabled"
    assert "sentence-transformers" in (r.dense_status.detail or "")


def _rewriter(policy="auto"):
    return QueryRewriter(
        MockProvider("mock-judge"), PromptStore(BACKEND_DIR / "prompts"), LLMParams(temperature=0, max_tokens=256),
        RewriteSection(policy=policy),
    )


def test_followup_triggers_llm_rewrite():
    history = [{"role": "user", "content": "리스트와 튜플의 차이가 뭐야?"}, {"role": "assistant", "content": "..."}]
    with usage_scope() as calls:
        res = _rewriter().rewrite("그럼 튜플은 왜 써?", history)
    assert res.applied and res.used_llm
    assert [c["task"] for c in calls] == ["query_rewrite"]
    assert "리스트와 튜플의 차이" in res.rewritten


def test_standalone_question_skips_rewrite():
    history = [{"role": "user", "content": "변수가 뭐야?"}, {"role": "assistant", "content": "..."}]
    with usage_scope() as calls:
        res = _rewriter().rewrite("파이썬 리스트에서 원소를 삭제하는 방법을 자세히 알려줘", history)
    assert not res.applied and calls == []


def test_first_turn_and_policy_never_skip_rewrite():
    assert not _rewriter().rewrite("그럼?", []).applied
    history = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]
    assert not _rewriter("never").rewrite("그럼 2번은?", history).applied
