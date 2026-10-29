from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import create_app
from app.service import HandleOptions, TutorService
from app.tutor.agent import parse_citation_indices


def test_kb_answer_has_citations_matching_chunks(make_settings):
    svc = TutorService(make_settings(GROUNDING_MODE="kb"))
    r = svc.handle([{"role": "user", "content": "리스트와 튜플의 차이가 뭐야?"}])
    assert r.grounding == "kb" and r.grounded is True and r.status == "answered"
    assert r.citations, "kb 모드 답변에는 출처가 있어야 한다"
    retrieved_ids = {c.chunk_id for c in r.trace.retrieved if c.passed_threshold}
    for c in r.citations:
        assert c.chunk_id in retrieved_ids
        assert c.doc_title and c.unit and c.section and c.text
        assert f"[{c.index}]" in r.answer
    assert r.meta.retrieved_count >= 1


def test_kb_no_evidence_uses_template_without_llm(make_settings):
    svc = TutorService(make_settings(GROUNDING_MODE="kb"))
    r = svc.handle([{"role": "user", "content": "오늘 서울 날씨 어때?"}])
    assert r.status == "no_evidence" and r.grounded is False
    assert "찾을 수 없" in r.answer and r.citations == []
    assert r.usage.llm_calls == 0  # 근거가 없으면 생성 LLM 을 부르지 않는다
    assert "리스트와 튜플" in r.answer  # 자료 범위의 관련 주제를 제안한다


def test_kb_followup_rewrites_query_before_retrieval(make_settings):
    svc = TutorService(make_settings(GROUNDING_MODE="kb"))
    q1 = "리스트와 튜플의 차이가 뭐야?"
    r1 = svc.handle([{"role": "user", "content": q1}])
    r2 = svc.handle(
        [{"role": "user", "content": q1}, {"role": "assistant", "content": r1.answer}, {"role": "user", "content": "그럼 그건 왜 써?"}]
    )
    assert r2.meta.rewrite_applied and r2.meta.rewritten_query
    assert r2.trace.retrieval_query == r2.meta.rewritten_query
    assert [c.task for c in r2.usage.calls] == ["query_rewrite", "tutor_answer"]
    assert r2.status == "answered"


def test_general_mode_skips_retrieval(settings):
    svc = TutorService(settings)
    r = svc.handle([{"role": "user", "content": "리스트가 뭐야?"}], HandleOptions(grounding_mode="general"))
    assert r.trace.retrieved == [] and r.citations == [] and r.grounded is None


def test_hint_mode_in_kb(make_settings):
    svc = TutorService(make_settings(GROUNDING_MODE="kb"))
    r = svc.handle([{"role": "user", "content": "for문으로 1부터 10까지 합 구하는 코드 짜줘"}], HandleOptions(answer_mode="hint"))
    assert "힌트 1" in r.answer and r.meta.question_type == "exercise"


def test_citation_parser_ignores_code_indexing():
    text = "리스트는 0부터 센다 [1]. 예: `a[2]` 와 아래 코드\n```python\nprint(a[3])\n```\n그리고 b[4] 도 인덱싱 [2][3]"
    assert parse_citation_indices(text) == [1, 2, 3]


def test_config_reports_kb_and_retrieval_status(make_settings):
    c = TestClient(create_app(make_settings(GROUNDING_MODE="kb")))
    body = c.get("/api/config").json()
    assert body["kb"]["status"] == "enabled" and body["kb"]["num_chunks"] > 0
    assert body["grounding_mode"] == "kb" and "kb" in body["grounding_modes"]
    assert body["retrieval"]["bm25"]["status"] == "enabled"
    assert body["retrieval"]["dense"]["status"] == "disabled"
    assert body["retrieval"]["hybrid"]["status"] == "disabled"
    assert len(body["kb"]["units"]) == 4


def test_kb_unavailable_is_reported_honestly(make_settings, tmp_path):
    s = make_settings(KB_DIR=str(tmp_path / "missing"), GROUNDING_MODE="general")
    svc = TutorService(s)
    assert svc.kb_available is False
    c = TestClient(create_app(s, service=svc))
    body = c.get("/api/config").json()
    assert body["kb"]["status"] == "disabled" and body["grounding_modes"] == ["general"]
    resp = c.post("/api/chat", json={"messages": [{"role": "user", "content": "q"}], "grounding_mode": "kb"})
    assert resp.status_code == 503 and resp.json()["error"]["code"] == "kb_unavailable"


def test_api_kb_chat_returns_citations(make_settings):
    c = TestClient(create_app(make_settings()))
    resp = c.post("/api/chat", json={"messages": [{"role": "user", "content": "print와 return의 차이"}], "grounding_mode": "kb"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["grounding"] == "kb" and body["citations"]
    assert {"index", "doc_title", "unit", "section", "chunk_id", "text"} <= set(body["citations"][0])
