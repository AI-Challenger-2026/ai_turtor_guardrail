from __future__ import annotations

import json

import pytest

from app.errors import InvalidRequestError
from app.service import HandleOptions, TutorService, truncate_history
from app.tutor.modes import classify_question, estimate_learner_level


def test_general_mode_answer_is_mock(settings):
    svc = TutorService(settings)
    r = svc.handle([{"role": "user", "content": "변수가 뭐야?"}])
    assert r.answer.startswith("[MOCK]")
    assert r.provider == "mock" and r.grounding == "general"
    assert r.citations == [] and r.guard_trace is None
    assert r.usage.llm_calls == 1 and r.usage.calls[0].provider == "mock"


def test_followup_receives_previous_context(settings):
    svc = TutorService(settings)
    first_q = "리스트랑 튜플 차이가 뭐야?"
    r1 = svc.handle([{"role": "user", "content": first_q}])
    r2 = svc.handle(
        [
            {"role": "user", "content": first_q},
            {"role": "assistant", "content": r1.answer},
            {"role": "user", "content": "그럼 튜플은 언제 써?"},
        ]
    )
    assert first_q[:20] in r2.answer  # Mock 은 직전 질문을 답변에 인용해 문맥 전달을 보여 준다
    assert r2.trace.history_messages == 2


def test_new_conversation_has_no_previous_context(settings):
    svc = TutorService(settings)
    r = svc.handle([{"role": "user", "content": "그럼 튜플은 언제 써?"}])
    assert "첫 질문" in r.answer and r.trace.history_messages == 0


def test_truncate_history_keeps_recent_turns():
    msgs = []
    for i in range(5):
        msgs += [{"role": "user", "content": f"q{i}"}, {"role": "assistant", "content": f"a{i}"}]
    msgs.append({"role": "user", "content": "q5"})
    kept, dropped = truncate_history(msgs, 2)
    assert [m["content"] for m in kept] == ["q4", "a4", "q5"]
    assert dropped == 8 and kept[0]["role"] == "user"


def test_max_turns_applied_in_handle(make_settings):
    svc = TutorService(make_settings(MAX_TURNS="1"))
    r = svc.handle(
        [
            {"role": "user", "content": "첫 질문"},
            {"role": "assistant", "content": "첫 답"},
            {"role": "user", "content": "두번째 질문"},
        ]
    )
    assert r.trace.truncated_messages == 2 and r.trace.history_messages == 0


def test_message_too_long_rejected(make_settings):
    svc = TutorService(make_settings(MAX_MESSAGE_CHARS="10"))
    with pytest.raises(InvalidRequestError) as info:
        svc.handle([{"role": "user", "content": "가" * 11}])
    assert info.value.code == "message_too_long"


def test_last_message_must_be_user(settings):
    svc = TutorService(settings)
    with pytest.raises(InvalidRequestError):
        svc.handle([{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}])


def test_unknown_profile_rejected(settings):
    svc = TutorService(settings)
    with pytest.raises(InvalidRequestError):
        svc.handle([{"role": "user", "content": "q"}], HandleOptions(guard_profile="no_such_profile"))


def test_trace_log_written_without_secrets(make_settings, tmp_path):
    secret = "sk-ant-api03-SECRETSECRETSECRET"
    s = make_settings(ANTHROPIC_API_KEY=secret, LOG_DIR=str(tmp_path / "logs"))
    svc = TutorService(s)
    r = svc.handle([{"role": "user", "content": f"내 키는 {secret} 야. 변수가 뭐야?"}])
    files = list((tmp_path / "logs").glob("turns_*.jsonl"))
    assert files, "턴 로그 파일이 생성되어야 한다"
    content = files[0].read_text(encoding="utf-8")
    assert secret not in content
    record = json.loads(content.strip().splitlines()[-1])
    assert record["trace_id"] == r.trace_id
    # Mock 답변이 질문(키 포함)을 그대로 인용하므로, 로그 속 답변에서도 키가 마스킹되어야 한다
    assert record["final_answer"] == r.answer.replace(secret, "***REDACTED***")
    assert record["llm_calls"][0]["task"] == "tutor_answer"


def test_answer_mode_reaches_prompt(settings):
    svc = TutorService(settings)
    r = svc.handle([{"role": "user", "content": "for문 설명해줘"}], HandleOptions(answer_mode="hint"))
    assert r.answer_mode == "hint" and "응답 모드: hint" in r.answer


def test_question_type_rules():
    hist = [{"role": "user", "content": "리스트가 뭐야?"}, {"role": "assistant", "content": "..."}]
    assert classify_question("리스트가 뭐야?") == "concept"
    assert classify_question("짝수만 출력하는 코드 짜줘") == "exercise"
    assert classify_question("아직도 헷갈려요 ㅠ") == "confusion"
    assert classify_question("그럼 2번은요?", hist) == "followup"
    assert classify_question("그럼 2번은요?", []) == "concept"


def test_learner_level_estimation():
    assert estimate_learner_level([{"role": "user", "content": "코딩 처음인데 변수가 뭐예요?"}]) == "beginner"
    assert estimate_learner_level(
        [{"role": "user", "content": "리스트 컴프리헨션이랑 제너레이터 표현식의 메모리 차이가 궁금해요"}]
    ) == "intermediate"
