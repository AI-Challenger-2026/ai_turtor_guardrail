# 안전벨트를 채운 AI 튜터

중학생이 쓰는 온라인 학습 플랫폼의 AI 튜터 서버입니다. 수업 자료(`kb/`)에 근거해 답하는 `/tutor`와, 학습자 입력과 후보 답변을 검사하는 `/guard`를 제공합니다.

## 엔드포인트

- `POST /tutor`: 대화(`messages`)를 받아 `answer`, `action`, `citations`를 돌려줍니다.
- `POST /guard`: 대화와 선택 항목인 후보 답변(`candidate_answer`)을 받아, 입력 판정(`input`)과 후보 답변 판정(`output`)을 돌려줍니다.
- 두 엔드포인트 모두 `Authorization: Bearer <TEAM_TOKEN>`을 확인합니다.
  - 인증 실패만 401로 답합니다.
  - 그 밖에는 깨진 요청, 내부 오류, 시간 초과까지 모두 HTTP 200과 규격 JSON으로 답합니다.

## 처리 흐름

**입력 분류** (`app/classify.py`, `app/classify_feat.py`, `app/normalize.py`)

- 모델은 범주를 직접 고르지 않고 예/아니오 신호 9개에 답합니다. 신호는 자기 개인정보, 정서적 고통, 탈옥, 해치기, 혐오, 제출용 대필, 정답만 요구, 수업 무관, 뜻 없는 입력입니다.
- 코드의 우선순위 규칙이 신호를 범주로 바꿉니다.
- 한국어 프롬프트 3표와 영어 프롬프트 2표를 동시에 보내고, 신호마다 과반으로 정합니다. 모든 신호의 과반이 정해지면 남은 표는 취소합니다.
- 분류가 실패하면 정상 입력으로 봅니다.
- 분류 전에 코드 규칙을 적용합니다.
  - 글자와 숫자가 하나도 없는 입력은 모델을 부르지 않고 `low_signal_input`으로 정합니다.
  - 글자마다 띄어 쓴 영문 낱말은 붙입니다.
  - 권한을 주장하거나 거절하지 말라고 압박하는 말은 판정기 입력에서 지웁니다.

**검색** (`app/kb.py`, `app/kb_en.py`)

- 섹션은 `kb/`의 문서 하나(딸린 표 CSV 포함)이고, 검색 단위는 `## ` 소제목 조각입니다.
- 한글은 글자 2-gram, 영문과 숫자는 낱말 단위로 BM25 점수를 매깁니다.
- 영어 질문은 조각마다 영어 번역을 더한 색인으로 찾습니다. 번역은 서버 시작 때 gpt-5-nano로 만들고 메모리에만 둡니다.

**/tutor** (`server.py`, `app/answer.py`, `app/excerpt.py`, `app/policy.py`)

- 분류와 동시에 검색 1위 섹션으로 답변 생성을 시작합니다.
- 응대 방식은 범주로 정합니다(아래 표). 거절, 전환, 안내, 되묻기는 고정 문구로 답하고 생성 중인 답변은 버립니다.
- 정답만 요구하는 질문은 힌트 모드로, 제출용 대필 요청은 대필 거절과 힌트 모드로 다시 생성합니다.
- 수업과 무관하다고 분류됐어도 자료에 답이 있으면 정상 질문으로 봅니다.
- 답을 자료에서 찾은 경우(일반 답변과 힌트)
  - 검색 1위 섹션을 인용합니다.
  - 그 섹션의 원문 문장을 답변 끝에 덧붙입니다.
- 일반 질문의 답을 자료에서 찾지 못한 경우
  - 지어내지 않고 자료에 없다고 답합니다(`abstain`).
  - 인용은 붙이지 않습니다.
- 답변 생성이 실패하면 검색 1위 섹션의 문장으로 답합니다. 대필 요청이면 고정 문구로 답합니다.

**/guard** (`server.py`, `app/verify.py`)

- 입력 분류, 후보 답변 검증, 자료 확인을 동시에 실행합니다.
- 후보 답변 검증
  - 과목 고유의 사실 주장을 자료와 대조해, 자료와 다르거나 자료에 없는 구간을 후보 답변에서 그대로 뽑습니다.
  - 3회 동시 판정의 다수결로 정합니다.
- 막아야 할 요청에 대한 후보 답변은 `safe: false`입니다.
- 정상 질문인데 자료에 답이 없으면 `out_of_material`입니다.

| 범주 | /guard 판정 | /tutor 응대 |
|---|---|---|
| normal, pii_disclosure | allow | answer |
| out_of_material | allow | abstain |
| answer_seeking | allow | scaffold |
| academic_dishonesty | block | scaffold |
| jailbreak_injection | block | refuse |
| harmful_content | block | refuse(해치기), redirect(혐오) |
| off_topic | block | redirect |
| emotional_distress | allow | escalate |
| low_signal_input | allow | clarify |

## 대회 규칙 준수

- 외부 호출은 대회가 제공하는 gpt-5-nano(`OPENAI_*`)뿐이고, 임베딩은 쓰지 않습니다.
- 다른 LLM, 웹 검색, 외부 데이터, 미리 학습된 모델 파일은 없습니다.
- 서버가 읽는 데이터는 `kb/`뿐이고 `public/`은 읽지 않습니다.
- 질문과 답을 짝지은 표, 캐시, 요청 사이 상태, 파일 쓰기가 없습니다.
- 로그(`app/logs.py`)
  - 남기는 것: 시각, 엔드포인트, 상태 코드, 지연, 입력 길이, 단계 이름, 오류 클래스 이름.
  - 남기지 않는 것: 요청·답변 본문, 모델 출력, 예외 메시지. uvicorn 접근 로그도 끕니다.
- 요청마다 23초 예산 안에서 처리하고, 응답 값은 Pydantic `Literal` 모델로 검증합니다.
- `tests/test_compliance.py`가 위 항목을 정적으로 점검합니다.

## 실행

필요 패키지는 `fastapi`, `uvicorn`, `openai`, `pydantic`입니다.

```bash
# environment: OPENAI_BASE_URL, OPENAI_MODEL, OPENAI_API_KEY, TEAM_TOKEN
python3 -m uvicorn server:app --host 0.0.0.0 --port 8000
```

서버를 시작하면 영어 색인을 만드느라 30초쯤 걸립니다. 로그에 `event=index`와 `Application startup complete`가 나온 뒤에 요청을 보냅니다.

## 테스트

```bash
# offline tests with a fake model client (no network)
python3 -m unittest discover -s tests -t . -v
# live smoke test against a running server
python3 tests/smoke_live.py --base http://localhost:8001
# local grader
python3 harness/score_local.py --tutor-url http://localhost:8001/tutor \
  --guard-url http://localhost:8001/guard --token "$TEAM_TOKEN"
```

## 디렉터리

| 경로 | 내용 |
|---|---|
| `server.py` | 엔드포인트와 요청 흐름 |
| `app/` | 분류, 검색, 답변 생성, 후보 답변 검증, 입력 규칙, 로그, 응답 모델 |
| `kb/` | 수업 자료(금융 리터러시, AI 프롬프트 활용, 파이썬 기초) |
| `public/` | 공개 문항(평가용) |
| `tests/` | 오프라인 테스트와 실서버 스모크 테스트 |
| `eval/` | 공개 문항으로 설정을 비교한 실험 스크립트와 평가용 분할(서버는 쓰지 않음) |
| `grader/`, `harness/` | 대회에서 받은 로컬 채점기 |
| `SPEC.md` | 대회 규격 |
