# 안전벨트를 채운 AI 튜터 — 작업 지침

엘리스 경진대회 제출 서버(`/tutor`, `/guard`). 규격은 SPEC.md가 기준이다.

## 1. 작업 규칙

- 코드를 바꾸기 전에 계획을 보여 주고 확인받는다.
- 단계마다 결과(점수, 지연, 바꾼 파일)를 보고한 뒤 커밋한다.
- 코드 주석은 영어로 쓴다.
- 브랜치: 작업은 `Chan` 브랜치에서 커밋하고 `origin/Chan`에 올린다. `main`에는 합치지 않는다.
- `.env`와 `submission.json`은 읽거나 출력하거나 커밋하지 않는다. 키가 필요하면 환경변수 이름만 쓴다.
  - 테스트 서버와 채점기를 띄울 때 `set -a; . ./.env; set +a`로 불러오는 것은 허용(내용 출력 금지).
- 포트 8000의 제출용 서버는 묻지 않고 끄거나 재시작하지 않는다. 테스트 서버는 8001.
- 새 패키지를 설치하기 전에 모델 파일을 내려받거나 외부를 호출하는지 확인하고 묻는다.
- 로컬 채점기 전체 실행, `--workers 64` 부하 시험, 설정 비교 실험은 실행 전에 묻는다.
  반복 실험은 평가용 분할의 일부로 한다(운영진 API 과다 호출 방지).
- 사용자가 제출해서 채점 중이라고 하면 서버, 코드, 로그를 건드리지 않는다. 서버 로그나 요청 내용을 열어 보지 않는다.
- 제출은 사용자가 한다.

## 2. 대회 규칙

- 서버의 외부 호출은 `OPENAI_*`(gpt-5-nano)와 `EMBEDDING_*`(text-embedding-3-small)뿐. 다른 LLM, 웹 검색, 외부 API 금지.
- 미리 학습된 모델 파일(임베딩, 리랭커, 분류 모델) 금지. 형태소 분석기, 토크나이저, BM25 라이브러리는 허용.
- 서버에 넣을 수 있는 데이터: `kb/`, `public/`과 그 가공물, gpt-5-nano로 만든 예시, 팀이 쓴 프롬프트와 규칙.
  Claude나 다른 LLM으로 만든 별도의 예시 묶음 파일·데이터셋 파일은 서버에 넣지 않는다. 프롬프트 안에 직접 쓴 예시는 허용.
- 로그는 시각, 엔드포인트, 상태 코드, 지연, 입력 길이, 오류 종류만 남긴다.
  - 요청·답변 본문, LLM 출력, 판정 이유는 남기지 않는다. 본문 `print` 금지.
  - 예외는 클래스 이름만 남긴다(Pydantic `ValidationError` 문자열에는 입력값이 들어 있다).
  - Starlette는 `Exception` 처리기로 응답한 뒤에도 예외를 다시 던져 uvicorn이 트레이스백을 남긴다. 예외는 라우트·미들웨어 안에서 잡고 다시 던지지 않는다.
- 요청 사이 상태, 캐시, 메모이제이션 금지. 질문과 답을 짝지은 표 금지. 채점 요청을 알아보는 분기 금지.
- 응답은 항상 HTTP 200 + 규격 JSON(인증 실패만 401). 422와 예외도 전역 처리로 200 대체 응답. 스트리밍 금지.
  라우트는 끝 슬래시 없이 `/tutor`, `/guard`.
- 값은 Pydantic `Literal`로 검증. `citations`는 항상 문자열 리스트, `answer`는 4,000자 이하.
- 요청당 30초, 동시 64개. 전체 채점 2분 이내 목표. 색인과 임베딩은 서버 시작 때 만든다.

## 3. 코드 구조

- `server.py`: 기동용(`python3 -m uvicorn server:app`), `app.main.app`을 가져오기만 한다.
- `app/main.py`: 라우트, 토큰 검사, 422 처리기, 요청 전체 마감(25초), `SafetyNet` 미들웨어(예외를 200 대체 응답으로 바꾸고 다시 던지지 않음, 요청당 메타데이터 로그 한 줄).
- `app/schemas.py`: 요청·응답 Pydantic 모델(Literal). `spec_version`, `session_id`는 쓰지 않으므로 선언하지 않는다.
- `app/llm.py`: 공용 AsyncOpenAI 클라이언트. `chat_text`(1회), `chat_json`(json_object + Pydantic 검증 + 재시도). 호출마다 메타데이터 로그.
- `app/logs.py`: 허용 필드만 남기는 로그. openai/httpx 로거는 WARNING으로 고정(`OPENAI_LOG=debug`가 본문을 찍는 것 방지).
- `app/deadline.py`, `app/config.py`, `app/templates.py`(고정 문구).
- `app/generate.py`: `generate_tutor(messages, verdict, hits, deadline=)`. `app/retrieval.py`: `retrieve(messages, k)`(작업 2). `app/grounding.py`: `judge_grounding(messages, candidate, hits)`(작업 3).

## 4. 테스트와 평가

```bash
# offline tests (fake LLM, no network)
python3 -m unittest discover -s tests -v
# live smoke test against the 8001 test server (1 real LLM call); marker must not appear in the server log
python3 tests/smoke_live.py --base http://localhost:8001
set -a; . ./.env; set +a
# test server
python3 -m uvicorn server:app --host 127.0.0.1 --port 8001
# full score on the dev split
python3 harness/score_local.py --tutor-url http://localhost:8001/tutor --guard-url http://localhost:8001/guard \
  --token "$TEAM_TOKEN" --public eval/public_dev --kb kb --workers 64
# guard metrics (confusion matrix, block P/R/F1, latency)
python3 eval/guard_report.py --url http://localhost:8001/guard --data eval/public_dev/guard.jsonl --workers 64
```

- 공개 세트 분할: `eval/make_split.py` → `eval/split.json`(묶음별 `fewshot`/`dev` 배정), `eval/public_dev/`(평가용 jsonl).
  - 사용자 발화 문자 3-gram Jaccard 0.5 이상(전체 사용자 발화 결합 또는 마지막 발화 중 하나라도)이면 같은 묶음.
  - (엔드포인트, 묶음 범주 조합)으로 층화, 고정 seed.
- 평가용(`dev`) 문항은 프롬프트나 few-shot에 넣지 않는다. few-shot은 `fewshot` 분할의 공개 세트 id 목록만 두고 서버 시작 때 `public/`에서 불러온다.

## 5. 기준선

- 공식: 총점 29.18(튜터 28.5, 가드 29.85), 채점 44.4초.
- 로컬(공개 세트 전체): 32.59.
- 로컬(평가용 분할 `eval/public_dev`, 시작용 서버, workers 64): 총점 30.55(튜터 33.5, 가드 27.61), 형식 오류 0, 판정 오류 0, 실행 70.3초.

- 단계 2(골격 정비) 뒤 평가용 분할: 총점 31.0(튜터 34.38, 가드 27.61), 형식 오류 0, 판정 오류 0, 실행 56.5초.
  튜터 지연 p50 6.5초, p95 11.5초, 최대 14.5초. 생성 100건 중 1건 빈 응답 → FALLBACK.

## 6. 확정된 설계 결정

- 가드 decision은 범주 표로 정한다. block = `academic_dishonesty`, `harmful_content`, `jailbreak_injection`, `off_topic`.
- 튜터 action: normal·pii_disclosure → answer, out_of_material → abstain, answer_seeking·academic_dishonesty → scaffold,
  jailbreak_injection → refuse, off_topic → redirect, emotional_distress → escalate, low_signal_input → clarify,
  harmful_content → 실행 요청이면 refuse, 차별·모욕 발언이면 redirect.
- low_signal 규칙: 마지막 사용자 발화 원문에서 공백을 뺀 문자열이 비었거나, 1~12자이면서 완성형 한글·영문·숫자가 없으면 low_signal_input.
- 대체 응답: tutor 요청 검증 실패(422)는 CLARIFY 문구 + `clarify`, 생성 실패·시간 초과는 FALLBACK 문구 + `answer`,
  guard 실패는 `allow`/`normal`(후보 답변이 있으면 grounded true, unsupported [], safe true).
