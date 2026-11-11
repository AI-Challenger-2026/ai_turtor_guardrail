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
- 대회 규칙에 어긋나거나 조금이라도 걸리는 것이 있으면 즉시 보고하고 허락을 받은 뒤 진행한다.
- 제출은 사용자가 한다.

## 2. 대회 규칙

- 서버의 외부 호출은 `OPENAI_*`(gpt-5-nano)와 `EMBEDDING_*`(text-embedding-3-small)뿐. 다른 LLM, 웹 검색, 외부 API 금지.
- 미리 학습된 모델 파일(임베딩, 리랭커, 분류 모델) 금지. 형태소 분석기, 토크나이저, BM25 라이브러리는 허용.
- 서버에 넣을 수 있는 데이터: `kb/`, `public/`과 그 가공물, gpt-5-nano로 만든 예시, 팀이 쓴 프롬프트와 규칙.
  Claude나 다른 LLM으로 만든 별도의 예시 묶음 파일·데이터셋 파일은 서버에 넣지 않는다. 프롬프트 안에 직접 쓴 예시는 허용.
  공개 문항에 Claude가 붙인 라벨(예: harm_kind)은 절대 쓰지 않는다. 라벨은 `public/` 라벨과 그것을 코드로 바꾼 값만 쓴다(few-shot id 파일은 id만 허용, 어기면 서버가 뜨지 않음).
- 로그는 시각, 엔드포인트, 상태 코드, 지연, 입력 길이, 오류 종류만 남긴다. uvicorn 접근 로그(클라이언트 IP·포트, 경로)는 `logs.setup()`에서 끈다.
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
  lifespan에서 지식 베이스 임베딩을 만든 뒤 요청을 받는다. /tutor와 /guard 모두 판정과 검색을 동시에 돌리고, 필요 없으면 검색을 취소한다.
  /guard: block이면 후보 답변은 `{grounded: true, [], safe: false}`. allow + 후보 답변이면 근거 판정. 후보 답변 없이 normal이면 자료 확인 → 없으면 out_of_material(decision은 그대로 allow).
- `app/schemas.py`: 요청·응답 Pydantic 모델(Literal). `spec_version`, `session_id`는 쓰지 않으므로 선언하지 않는다.
- `app/llm.py`: 공용 AsyncOpenAI 클라이언트(채팅은 `OPENAI_*`, 임베딩은 `EMBEDDING_*`만). `chat_text`(1회), `chat_json`(json_object + Pydantic 검증 + 재시도), `embed`. 호출마다 메타데이터 로그.
  운영진 프록시는 스트리밍이 아닌 호출의 `max_completion_tokens`가 2000을 넘으면 HTTP 400으로 거절한다(생성·판정 모두 2000 이하).
- `app/logs.py`: 허용 필드만 남기는 로그. openai/httpx 로거는 WARNING으로 고정(`OPENAI_LOG=debug`가 본문을 찍는 것 방지).
- `app/deadline.py`, `app/config.py`, `app/templates.py`(고정 문구).
- `app/classify.py`: low_signal 규칙 → LLM 판정(JSON, `{"category", "harm_kind"}`, 시도당 10초, 재시도 1회, 모두 실패 시 normal).
  다수결: 판정 2회를 동시에 보내고 (범주, harm_kind)가 다르면 3번째로 다수결(`CLASSIFY_VOTES`, 환경변수 1 또는 3, 기본 3). 다수가 없으면 범주 → block/allow 다수 → 첫 호출 순. out_of_material은 normal로 바꾸고, 자료 유무는 생성기가 정한다. 판정 결과는 로그에 남기지 않는다.
  자기소개 규칙: 다수결 결과가 normal이고 마지막 발화가 자기소개 문형(나/저 + 학교 + 학년 + 이름, 또는 "내 이름은 ○○")이면 pii_disclosure. 일반 문형만 보고 이름·학교 목록은 쓰지 않는다. 판정 호출이 모두 실패한 경우에는 적용하지 않는다.
  few-shot = 고정 20개 + 요청마다 고르는 비슷한 예시 6개(`CLASSIFY_DYNAMIC_SHOTS`, 가장 비슷한 것이 마지막). 예시 저장소는 예시용 분할 중 예시가 될 수 있는 143건(low_signal·out_of_material·공개 라벨로 harm_kind를 알 수 없는 가드 harmful 제외), 글자 2-gram BM25(마지막 발화 2배 + 이전 학습자 발화).
  `classify(..., exclude=)`는 평가 전용(묶음 제외 평가). 라우트는 쓰지 않는다(규칙 점검 테스트).
- `app/fewshot.py` + `app/fewshot_ids.json`: id만 담는다. harm_kind는 공개 튜터 action에서만 얻는다(refuse → request, redirect → statement).
- `app/prompts.py`: 판정 프롬프트, 튜터 생성 프롬프트(answer·scaffold 규칙, 인용 규칙), `FOUND_RULES`(자료 유무 판단, 튜터 생성과 가드 자료 확인이 같은 문구를 씀), 근거 판정 프롬프트, 영어 질문 번역 프롬프트, 대화 렌더링(긴 발화는 앞과 끝을 남기고 자름). `app/fewshot.py` + `app/fewshot_ids.json`: few-shot은 공개 세트 id만 두고 시작 때 `public/`에서 읽음. 평가용 id가 섞이면 서버가 뜨지 않는다.
- `app/policy.py`: 범주 → decision, (범주, harm_kind) → action·템플릿. `app/templates.py`: 고정 문구(사용자 발화를 끼워 넣지 않음).
- `app/retrieval.py`: 섹션 = md + 같은 이름의 표 CSV(채점기 `load_kb_dir`와 같은 결합). 글자 2-gram BM25(모듈을 불러올 때 색인), 소제목(`## `) 조각 244개 임베딩(시작 때).
  질의 = 대화 전체(튜터의 이전 답변이 학습자가 보는 페이지를 알려 줌) + `extra`(가드의 후보 답변). 후보 5개 = BM25 상위 4 + BM25에 없는 임베딩 1위. 질의 임베딩이 3초를 넘거나 실패하면 BM25만. 요청 중에는 상태를 쓰지 않는다.
  마지막 발화가 영어 위주(라틴 문자 12자 이상, 한글의 2배 초과)면 minimal 번역 1회(4초)를 임베딩과 동시에 돌려 BM25 질의에 덧붙인다. 번역 프롬프트의 용어집은 시작 때 지식 베이스(섹션 제목, 과정 고유 이름)에서 만든다.
- `app/generate.py`: `generate_tutor(messages, verdict, hits, mode=, deadline=)`. JSON 1회 호출 `{"found", "citations", "answer"}`(재시도 1회, effort low).
  found=false면 action abstain(자료에 없다고 말하고 가까운 자료 내용을 덧붙여 인용). answer_seeking은 scaffold 모드(자료의 규칙·값은 말하고 최종 답은 말하지 않음), 실패하면 SCAFFOLD_ANSWER_SEEKING 문구.
  인용 = 모델이 고른 섹션(보여 준 것만) + 검색 1위, 최대 2개. pii면 이름·학교를 되풀이하지 말라는 지시 + 끝에 PII_NOTICE.
- `app/grounding.py`: `judge_grounding(messages, candidate, hits, deadline=)` → (grounded, 근거 없는 구간).
  배심원: 판정 `GROUND_JUDGES`회(환경변수 1~3, 기본 3)를 동시에, 다수결(동수면 첫 판정). 전체의 과반이 같은 결론이면 나머지 판정은 취소하고 바로 정한다(3명이면 2명 일치).
  구간은 근거 없음이라 한 첫 두 판정이 겹치는 글자만(4자 미만이면 첫 판정 구간). "첫"은 끝난 순서가 아니라 보낸 순서. 구간은 후보 답변의 부분 문자열만 남긴다(따옴표·끝 구두점 정리, 겹침 제거, 최대 4개).
  근거 없음인데 쓸 구간이 없으면 후보 답변 전체를 구간으로 낸다. 판정 실패 시 `(True, [])` + degraded=ground. `judge_found(messages, hits, deadline=)` → 자료 유무(실패 시 normal 유지 + degraded=found).
- 판정기 reasoning_effort는 환경변수 `CLASSIFY_EFFORT`(기본 low, 단계 4에서 결정).

## 4. 테스트와 평가

```bash
# offline tests (fake LLM, no network)
python3 -m unittest discover -s tests -t . -v
# live smoke test against the 8001 test server (2 tutor generations); marker must not appear in the server log
python3 tests/smoke_live.py --base http://localhost:8001
set -a; . ./.env; set +a
# test server
python3 -m uvicorn server:app --host 127.0.0.1 --port 8001
# full score on the dev split
python3 harness/score_local.py --tutor-url http://localhost:8001/tutor --guard-url http://localhost:8001/guard \
  --token "$TEAM_TOKEN" --public eval/public_dev --kb kb --workers 64
# guard metrics (confusion matrix, block P/R/F1, latency)
python3 eval/guard_report.py --url http://localhost:8001/guard --data eval/public_dev/guard.jsonl --workers 64
# classifier only, more samples: also send dev tutor items to /guard (no generation calls)
python3 eval/guard_report.py --url http://localhost:8001/guard --data eval/public_dev/guard.jsonl eval/public_dev/tutor.jsonl --workers 64
# tutor actions, abstain P/R, citation P/R/F1, latency, PII name repetition (no judge calls); --categories/--ids to run a subset
python3 eval/tutor_report.py --url http://localhost:8001/tutor --categories pii_disclosure
# tuning set for generation (fewshot split minus few-shot ids), generation categories only
python3 eval/tune_ids.py > /tmp/tune_ids.txt
python3 eval/tutor_report.py --url http://localhost:8001/tutor --data public/tutor.jsonl --ids @/tmp/tune_ids.txt \
  --categories normal,pii_disclosure,answer_seeking,out_of_material --workers 16
# retrieval recall (embedding calls only; --vec-cache keeps vectors in a local eval-only file)
python3 eval/retrieval_report.py --split fewshot
# tutor points split into the grader's items (action, citation F1, gold facts, faithfulness); uses the local judge
python3 eval/tutor_breakdown.py --url http://localhost:8001/tutor --data public/tutor.jsonl --ids @/tmp/tune_ids.txt \
  --categories normal,pii_disclosure,answer_seeking,out_of_material --dump /tmp/breakdown.json
# classifier only, in-process, leave-group-out on fewshot-split items (--dynamic 0 = static shots only, --votes 1|3)
python3 eval/classify_report.py --ids @/tmp/tune_ids.txt --dynamic 6 --votes 3 --json-out /tmp/run.json
# guard candidate judge only, in-process (no public examples in the judge, so every fewshot-split item is fair)
python3 eval/ground_report.py --split fewshot --judges 3 --runs 2
```

- `tests/test_task2.py`(검색·생성), `tests/test_task3.py`(가드 근거 판정·자료 확인·영어 번역 검색).
- `tests/test_compliance.py`: 대회 규칙 정적 점검(본문 print·로그, 다른 네트워크 클라이언트·호스트, 캐시, 파일 쓰기, `app/`의 데이터 파일, 평가용 문장·참고 답안이 코드에 들어갔는지). 규칙을 증명하지는 않고 어긋나면 알린다.
- 실서버 로그 감사: 모든 줄이 `event=` 메타데이터(허용 필드만) 또는 uvicorn 시작·접근 줄이어야 한다.

- 공개 세트 분할: `eval/make_split.py` → `eval/split.json`(묶음별 `fewshot`/`dev` 배정), `eval/public_dev/`(평가용 jsonl).
  - 사용자 발화 문자 3-gram Jaccard 0.5 이상(전체 사용자 발화 결합 또는 마지막 발화 중 하나라도)이면 같은 묶음.
  - (엔드포인트, 묶음 범주 조합)으로 층화, 고정 seed.
- 평가용(`dev`) 문항은 프롬프트나 few-shot에 넣지 않는다.
- 프롬프트·few-shot 조정은 조정용 세트(`python3 eval/tune_ids.py`: 예시용 분할 중 few-shot이 아닌 문항)로 하고, 평가용은 단계 끝 확인에만 쓴다. few-shot은 `fewshot` 분할의 공개 세트 id 목록만 두고 서버 시작 때 `public/`에서 불러온다.

## 5. 기준선

- 공식: 총점 29.18(튜터 28.5, 가드 29.85), 채점 44.4초.
- 로컬(공개 세트 전체): 32.59.
- 로컬(평가용 분할 `eval/public_dev`, 시작용 서버, workers 64): 총점 30.55(튜터 33.5, 가드 27.61), 형식 오류 0, 판정 오류 0, 실행 70.3초.

- 단계 2(골격 정비) 뒤 평가용 분할: 총점 31.0(튜터 34.38, 가드 27.61), 형식 오류 0, 판정 오류 0, 실행 56.5초.
  튜터 지연 p50 6.5초, p95 11.5초, 최대 14.5초. 생성 100건 중 1건 빈 응답 → FALLBACK.
- 단계 3(판정기 minimal, few-shot 20개) 뒤 평가용 분할: 총점 65.35(튜터 59.0, 가드 71.71), 형식 오류 0, 판정 오류 0, 실행 59.8초.
  가드 block F1 0.759, 범주 정확도 0.690(상한 0.857), 후보 답변 0.682. 튜터 지연 p50 8.1초, p95 14.4초, 최대 21.5초.
- 단계 4(reasoning_effort): 평가용 200건 + 조정용 180건(guard_report, guard와 tutor 문항)에서 low가 두 표본 모두 우세.
  합계 block F1 minimal 0.787 → low 0.837, 범주 정확도(oom 제외) 0.750 → 0.785, harmful 재현율 27/38 → 34/38, pii 재현율 둘 다 4/19.
  판정 호출 지연(64 동시) p50 약 2.1초 → 2.3초. low는 JSON 형식 실패가 약 1%(재시도로 대부분 복구).
- 프롬프트 v2(경계 원칙 보강, few-shot 3개 교체: pii 2개, 남의 이름·학교 요구 1개) + low:
  조정용 세트(두 버전 모두 few-shot이 아닌 묶음 177건) 범주 정확도 0.844 → 0.922, pii 2/7 → 6/7, block F1 0.875 → 0.867(오차단 5 → 7).
  평가용 200건: block F1 0.904(v1 minimal 0.768), 범주 정확도(oom 제외) 0.896(0.740), pii 6/10, harmful 20/20, normal 오차단 7(10).
  평가용 guard 문항만의 가드 점수 80.41(작업 1 상한 84.94).
  남은 약점: jailbreak 표현을 배우려는 질문의 오차단, 인용한 차별 문장, 게임 공략 off_topic 누락, 제출용 academic 누락. 규칙 문장으로는 잘 안 고쳐져 동적 few-shot이 후보.
- 단계 5(v2 + low, `b02d853`) 평가용 분할, workers 64: 총점 69.03(튜터 59.75, 가드 78.3), 형식 오류 0, 판정 오류 0, 실행 약 51초.
  튜터 지연 p50 6.8초, p95 13.0초, 최대 15.1초(25초 마감 도달 0, FALLBACK 0). 가드 p50 1.9초, 최대 4.7초.
  판정 호출 196건 모두 1회에 성공(재시도 0), 생성 64건 오류 0.
  튜터 normal 48.98: 인용이 빈 리스트라 인용 F1 0, 근거 문단이 없어 충실성 판정도 낮음 → 작업 2(검색·인용)가 주된 개선 지점.

- 작업 2(검색, 생성, abstain, 인용) 평가용 분할(`chan-task2`, workers 64): 총점 82.18(튜터 86.58, 가드 77.78), 형식 오류 0, 판정 오류 0, 실행 56.4초.
  튜터 범주: normal 48.98 → 86.39, out_of_material 40 → 80, pii 57.14 → 96.43, answer_seeking 43.75 → 75.
  튜터 지연 p50 6.6초, p95 11.8초, 최대 15.6초, FALLBACK 0. 생성 70건 오류 0, 질의 임베딩 96건 중 1건 3초 초과(BM25로 처리).
  검색(예시용 분할 59건): 1위 0.881, 5위 안 0.966. 조정용 세트 3회차: 인용 F1 0.78, oom 8/10, 정상 오abstain 3/51.
  남은 약점: 영어 질문의 검색 누락, 학습자 주장 확인 질문("~는 진짜 ~임")의 오abstain, 특정 기준을 묻는 oom 일부.
- 작업 3(가드 근거 판정, 가드 out_of_material, 자료 판단 규칙 공유, 영어 번역 검색) 평가용 분할(`chan-task3`, workers 64): 총점 85.45(튜터 82.67, 가드 88.24), 형식 오류 0, 판정 오류 0, 실행 58.9초.
  가드 77.78 → 88.24. 튜터 86.58 → 82.67은 주로 판정기 변동(정상 4건 거절, off_topic·harmful 각 1건 놓침, 판정기 코드는 그대로).
  지연: 튜터 p50 7.1초, p95 12.1초, 최대 16.9초. 가드 p50 4.2초, p95 8.5초, 최대 13.2초. FALLBACK·degraded 0.
  조정용 가드(89건): out_of_material 6/6(정상을 oom으로 예측 0), 근거 없음 14/15 검출, 근거 있음 오판 1/17, 구간 F1 0.386 → 0.592(구간 필수 지시 + 전체 답변 대체).
  조정용 튜터: out_of_material 9/10, 정상 오abstain 1, 인용 F1 0.812. 평가용 생성 범주: oom 6/10, 오abstain 2, 인용 F1 0.772.

- 판정기: Claude가 붙인 harm_kind 라벨 3건을 공개 라벨에서 harm_kind가 나오는 문항으로 교체, 비슷한 예시 6개 추가, uvicorn 접근 로그 끔.
  조정용 180건(묶음 제외 평가): 고정만 161건 정답(block F1 0.833, 범주 0.833, 정상 차단 7) → 동적 2회 162·165건(F1 0.840·0.892, 범주 0.833·0.879, 정상 차단 6·5).
  평가용 분할(workers 64): 총점 87.69(튜터 84.17, 가드 91.2), 형식 오류 0, 판정 오류 0, 실행 66.9초. 튜터 p95 12.9초·최대 14.4초, 가드 p95 8.8초·최대 11.6초, FALLBACK 0.
  평가용에서 매번 틀리던 가드 5건(정상 질문 차단 4, pii 누락 1)은 그대로 틀림. 점수 상승은 다른 문항(가드 harmful·jailbreak 100)에서 왔고, 일부는 회차 변동일 수 있다.

- 공식 제출(`b394ab8`): 86점(로컬 평가용 87.69와 일치).
- 생성 프롬프트 보강(용어 그대로, 관계없는 사실 덧붙이지 않기, 행동·단계 빠짐없이, 스캐폴드 조건 모두, 주장 확인 예시, found=false 전 모든 섹션 확인).
  진단(조정용 생성 범주 65건, 로컬 판정): normal 84.75, 손실 행동 3.2·인용 5.1·사실 3.7·충실 3.2.
  조정용 3회: 생성 범주 가중 평균 85.4 → 85.4 → 83.8 → 87.4. 2회차의 "답했으면 found=true" 규칙은 oom 행동을 0.8 → 0.6으로 망쳐 뺐다.
  최종: normal 87.4(충실 0.872 → 0.979), oom 85, answer_seeking 93.8. 평가용 분할: 총점 88.28(튜터 86.08, 가드 90.47), 실행 56.0초, FALLBACK 0.

- 판정기 다수결(`74a422a`): 조정용 180건 묶음 제외 2회씩, 투표 없음 → 투표: 정답 162·165 → 164·165, block F1 0.840·0.892 → 0.878·0.892,
  같은 설정 2회 사이 범주가 바뀐 문항 9 → 5, block/allow가 바뀐 문항 4 → 1. 판정 호출 문항당 약 2.0회(3번째 호출은 약 5%).
  평가용 분할(workers 64): 총점 88.3(튜터 86.0, 가드 90.6), 형식 오류 0, 판정 오류 0, 실행 65.9초, FALLBACK·degraded 0, 로그 감사 0줄.
  지연: 튜터 p50 8.8초·p95 16.8초·최대 19.4초, 가드 p50 7.8초·p95 13.4초·최대 17.3초. 같은 실행에서 생성·근거 판정·임베딩 호출도 함께 느려져(프록시 전반) 투표만의 몫은 가르지 못함.
- 같은 날 시험했으나 채택하지 않은 것(다시 하지 않기):
  - 판정기 medium: pii 2~3/8 → 6~7/8이지만 정상 차단 5 → 6~8, block F1 하락, 판정 p95 약 11초·최대 24.5초.
  - 판정기에 지식 베이스 섹션 제목 목록 넣기: 정답 165·165(투표만 164·165), 정상 차단 그대로 5.
  - block 판정 재검증 호출(배우려는가/시키는가 이진 판정): 조정용 block 판정 83건 중 뒤집은 것 0(명백한 지어낸 예시는 맞게 판정).
  - 예시 저장소를 공개 세트 전체(311건)로 확장: 묶음 제외로 평가용 200건 정답 175 → 175, 조정용 164·165 → 165. 저장소는 예시용 분할 그대로.
  - 인용을 답변과 글자 겹침이 큰 섹션 하나로: 두 섹션 인용 문항 38건에서 F1 합 23.3 → 25.0, 회차별로 들쭉날쭉하고 충실성 위험.
  - BM25 질의 변형(마지막 발화 가중, 학습자 발화만, 마지막 발화만): 지금(대화 전체)이 1위 52/59로 가장 좋음.
  - 근거 판정 medium: 34건 중 6~7건 실패(시간·토큰), 근거 없음 놓침 1 → 6. low는 14/15 검출, 구간 F1 0.65~0.68(문장 통째로 잡는 경향).
  - 튜터에 자료 확인 호출(`judge_found`) 추가: oom 7/10, 자료 있음 오판 2~3/55로 생성기(6~8/10, 1~4/55)와 비슷.
  - 공개 라벨로 직접 학습한 분류기(SPEC §7.1 허용; 임베딩 또는 글자 n-gram TF-IDF + 로지스틱 회귀, 묶음 10겹 교차 검증, numpy·scikit-learn은 실험용으로만 설치):
    혼자서 block F1 0.66~0.71(LLM 투표 0.86~0.87). LLM 판정을 분류기가 확신할 때만 뒤집으면 369건 중 0~1건 변화.
    LLM이 틀린 문항에서 분류기 확률도 중앙값 0.43~0.46(정상 차단 17건 중 확률 0.3 미만 3건, 맞게 막은 82건 중에도 5건)이라 가를 수 없음.

- 전략 단계(라우팅·배심원 계열, `chan-strategy`):
  - S3 근거 판정 배심원(3명 다수결, 겹치는 구간) 채택. 예시용 가드 34건: 구간 F1 0.649~0.676(단일 4회) → 0.721·0.773, 후보 답변 점수 평균 0.931 → 0.938.
    평가용 33건(마무리 확인): 구간 F1 0.627·0.661 → 0.767·0.719, 후보 답변 점수 0.925·0.921 → 0.953·0.933. 판정 지연 중앙값 약 +1초, 최대 10.5초.
    "하나라도 근거 없음이면" 규칙(2명)은 근거 있는 답변 오판이 늘어 점수 0.934, "다르면 첫 판정"(2명)은 0.924.
    평가용 분할 전체(`12d7528`, 셋 다 기다림, workers 64): 총점 87.08(튜터 83.5, 가드 90.65), 형식 오류 0, 판정 오류 0, 로그 감사 0줄.
    후보 답변 문항 44건 평균 95.19 → 96.40. 튜터 -2.5는 판정기 회차 변동(공격 3건 놓침, 판정 호출 실패 0, 튜터 경로는 코드 변화 없음).
    가드 지연 p50 7.9 → 8.3초, p95 14.7 → 16.1초, 최대 17.3 → 24.6초(판정 하나가 13.7초에 시간 초과, 가장 느린 판정을 기다림).
  - 배심원 조기 결정(2명 일치하면 셋째 취소): 같은 판정 출력으로 두 규칙을 비교(예시용 34건 × 2회).
    결정은 같고 구간 F1 0.713·0.743 → 0.730·0.747, 후보 답변 점수 0.928·0.923 → 0.931·0.924, 판정 단계 지연 p50 4.7·4.4 → 3.2·3.3초, 최대 9.1·9.0 → 9.1·6.0초.
    끝난 순서별 구간 F1 0.765·0.723·0.741로 빠른 판정의 구간이 더 나쁘지 않음. 같은 표본의 단일 판정은 구간 F1 0.587·0.637, 점수 0.910·0.888.
  - S5 판정기 비슷한 예시를 임베딩으로 고르기: 오프라인(API 호출 없음, 공개 문항 임베딩은 평가 전용 파일)으로 조정용 173건 묶음 제외 비교.
    고른 예시 중 정답 범주 비율 BM25 0.585 > 혼합(3+3) 0.577 > 임베딩 0.560, 6-NN 범주 정답 132 > 130 > 125.
    투표 2회에서 틀린 18건에서 정답 범주 예시 합 39 → 43(혼합)뿐이고, 오차단된 정상 문항은 BM25로도 정상 예시 3~5개를 받고 막힘(예시 선택 문제가 아님).
    분류 전에 질의 임베딩을 기다려야 해 지연도 늘어 LLM 비교 없이 채택 안 함.
  - S2 페이지 재순위 호출(학습자가 보는 섹션을 LLM이 고름): 예시용 59건 정답 49·48 < 검색 1위 52. 채택 안 함.
  - S1 투표가 갈릴 때 3번째 표를 medium으로(8초 상한, low 예비 표): 조정용 정답 164·165 → 165·167이었으나 평가용 200건 175 → 177, block F1 0.852 → 0.836(정상 차단 12 → 14), 판정 p95 5.3 → 9.2초. 채택 안 함.

- 고정 손실 감사(조정용, 코드 변경 없음): 생성 범주 65건 2회에서 두 번 다 잃은 손실 9.23점 = 검색 실패 0, 입력만으로 맞힐 수 없음 1.92(같은 사실이 다른 섹션에도 있음), 판정기 오차단 2.31, 자료 유무 판단 1.92, 생성 실패 3.08. 원인 분류는 Claude가 문항을 읽고 판단한 것이며 분석용으로만 쓰고 서버에 넣지 않는다.
  평가용 실행 3회 비교: 손실 12.7점 중 8.4점이 가장 잘 나온 회차에서도 잃는 몫. 다수결·배심원 같은 반복 호출은 나머지 4.3점만 건드린다.
  - 차단 판정을 수업 자료와 함께 재판정(채택 안 함, 다시 하지 않기): 조정용 차단 판정 41·42건에서 오차단 2~3건을 고치지만 진짜 공격 6~8건을 풀어 줌. 정답 164·165 → 161·160, block F1 0.878·0.892 → 0.822·0.806(프롬프트 1회 수정 후).
  - 자기소개 pii 규칙 채택: 저장된 묶음 제외 판정에 적용해 조정용 2회 5건씩 고침·잘못 바뀜 0, 범주 정확도 0.833·0.864 → 0.909·0.939. 평가용(패턴 작성 때 안 본 문항) 5건 고침·잘못 바뀜 0, pii 4/10 → 9/10, 범주 정확도 0.844 → 0.909. 다른 범주 380건 중 패턴에 걸린 것 0.
  - 평가용 분할 전체(`fffa96c`, 다수결 + 배심원 조기 결정 + 자기소개 규칙, workers 64): 총점 88.45(튜터 84.08, 가드 92.82), 형식 오류 0, 판정 오류 0, degraded 0, 실행 66.8초, 로그 감사 0줄.
    가드 후보 답변 문항 평균 97.43, 가드 pii 100. 튜터는 코드가 거의 같은 4회가 83.5~86.08(평균 84.92)로 회차 변동이 크다.
    지연: 튜터 p50 10.5초·p95 19.7초·최대 22.5초, 가드 p50 9.1초·p95 17.3초·최대 20.8초. 호출 수는 `12d7528`과 거의 같고(810 → 807) 질의 임베딩 시간 초과 12 → 28건이라 프록시 혼잡으로 보임.
  - 공식 제출(`fffa96c`): 85점(튜터 81, 가드 88). 로컬 88.45와 차이 -3.5, 이전 공식 차이(-1.7, -3.2)와 같은 경향. 비공개 문항은 공개 세트와 겹치지 않는다.
  - 튜터 생성 변형(조정용 63건, 같은 측정 도구, 판정기 오류 제외, 다시 하지 않기): 기준 90.61·89.02. 근거 문장을 먼저 뽑기(+ abstain 덧붙임 제한) 86.90·89.81, 보여 주는 섹션 3개 89.02·88.36, 사례 앞뒤 사실 추가 88.49·88.62. 모두 채택 안 함. 모델은 틀린 실습의 문장을 정확히 복사하므로 인용문 검증으로 잡히지 않고, 검증 안 된 found=true 4건은 모두 답이 있는 문항.
  - 답변은 검색 1위만 인용(채택 안 함): 조정용 같은 답변 재판정 +0.40·+0.79였지만 평가용 전체에서 인용 변화로 보이는 문항은 +8.3 5건, -16.7 5건으로 손해. 검색 1위 정확도는 조정용 0.898, 평가용 0.900으로 같고, 1위가 틀린 약 10% 문항을 모델의 선택이 구해 주는 몫과 정답인데 인용이 2개라 깎이는 몫이 비슷하다.
    같은 실행 총점 86.75(튜터 83.67, 가드 89.82). 가드 코드는 `fffa96c`와 같은데 92.82 → 89.82여서, 평가용 전체 1회의 가드 회차 변동은 3점까지 난다.

## 6. 확정된 설계 결정

- 가드 근거 판정은 3명 다수결 배심원(2명이 일치하면 바로 결정), 구간은 판정끼리 겹치는 부분.
- 판정기는 low + 다수결(2회 동시, 다르면 3번째). 예시 저장소는 예시용 분할만(공개 세트 전체로 넓혀도 묶음 제외 평가에서 이득 없음).
- 가드 decision은 범주 표로 정한다. block = `academic_dishonesty`, `harmful_content`, `jailbreak_injection`, `off_topic`.
- 튜터 action: normal·pii_disclosure → answer, out_of_material → abstain, answer_seeking·academic_dishonesty → scaffold,
  jailbreak_injection → refuse, off_topic → redirect, emotional_distress → escalate, low_signal_input → clarify,
  harmful_content → 실행 요청이면 refuse, 차별·모욕 발언이면 redirect.
- low_signal 규칙: 마지막 사용자 발화 원문에서 공백을 뺀 문자열이 비었거나, 1~12자이면서 완성형 한글·영문·숫자가 없으면 low_signal_input.
- 가드 out_of_material은 판정기가 아니라 지식 베이스로 정한다(후보 답변 없는 normal 입력만). 정상 문항을 oom으로 답해도 범주 정확도(정답 normal은 채점 안 함)와 block F1(둘 다 allow)에 손해가 없다.
- abstain은 고정 문구가 아니라 생성한다: 자료에 없다고 말하고, 자료의 가장 가까운 내용을 덧붙여 그 섹션을 인용한다(사용자 결정).
- answer_seeking 스캐폴드는 자료의 규칙·값은 알려 주고 최종 선택지·계산 결과는 말하지 않는다(정답 사실 라벨이 규칙이라서).
- 대체 응답: tutor 요청 검증 실패(422)는 CLARIFY 문구 + `clarify`, 생성 실패·시간 초과는 FALLBACK 문구 + `answer`(answer_seeking은 SCAFFOLD_ANSWER_SEEKING + `scaffold`),
  guard 실패는 `allow`/`normal`(후보 답변이 있으면 grounded true, unsupported [], safe true).
