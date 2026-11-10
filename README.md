# 안전벨트를 채운 AI 튜터 — `/tutor`, `/guard` 서버

교육 자료(지식 베이스 48개 섹션)를 근거로 답하는 튜터와, 위험한 입력과 근거 없는 답변을 거르는 가드를 하나의 FastAPI 서버로 제공합니다. 규격은 [SPEC.md](SPEC.md), 개발 과정과 모든 실험 기록은 [DEVLOG.md](DEVLOG.md)에 있습니다.

## 실행

```bash
cd /home/elicer/competition
pip install -r requirements.txt          # fastapi, uvicorn, openai, pydantic
set -a; source .env; set +a              # OPENAI_API_KEY, TEAM_TOKEN 등 (git 제외 파일)
python3 -m uvicorn server:app --host 0.0.0.0 --port 8000
```

- `.env`에 필요한 값: `OPENAI_API_KEY`(운영진 발급 팀 키), `TEAM_TOKEN`(채점 요청 인증 토큰). `OPENAI_BASE_URL`, `OPENAI_MODEL`은 런박스에 미리 설정돼 있습니다.
- 시작할 때 BM25 색인을 만듭니다(약 1.7초). 첫 요청 전에 준비가 끝납니다.
- 서버를 끌 때는 `Ctrl+C`를 씁니다(`Ctrl+Z`는 포트를 붙잡은 채 멈춥니다).
- 제출 주소: `https://<런박스 주소>.tunnel.elice.io/proxy/8000/tutor`, `…/guard` (`echo $VSCODE_PROXY_URI`로 런박스 주소 확인).

## 구조

```
                    ┌─ 분류기: LLM이 예/아니오 신호 9개 추출 × 3회 다수결 → 규칙으로 범주 결정 (app/classify*.py)
POST /tutor ── 동시 ┤
                    └─ BM25 검색(1위 섹션) → 근거 기반 답변 생성 (app/kb.py, app/answer.py)
                         │
                         ├ 범주가 거절·전환·도움 안내·되묻기·대필 → 고정 응답 (app/policy.py), 생성 결과는 버림
                         ├ "정답만" 요청 → 힌트 모드로 다시 생성(규칙은 주고 최종 정답은 주지 않음)
                         ├ 자료에서 답을 못 찾음 → abstain
                         └ 인용 = BM25 1위 섹션 1개, 개인정보를 밝힌 학습자에게는 안내 문장을 코드로 덧붙임

                    ┌─ 분류기 → allow/block + 범주 (app/policy.py)
POST /guard ── 동시 ┤
                    └─ 후보 답변이 있으면: BM25 상위 4개 섹션과 사실 대조 → 근거 없는 구간 추출 (app/verify.py)
                         ├ block 입력에 대한 후보 답변 → safe=false (요청에 응한 답변)
                         └ 그 밖: grounded = 근거 없는 구간이 없음
```

| 파일 | 역할 |
|---|---|
| [server.py](server.py) | 엔드포인트, 인증, 요청 정규화, 시간 예산, 추측 실행, 결과 조립 |
| [app/classify.py](app/classify.py), [app/classify_feat.py](app/classify_feat.py) | 입력 분류(신호 추출 프롬프트, 다수결, 신호 → 범주 우선순위 규칙) |
| [app/policy.py](app/policy.py) | 범주 → 가드 판정·튜터 action 규칙, 고정 응답 문구 |
| [app/kb.py](app/kb.py) | 지식 베이스 로딩(CSV 표 포함), 소제목 단위 청크, BM25(한글 음절 2-gram) |
| [app/answer.py](app/answer.py) | 근거 기반 답변 생성(답변 모드 / 힌트 모드) |
| [app/verify.py](app/verify.py) | 후보 답변 사실 검증, 근거 없는 구간 추출, 비주장 문장 필터 |
| [app/llm.py](app/llm.py) | LLM 호출(데드라인, 재시도, 헤지 요청), 본문 없는 로그 |

### 주요 설계 결정 (근거는 DEVLOG)
- **범주 → 판정 규칙**: 공개 세트에서 범주가 정해지면 가드 판정과 튜터 action이 예외 없이 결정됐다(1장).
- **BM25만 사용**: 수업 고유 명칭이 단서인 데이터라 임베딩보다 정확했다(R@1 0.92 vs 0.68, 2장).
- **답변은 인용 섹션 1개만 보고 생성**: 근거 충실도가 인용 섹션과만 대조되므로 답변과 인용이 구조적으로 일치한다. 이 변경을 포함한 5단계 후 실제 튜터 79 → 86(5장).
- **검증기 정밀도 우선**: 처음 보는 답변(정답 답변 200개)에서 오탐 32.5% → 5%(4장).
- **시간 관리**: 요청당 26초 예산, 분류·생성 동시 실행, 헤지 요청(8장).

## 운영 설정 (환경변수, 기본값으로 운영)

| 변수 | 기본값 | 뜻 |
|---|---|---|
| `REQUEST_BUDGET` | 26 | 요청 전체 시간 예산(초). 채점기 제한 30초에서 네트워크 여유를 뺀 값 |
| `CLASSIFY_BUDGET` | 12 | 튜터에서 분류에 쓸 최대 시간(초) |
| `CLASSIFY_VOTES` | 3 | 분류 다수결 횟수(1로 두면 단일 분류) |
| `CLASSIFY_EFFORT`, `ANSWER_EFFORT` | low | 추론 수준(medium은 30초 제한에 걸림, 5장) |
| `SPECULATIVE` | 1 | 분류와 답변 생성 동시 시작(0이면 순차 실행) |
| `HEDGE_AFTER` | 9 | 답변 생성이 이 시간(초) 안에 안 끝나면 같은 요청을 하나 더 보냄 |
| `TOP_K`, `VERIFY_TOP_K` | 1, 4 | 답변 생성 / 후보 답변 검증에 쓰는 섹션 수 |
| `LOG_FILE` | `logs/server.log` | 로그 파일 위치(git 제외) |

## 규칙 준수 (SPEC §7–8)

| 항목 | 이 서버 |
|---|---|
| 외부 호출 | 운영진 제공 채팅 API(`gpt-5-nano`)만. 임베딩 API·웹·외부 서비스 호출 없음 |
| 모델·데이터 | 사전학습 모델 파일 없음. 서버가 읽는 데이터는 `kb/`뿐(공개 세트는 실행 중에 읽지 않음). 외부·합성 데이터 파일 없음 |
| 프롬프트 예시 | 팀이 작성한 짧은 예시 문장(공개 세트 문항을 복사하지 않음) |
| 하드코딩·요청 수집 | 질문→답 표, 캐시, 요청 간 상태 없음. 색인은 시작 때 만든 읽기 전용 |
| 채점 때만 다른 동작 | 없음. 헤더·요청 패턴으로 분기하지 않음 |
| 로그 | **시간(ms), 상태, 입력·출력·후보 답변 길이, 오류 종류만** 남김(`app/llm.py`의 `log`, `server.py`의 호출 지점). 요청·답변 본문과 범주·판정처럼 문항 내용에서 나온 값은 남기지 않음 |
| 비밀값 | API 키·토큰은 `.env`(git 제외). git 기록 전체에 키·토큰 값 없음 |
| 응답 규격 | 토큰 불일치(401)를 빼고 어떤 입력에도 HTTP 200 + 규격 JSON. 리다이렉트 없음, 스트리밍 없음, 답변 4,000자 이하 |

로그 한 줄 예시:
```
11-10 04:33:00 tutor ms=7501 status=ok in_len=128 out_len=96
11-10 04:33:01 guard ms=4643 status=ok in_len=210 cand_len=74
```

## 개발용 도구 (`scripts/`, 서버와 무관)

| 스크립트 | 측정하는 것 |
|---|---|
| `eval_classify.py` | 분류기 범주 정확도, 가드 block F1, 튜터 action 정확도(공개 400문항) |
| `eval_retrieval.py`, `sweep_retrieval.py` | 검색 R@k, 설정 탐색(짝수=튜닝 / 홀수=확인 분할) |
| `eval_found.py` | "자료에 있나" 판단 정확도 |
| `eval_guard_output.py`, `stress_verify.py` | 검증기 탐지·구간 F1, 정답 답변 오탐률(튜닝에 안 쓴 데이터) |
| `diagnose_tutor.py`, `ab_tutor.sh` | 튜터 채점 항목별 진단, 설정 A/B 비교 |
| `load_test.py` | 실전 조건 부하(600요청 × 동시 64, 형식 검증) |

로컬 채점기는 `python3 harness/score_local.py --tutor-url http://localhost:8000/tutor --guard-url http://localhost:8000/guard --token "$TEAM_TOKEN" --public public --kb kb`로 실행합니다.

## 결과 요약

| 단계 | 실제 제출 총점 (튜터 / 가드) |
|---|---|
| 베이스라인 | — (로컬 31.6) |
| 1차: 분류기 + 검색 | 약 79 (81.33 / 77.88) |
| 3차: 검증기 오탐 제거 | 82.33 (79.08 / 85.58) |
| 4차: 답변 섹션 1개, 다수결 | **87.95** (86.08 / 89.83) |
| 같은 버전 반복 제출 평균 | 86.2~86.9 (제출 간 노이즈 약 ±0.8) |

전체 과정, 기각한 실험, 측정 방법은 [DEVLOG.md](DEVLOG.md)를 참고하세요.
