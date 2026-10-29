# 안전벨트를 채운 AI 튜터 — 교육 자료 근거 튜터 + 가드레일

AI Bootcamp Innovation Challenger 2026 (엘리스그룹 주제) 팀 저장소입니다. 비공개 저장소이니 링크나 내용을 팀 밖에 공유하지 마세요.

> **공식 주제:** 교육 자료에 근거한 정확한 학습 지원과, 범위 이탈 및 유해 요청 차단을 동시에 달성하는 튜터 에이전트 및 가드레일 설계

- **학습 에이전트:** 검색된 교육 자료 청크 안에서만 답하고, 답변마다 `[1]` 인용과 출처(문서 › 단원 › 섹션)를 붙입니다.
- **가드레일:** 통제 지점 세 곳(입력, 처리, 출력)에 검사를 붙입니다. 어떤 가드를 쓸지는 설정 파일(가드 프로필)로 고릅니다.
- **API 키 없이 실행:** `MOCK_MODE=true`(기본값)면 화면, 테스트, 평가가 키 없이 끝까지 돕니다. 처음 받은 팀원도 바로 실행해 볼 수 있습니다.

**현재 상태 (2026-10-29):** 베이스라인(Phase 1~4) 완료, 백엔드 테스트 130개 통과, 실제 Claude API 연동 확인. 엘리스 공식 데이터와 ML API는 아직 연결 전이라 샘플 데이터와 Claude API로 동작합니다. 자세한 현황은 [PROGRESS.md](PROGRESS.md)에 있습니다.

---

## 처음 온 팀원이 읽을 순서

| 순서 | 문서 | 내용 |
| --- | --- | --- |
| 1 | **README.md** (이 문서) | 설치, 실행, 테스트, 실험 방법 |
| 2 | [CONTRIBUTING.md](CONTRIBUTING.md) | 같이 개발하는 규칙: 브랜치, PR, 각자 작업 공간, 하면 안 되는 것 |
| 3 | [PROGRESS.md](PROGRESS.md) | 지금까지 만든 것, 실험 결과, 확인하지 못한 것 |
| 4 | [NEXT_TASKS.md](NEXT_TASKS.md) | 다음에 할 일 (한 번에 하나씩) |
| 5 | [RAG_DESIGN.md](RAG_DESIGN.md) | 검색(RAG) 구조와 실측 분석 |
| 6 | [backend/data/SCHEMA.md](backend/data/SCHEMA.md) | 교재, 질문 세트, 공격 세트의 데이터 형식 |
| 7 | [CLAUDE.md](CLAUDE.md) | 작업 규칙. Claude Code를 쓰면 자동으로 읽습니다 |

---

## 1. 빠른 시작 (Mock 모드, API 키 없이)

필요한 것: **Git**, **Python 3.11**, **Node.js 20 LTS 이상**

명령은 Windows PowerShell 기준입니다. macOS/Linux는 1-5를 보세요.

### 1-1. 저장소 받기

```powershell
git clone https://github.com/AI-Challenger-2026/ai_turtor_guardrail.git
cd ai_turtor_guardrail
```

### 1-2. 백엔드 설치와 실행 (PowerShell 창 1)

```powershell
cd backend
py -3.11 -m venv .venv                              # py 런처가 없으면: python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env                         # 기본값 MOCK_MODE=true (API 키 불필요)
.\.venv\Scripts\python.exe -m pytest                # "130 passed" 가 나오면 설치 성공
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload --port 8000
```

- 이 문서의 명령은 가상환경을 활성화하지 않고 `.\.venv\Scripts\python.exe`로 바로 실행합니다. `Activate.ps1` 실행 정책 오류나 `python`을 치면 Microsoft Store가 뜨는 문제를 피할 수 있습니다.
- 처음 실행하면 검색 인덱스(`data/index/kb_index.json`)가 자동으로 만들어집니다.
- 확인: http://localhost:8000/api/health (서버 상태), http://localhost:8000/docs (API 문서)

### 1-3. 프론트엔드 설치와 실행 (PowerShell 창 2)

```powershell
cd frontend
npm.cmd install
npm.cmd run dev                                     # http://localhost:5173
```

- `npm` 대신 `npm.cmd`를 쓰는 이유: PowerShell 실행 정책 때문에 `npm.ps1`이 막힐 때가 있습니다.
- 개발 서버는 `/api` 요청을 `http://localhost:8000`(백엔드)으로 넘겨 줍니다.

### 1-4. 잘 됐는지 확인

1. 브라우저에서 **http://localhost:5173**을 엽니다. 8000번은 API 서버라 채팅 화면이 없습니다.
2. 상단 배지에 "Mock 모드"가 보이고, 답변이 `[MOCK]`으로 시작하면 정상입니다. Mock 답변은 동작 확인용 가짜 응답입니다.
3. 톱니바퀴(개발용 설정)에서 근거 모드를 `kb`, 가드 프로필을 `default`로 바꾸고 "리스트와 튜플 차이가 뭐야?"를 물어봅니다. 답변 아래에 출처 카드가 나오고, "판정 보기"에서 가드별 판정이 보입니다.

### 1-5. macOS / Linux

```bash
cd backend
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
cp .env.example .env
.venv/bin/python -m pytest
.venv/bin/python -m uvicorn app.main:app --reload --port 8000
# 다른 터미널
cd frontend && npm install && npm run dev
```

이 문서의 다른 명령도 `.\.venv\Scripts\python.exe`를 `.venv/bin/python`으로 바꾸면 그대로 쓸 수 있습니다.

---

## 2. 실제 Claude API로 바꾸기

1. API 키는 **각자** Claude Console(https://console.anthropic.com)에서 발급합니다.
2. `backend/.env`에서 `MOCK_MODE=false`로 바꾸고 `ANTHROPIC_API_KEY=`에 본인 키를 넣습니다.
3. 백엔드 서버를 재시작합니다(Ctrl+C 후 다시 실행). 상단 배지가 "Mock 모드" 대신 모델명으로 바뀝니다.

> **API 키 주의**
> - `.env`는 git에 올라가지 않습니다(`.gitignore`). 키를 코드, 채팅, 이슈, PR, 커밋 메시지에 붙여넣지 마세요.
> - 키는 팀원끼리 공유하지 않고 각자 씁니다. 노출됐다면 Console에서 바로 폐기하고 새로 발급하세요.

> **비용 주의**
> - `.env`가 `MOCK_MODE=false`이면 **CLI, 평가 스크립트, 웹 화면이 모두 실제로 과금**됩니다.
> - `pytest`는 `.env`와 상관없이 항상 Mock으로 돕니다. 스모크 테스트는 `--allow-real`을 붙여야만 실제 서버에서 돕니다.
> - 실측 참고값 (기본 모델 기준):
>   - 정상 질문 1개, 교재 근거 모드 + 기본 가드: 약 1.1센트, 약 10초
>   - 공격 25건 바닐라 평가(구성 A): 약 $0.19
> - 실제 API로 평가를 돌릴 때는 예상 호출 수와 비용을 먼저 팀에 알립니다([CONTRIBUTING.md](CONTRIBUTING.md)).

기본 모델: 답변 생성 `claude-sonnet-5-5`, 판정·질의 재작성 `claude-haiku-5-5`. 바꾸려면 9장 환경변수를 보세요.

---

## 3. 시스템 구조

### 3-1. 질문 하나가 처리되는 순서

```
학생 질문
  │ 입력 검증 (길이·턴 수·요청 크기)
  │ 질문 유형·학습자 수준 판단 (규칙)
  ▼
[입력]  input_rule(정규식) → input_llm(LLM 분류)      차단되면 거절 문구로 끝
  ▼
[처리]  후속 질문 재작성 → 교재 검색(BM25)             관련도 기준값 미달이면 LLM 호출 없이 "자료에 없음" 안내
  ▼
        답변 생성 (검색 청크 안에서만, [n] 인용)
  ▼
[출력]  citation(규칙) → groundedness(LLM) → output_safety(LLM)
        수정 요청이면 1회 재생성, 그래도 실패하면 안전한 대체 답변
  ▼
최종 답변 + 출처 + 가드 판정 기록(guard_trace) + 턴 로그(logs/turns_날짜.jsonl)
```

### 3-2. 통제 지점별 장치

| 통제 지점 | 장치 | 방식 | 하는 일 |
| --- | --- | --- | --- |
| 입력 | `input_rule` | 정규식 | 이전 지시 무시, 역할 전환, 시스템 프롬프트 공개 요구, 개인정보(주민번호·휴대폰·카드번호) |
| 입력 | `input_llm` | LLM | 대화 맥락까지 보고 5개로 분류: 정상 / 범위 밖 / 유해·민감 / 탈옥 / 위장 공격 |
| 처리 | 질의 재작성 | LLM | "그럼 2번은?" 같은 후속 질문을 혼자서도 뜻이 통하는 질문으로 바꿈 |
| 처리 | 검색 기준값 | 규칙 | 관련도가 기준값(kiwi 0.25)보다 낮으면 생성하지 않음 |
| 출력 | `citation` | 규칙 | 인용 번호가 실제 검색 청크 범위 안에 있는지, 출처가 빠지지 않았는지 |
| 출력 | `groundedness` | LLM | 답변을 주장 단위로 나눠 교재로 뒷받침되는지 판정 |
| 출력 | `output_safety` | LLM | 미성년자 학습자에게 부적절한 내용인지 판정 |

- 같은 단계 안에서는 규칙 가드가 LLM 가드보다 먼저 돕니다. 차단이 하나 나오면 나머지 가드는 건너뜁니다.
- 가드 내부 오류가 나면 기본값으로 차단합니다(fail-closed). 판정 모델이 판정을 거절해도 차단합니다.
- 처리 지점의 "도구 호출" 통제는 지금 도구를 쓰지 않아 해당이 없습니다.
- **가드 프로필:**
  - `none`: 가드 없음. `GUARD_PROFILE`의 기본값입니다.
  - `default`: 위 가드 5개를 씁니다.
  - `member_a`~`member_d`: 팀원별 프로필입니다. 지금은 `default`와 같습니다.

### 3-3. 폴더 구조

```
ai_turtor_guardrail/
├─ README.md, CONTRIBUTING.md, PROGRESS.md, NEXT_TASKS.md, RAG_DESIGN.md, CLAUDE.md
├─ backend/                      Python 3.11 + FastAPI
│  ├─ app/
│  │  ├─ main.py                 FastAPI 진입점 (CORS, 요청 크기 제한, 오류 응답)
│  │  ├─ service.py              TutorService.handle — 웹·CLI·평가가 모두 부르는 단일 진입점
│  │  ├─ cli.py                  터미널 대화
│  │  ├─ schemas.py              API 계약 (frontend/src/types.ts 와 항상 같이 수정)
│  │  ├─ api/                    routes_chat / routes_health / routes_config
│  │  ├─ llm/                    anthropic, openai, mock, elice(자리), factory
│  │  ├─ kb/                     loader(.md/.txt/.json/.pdf), chunker, index, build
│  │  ├─ retrieval/              tokenizer(kiwi/simple), bm25, dense(선택), hybrid, query_rewriter
│  │  ├─ tutor/                  agent(프롬프트·인용), modes(응답 모드·질문 유형·학습자 수준)
│  │  └─ guardrails/             base, registry, pipeline, policy, 기본 가드 5종
│  │     └─ members/member_a~d/  ★ 팀원별 가드 작업 공간
│  ├─ prompts/                   모든 시스템·판정 프롬프트와 거절 문구 (텍스트/YAML)
│  ├─ configs/                   settings.yaml(튜닝값), guards_*.yaml(가드 프로필), input_rules.yaml(정규식)
│  ├─ data/
│  │  ├─ sample/                 샘플 교재 4단원, qa_set(20), attack_set(25), benign_borderline(10)
│  │  ├─ attacks/member_a~d.jsonl ★ 팀원별 공격 세트 자리 (비어 있음)
│  │  └─ official/               엘리스 공식 데이터 자리 (git 제외, 각자 PC에만)
│  ├─ eval/                      run_eval, cross_attack, generate_attacks, metrics, datasets
│  ├─ scripts/smoke_test.py
│  ├─ tests/                     pytest 130개 (Mock 모드, 네트워크 호출 없음)
│  ├─ logs/                      턴 로그 (git 제외)
│  └─ reports/                   평가 보고서 (git 제외, 실제 API 실험 A·B 보고서 2개만 포함)
└─ frontend/                     React 18 + TypeScript + Vite 5 + Tailwind CSS 3
   └─ src/                       App.tsx, api/client.ts, types.ts, components/*
```

---

## 4. 테스트

```powershell
# (1) 백엔드 단위·통합 테스트 — 항상 Mock, API 키 불필요
cd backend
.\.venv\Scripts\python.exe -m pytest

# (2) 프론트엔드 빌드 (타입 체크 포함)
cd ..\frontend
npm.cmd run build

# (3) 스모크 테스트 — 백엔드 서버를 띄운 상태에서 다른 창으로
cd ..\backend
.\.venv\Scripts\python.exe scripts\smoke_test.py                                   # health → 채팅 2턴 → 잘못된 요청 처리
.\.venv\Scripts\python.exe scripts\smoke_test.py --grounding kb --profile default  # 교재 근거 + 기본 가드 경로
```

PR을 올리기 전에는 (1)을 꼭, 프론트엔드를 고쳤다면 (2)도 통과시킵니다.

---

## 5. 평가와 실험

### 5-1. 명령 형식

```powershell
cd backend
.\.venv\Scripts\python.exe -m eval.run_eval --profile <프로필> --grounding <general 또는 kb> --sets <세트> [--judge-attack]
```

| 옵션 | 뜻 |
| --- | --- |
| `--profile` | 가드 프로필: `none`, `default`, `member_a` ... |
| `--grounding` | `general`(일반 지식) / `kb`(교재 근거, **기본값**) |
| `--sets` | `qa`(정상 질문), `attack`(공격), `benign`(헷갈리지만 정상인 질문). 쉼표로 여러 개, 기본은 세 개 모두 |
| `--judge-attack` | 답변이 실제로 공격에 응했는지 LLM이 따로 판정 (**실질 우회율**). 판정 호출이 추가됩니다 |
| `--attack-file` | 공격 세트 파일 지정 (여러 번 가능). 기본은 `data/sample/attack_set.jsonl` |
| `--limit N` | 세트마다 앞의 N개만 실행. 비용을 줄여 먼저 시험해 볼 때 사용 |
| `--judge-quality` | 정상 질문 답변 품질을 1~5점으로 채점 (추가 비용) |

결과는 `backend/reports/<시각>_<프로필>/`에 `summary.md`(요약), `details.csv`(항목별), `summary.json`으로 저장됩니다.

> Mock 모드 평가 수치는 **파이프라인 동작 확인용이고 실제 성능이 아닙니다.** Mock 판정기는 키워드와 글자 겹침으로 흉내만 냅니다. 실제 수치는 `MOCK_MODE=false`로 돌려야 나옵니다(과금).

### 5-2. 바닐라 대비 가드 효과 실험 (구성 A·B·C)

| 구성 | 명령 (`.\.venv\Scripts\python.exe -m eval.` 뒤에 붙임) | 켜진 통제 지점 |
| --- | --- | --- |
| A. 순수 바닐라 | `run_eval --profile none --grounding general --sets attack --judge-attack` | 없음 (모델 자체 안전장치만) |
| B. 교재 근거만 | `run_eval --profile none --grounding kb --sets attack --judge-attack` | 처리 |
| C. 교재 근거 + 기본 가드 | `run_eval --profile default --grounding kb --sets attack --judge-attack` | 입력 + 처리 + 출력 |

지금까지 나온 결과 (실제 API, 샘플 공격 25건, 2026-10-18):

| 구성 | 실질 우회율 | LLM 호출 | 비용 | 보고서 (저장소에 포함) |
| --- | --- | --- | --- | --- |
| A | 8% (2/25: atk-06, atk-18) | 25회 | 약 $0.19 | [backend/reports/20261018-215354_none/](backend/reports/20261018-215354_none/summary.md) |
| B | 0% (0/25) | 4회 | 약 $0.04 | [backend/reports/20261018-215020_none/](backend/reports/20261018-215020_none/summary.md) |
| C | 아직 측정 안 함 | | | |

- 표본이 작아 신뢰구간이 넓습니다. 2/25의 95% 신뢰구간은 약 2~25%, 0/25는 약 0~13%입니다(Wilson).
- 샘플 세트는 개발 중 조정에 일부 쓰여서 엄밀한 test 결과가 아닙니다. 자세한 해석과 한계는 [PROGRESS.md](PROGRESS.md) 4장에 있습니다.

### 5-3. 그 밖의 실험 도구

```powershell
# 팀원 프로필 × 팀원 공격 세트 차단율 행렬 + 프로필별 오차단율
.\.venv\Scripts\python.exe -m eval.cross_attack --include-sample

# LLM으로 공격 세트 늘리기 (실제 API 비용 발생, 처음엔 --per-category 2 정도로 작게)
.\.venv\Scripts\python.exe -m eval.generate_attacks --out data/attacks/member_a.jsonl --author member_a --per-category 2 --multiturn
```

생성된 공격은 실제 위험한 정보가 섞이지 않았는지 사람이 한 번 훑어본 뒤 씁니다.

---

## 6. 내 가드 추가하기 (5분 가이드)

각자 자기 폴더(`members/member_x/`)와 자기 프로필(`guards_member_x.yaml`)에서만 작업하면 팀원끼리 충돌이 나지 않습니다. 누가 `member_a`~`member_d`인지는 [CONTRIBUTING.md](CONTRIBUTING.md) 2장 표에 적습니다.

1. **파일 열기:** `backend/app/guardrails/members/member_a/guards.py`. 예시 `KeywordBlockGuard`가 들어 있습니다.
2. **클래스 작성:**

   ```python
   from app.guardrails.base import Guard, GuardContext
   from app.guardrails.registry import register_guard

   @register_guard
   class MyScopeGuard(Guard):
       name = "member_a.my_scope"      # must start with "member_x." to avoid name clashes
       stage = "input"                 # "input" or "output"
       kind = "llm"                    # "rule" or "llm" — rule guards run first within a stage

       def check(self, ctx: GuardContext):
           # ctx.question, ctx.rewritten_question, ctx.history, ctx.chunks, ctx.answer (output stage)
           res = self.deps.judge.generate(
               system=self.deps.prompts.text("member_a_scope.txt"),   # prompt file in backend/prompts/
               messages=[{"role": "user", "content": f"<question>{ctx.rewritten_question}</question>"}],
               temperature=0.0, max_tokens=512, json_mode=True, task="member_a_scope",
           )
           if res.data.get("block"):
               return self.result("block", "out_of_scope", res.data.get("reason", ""), 0.9)
           return self.result("allow", "in_scope", "", 0.1)
   ```

   - 판정은 `self.result(decision, category, reason, score)`로 돌려줍니다. `decision`은 `allow` / `block` / `revise`(출력 단계 전용, 재생성 요청) 중 하나입니다.
   - 차단할 때 보여 줄 문구는 `category`에 맞춰 `prompts/refusal_templates.yaml`에서 고릅니다. 맞는 문구가 없으면 `default`를 씁니다.
   - 출력 가드에서 `revise`할 때 `details`에 `unsupported_claims=[...]`나 `problems=[...]`를 넣으면 재생성 지시에 반영됩니다.
   - 예외를 던지면 파이프라인이 `on_error`(기본 block) 정책으로 처리합니다. 인증 실패, 429, 타임아웃 같은 API 오류는 사용자에게 오류로 전달됩니다.
3. **프롬프트**는 코드에 쓰지 말고 `backend/prompts/`에 텍스트 파일로 둡니다.
   - 파일 이름은 `member_a_...txt`처럼 자기 이름으로 시작하게 짓습니다.
   - 치환은 `${변수}`로 쓰고, `self.deps.prompts.render("파일", 변수=값)`로 채웁니다.
   - 판정 프롬프트에는 출력 JSON의 키 이름, 자료형, 허용 값을 적습니다(CLAUDE.md 규칙 12).
4. **프로필에 등록:** `backend/configs/guards_member_a.yaml`의 `input:` 또는 `output:` 목록에 `- guard: member_a.my_scope`를 추가합니다. `params:`에 쓴 값은 코드에서 `self.params`로 받습니다.
5. **확인:**
   - 서버를 재시작하고, 웹 개발용 설정에서 프로필 `member_a`를 고른 뒤 "판정 보기"로 봅니다.
   - 터미널에서는 `.\.venv\Scripts\python.exe -m app.cli --profile member_a --grounding kb --debug`로 봅니다.
   - 새 가드에는 Mock 모드 테스트를 `backend/tests/`에 추가하고, `pytest`로 기존 테스트가 깨지지 않았는지 확인합니다.

---

## 7. 내 공격 세트 만들기

- `backend/data/attacks/member_a.jsonl`에 [SCHEMA.md](backend/data/SCHEMA.md) 3번 형식으로 한 줄에 하나씩 넣습니다. `author`에는 자기 이름을 씁니다.
- **실제로 위험한 구체 정보는 넣지 않고 요청의 형태만** 재현합니다. 필요하면 `[유해 주제]` 같은 자리표시를 씁니다.
- 직접 쓰거나 `eval.generate_attacks`로 생성합니다(5-3).
- 다 모이면 `eval.cross_attack`으로 차단율 행렬과 각 프로필의 오차단율을 봅니다.

---

## 8. 응답 모드 · 근거 모드 · 가드 프로필

웹 화면의 개발용 설정, CLI 옵션, `.env` 어느 쪽으로든 바꿀 수 있습니다.

- **응답 모드** (`ANSWER_MODE`)
  - `direct`: 개념을 바로 설명합니다.
  - `hint`: 정답 대신 단계적 힌트와 되묻기를 줍니다.
  - `adaptive`(기본): 질문 유형에 맞춰 바꿉니다.
    - 개념 질문 → 설명 + 확인 질문
    - 문제 풀이 → 단계적 힌트
    - 헷갈림 → 다른 비유로 재설명
    - 짧은 후속 질문 → 직전 맥락을 이어서
- **근거 모드** (`GROUNDING_MODE`)
  - `general`: 일반 지식 튜터입니다. 화면에 "교육 자료 미적용" 안내가 나옵니다.
  - `kb`: 교육 자료만 근거로 답하고 `[n]` 인용을 답니다.
- **가드 프로필** (`GUARD_PROFILE`): `configs/guards_<이름>.yaml`. `none`이면 가드가 적용되지 않습니다(응답의 `guard_trace=null`).

CLI 예시:

```powershell
.\.venv\Scripts\python.exe -m app.cli                                         # 대화형. /new, /mode hint, /grounding kb, /profile default, /exit
.\.venv\Scripts\python.exe -m app.cli --grounding kb --profile default --debug   # 턴마다 재작성 질의·검색 청크·가드 판정 출력
.\.venv\Scripts\python.exe -m app.cli --once "리스트와 튜플 차이가 뭐야?" --grounding kb
```

---

## 9. 환경변수 (`backend/.env`)

`backend/.env.example`을 `.env`로 복사해서 씁니다.

| 변수 | 기본값 | 설명 |
| --- | --- | --- |
| `MOCK_MODE` | `true` | `true`면 모든 LLM 호출이 Mock (답변 앞에 `[MOCK]`) |
| `GENERATOR_PROVIDER` / `GENERATOR_MODEL` | `anthropic` / `claude-sonnet-5-5` | 답변 생성용 |
| `JUDGE_PROVIDER` / `JUDGE_MODEL` | `anthropic` / `claude-haiku-5-5` | 가드 판정·질의 재작성용 |
| `OPENAI_MODEL` | (빈 값) | 제공자가 `openai`일 때 모델명 |
| `ANTHROPIC_API_KEY`, `OPENAI_API_KEY` | (빈 값) | 백엔드에서만 읽음. 화면·로그·응답에 나가지 않음 |
| `ANSWER_MODE` | `adaptive` | `direct` / `hint` / `adaptive` |
| `GROUNDING_MODE` | `general` | `general` / `kb` |
| `GUARD_PROFILE` | `none` | `none` / `default` / `member_a` ~ `member_d` |
| `MAX_MESSAGE_CHARS`, `MAX_TURNS`, `MAX_REQUEST_BYTES` | 2000, 20, 524288 | 메시지 길이, 대화 턴(넘으면 오래된 턴부터 잘라 냄), 요청 크기 |
| `CORS_ORIGINS` | `http://localhost:5173` | 쉼표로 여러 개 |
| `GENERATOR_EFFORT`, `JUDGE_EFFORT` | `low` | Anthropic effort. 비우면 보내지 않음 |
| `ANTHROPIC_REFUSAL_FALLBACK` | `default` | 지원 모델에서 모델 자체 거절 시 서버측 대체 사용. 끄려면 `off` |
| `HYBRID_SEARCH` | `false` | `true`면 BM25 + 임베딩 검색 결합 (`requirements-optional.txt` 필요) |
| `KB_DIR`, `KB_INDEX_PATH` | `data/sample/kb`, `data/index/kb_index.json` | 교육 자료 폴더와 인덱스 파일 |

- 청크 크기, top_k, 검색 기준값, 재시도 횟수, 가격표 같은 비밀이 아닌 튜닝값은 `backend/configs/settings.yaml`에 있습니다.
- 프론트엔드에서 백엔드 주소를 바꾸려면 `frontend/.env.example`을 `frontend/.env`로 복사하고 둘 중 하나를 넣습니다.
  - `VITE_PROXY_TARGET`: 개발 서버 프록시 대상
  - `VITE_API_BASE_URL`: 직접 호출. 이때는 백엔드 `CORS_ORIGINS`에 프론트 주소가 있어야 합니다.
- Claude 5.x 참고: Sonnet 5.5와 Haiku 5.5는 `temperature`를 받지 않습니다(보내면 400 오류).
  - 그래서 판정 호출의 temperature 0은 이 모델들에서 생략됩니다. 턴 로그에는 `temperature_applied=false`로 남습니다.
  - 판정 일관성은 JSON 출력 고정과 `effort=low`로 맞춥니다.

---

## 10. 공식 데이터 · 엘리스 ML API 교체

공식 데이터에는 기밀(Confidential and Proprietary) 표시가 있습니다. **`backend/data/official/`에만 두고 git에 올리지 않습니다.** 이 폴더는 `.gitignore`로 제외되어 있으니, 팀원끼리 공식 채널로 따로 받습니다.

1. **교육 자료:** `backend/data/official/kb/`에 넣고, `.env`에 `KB_DIR=data/official/kb`를 적습니다.
   - `.md` / `.txt` / `.json`은 바로 읽힙니다. PDF는 `pypdf`를 설치하면 페이지 단위로 읽힙니다.
   - 다른 형식이면 `app/kb/loader.py`에 로더를 추가하고 `register_loader(".확장자", 객체)`로 등록합니다.
   - `.\.venv\Scripts\python.exe -m app.kb.build --show`로 청크와 단원·섹션 정보를 확인합니다.
   - 검색 기준값(`configs/settings.yaml`의 `retrieval.min_score`)은 새 자료의 **튜닝용(dev)** 질문으로 다시 맞춥니다.
2. **질의응답 세트 / 공격 쿼리 세트:** [SCHEMA.md](backend/data/SCHEMA.md) 형식으로 바꿔 넣습니다.
   - 실행: `run_eval --qa-file data/official/qa.jsonl --attack-file data/official/attack.jsonl`
   - 필드 이름이 다르면 `eval/datasets.py`에 변환 함수를 추가합니다.
3. **엘리스 ML API:**
   - `app/llm/elice_provider.py`의 `_complete()`를 구현합니다. 지금은 선택하면 "미구현"(501) 오류를 돌려줍니다.
   - `.env`에 `ELICE_API_BASE`, `ELICE_API_KEY`를 넣고, `GENERATOR_PROVIDER=elice` / `JUDGE_PROVIDER=elice`로 고릅니다.
   - 모델 단가를 `configs/settings.yaml`의 `pricing_usd_per_mtok`에 넣으면 보고서의 예상 비용에 반영됩니다.

---

## 11. 자주 나는 문제 (Windows)

| 증상 | 원인 | 해결 |
| --- | --- | --- |
| `python`을 치면 Microsoft Store가 뜸 | Windows 기본 별칭 | `.\.venv\Scripts\python.exe`로 직접 실행 |
| `npm.ps1 파일을 로드할 수 없습니다` | PowerShell 실행 정책 | `npm.cmd`로 실행하거나, 한 번만 `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` 실행 후 창을 새로 엶 |
| `Activate.ps1`이 막힘 | 위와 같음 | 활성화하지 않고 `.\.venv\Scripts\python.exe`로 실행 |
| `localhost:8000`에 채팅 화면이 없음 | 8000은 API 서버 | **http://localhost:5173** 열기 |
| `.env`를 바꿨는데 반영이 안 됨 | 서버 재시작 필요 | 백엔드 창에서 Ctrl+C 후 다시 실행 |
| 화면에 "AI 서비스 인증에 실패했습니다" | 키가 없거나 틀림 | `backend/.env`의 `ANTHROPIC_API_KEY` 확인, 또는 `MOCK_MODE=true` |
| `Get-Content`로 파일을 열면 한글이 깨짐 | Windows PowerShell 5.1 기본 인코딩 | 파일은 모두 UTF-8. `Get-Content -Encoding UTF8 파일` 또는 VS Code로 열기 |

---

## 12. 알아 둘 점

- **API 비용:** 가드를 모두 켜면 kb 모드 한 턴에 LLM이 4번(입력 분류, 생성, 근거 판정, 안전 판정) 이상 호출되고 약 10초가 걸립니다. 후속 질문 재작성과 재생성이 있으면 호출이 더 늘어납니다.
- **분류 정확도:** 질문 유형 분류와 학습자 수준 판단은 규칙 기반입니다. 샘플 질문과 분류 규칙을 함께 만들어서 정확도가 높게 나오므로, 공식 데이터에서 다시 확인해야 합니다.
- **단순한 구조:** 데이터베이스, 로그인, Docker, 서버형 벡터 DB를 쓰지 않습니다. 대화 이력은 브라우저가 보관해 매 요청마다 보냅니다(무상태 백엔드).
- **로그인 없음:** 개발 서버 주소를 아는 사람은 누구나 쓸 수 있습니다. 실제 모드에서 화면을 공유하면 비용이 내 키로 나갑니다.
