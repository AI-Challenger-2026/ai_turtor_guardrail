# 같이 개발하는 법

이 저장소에서 팀원끼리 충돌 없이 작업하기 위한 규칙입니다. 설치와 실행은 [README.md](README.md), 전체 작업 규칙의 원문은 [CLAUDE.md](CLAUDE.md)에 있습니다.

---

## 1. 처음 한 번만

1. [README.md](README.md) 1장대로 설치합니다. `pytest`에서 **130 passed**가 나오면 준비가 끝난 것입니다.
   - 조직 멤버는 기본으로 **읽기 권한**만 있습니다. 브랜치를 push하려면 저장소 관리자에게 쓰기 권한을 요청하세요.
2. git 작성자 정보를 설정합니다. 커밋에 이 이름이 남습니다.
   ```powershell
   git config --global user.name "내 이름"
   git config --global user.email "내 이메일"
   ```
3. `backend/.env`는 각자 만듭니다(`Copy-Item .env.example .env`). 실제 API를 쓸 때는 **본인 키**를 넣습니다.

---

## 2. 각자 작업 공간

자기 슬롯의 파일만 고치면 다른 팀원과 충돌이 나지 않습니다. **담당 칸은 팀에서 정해 채워 주세요.**

| 슬롯 | 담당 | 가드 코드 | 가드 프로필 | 공격 세트 | 프롬프트 |
| --- | --- | --- | --- | --- | --- |
| `member_a` | (이름) | `backend/app/guardrails/members/member_a/guards.py` | `backend/configs/guards_member_a.yaml` | `backend/data/attacks/member_a.jsonl` | `backend/prompts/member_a_*.txt` |
| `member_b` | (이름) | `.../members/member_b/guards.py` | `.../guards_member_b.yaml` | `.../attacks/member_b.jsonl` | `.../member_b_*.txt` |
| `member_c` | (이름) | `.../members/member_c/guards.py` | `.../guards_member_c.yaml` | `.../attacks/member_c.jsonl` | `.../member_c_*.txt` |
| `member_d` | (이름) | `.../members/member_d/guards.py` | `.../guards_member_d.yaml` | `.../attacks/member_d.jsonl` | `.../member_d_*.txt` |

- 가드 이름은 `member_a.my_guard`처럼 슬롯 이름으로 시작합니다. 만드는 방법은 [README.md](README.md) 6장에 있습니다.
- 새 가드의 테스트는 `backend/tests/test_member_a.py`처럼 슬롯 이름을 붙인 파일에 둡니다.

**공용 파일**은 모두에게 영향을 줍니다. 고치기 전에 팀 채팅에 먼저 알리고, PR 설명에 이유를 적어 주세요.

| 공용 파일 | 영향 |
| --- | --- |
| `backend/app/service.py` | 한 턴 처리 흐름 전체 |
| `backend/app/schemas.py` ↔ `frontend/src/types.ts` | API 계약. **둘은 항상 같이 고칩니다** |
| `backend/configs/settings.yaml`, `guards_default.yaml`, `input_rules.yaml` | 기본 동작과 기본 가드 |
| `backend/prompts/` (슬롯 이름이 없는 파일) | 튜터와 기본 가드의 프롬프트 |
| `backend/eval/` | 모든 실험의 지표 계산 |
| `backend/data/sample/` | 공용 평가 데이터. 내용을 고치면 이전 결과와 비교할 수 없게 됩니다 |

---

## 3. 작업 흐름: 브랜치 → PR → 병합

`main`에는 직접 push하지 않습니다. 작업마다 브랜치를 만들고 PR로 합칩니다.

```powershell
git switch main
git pull                                   # 최신 main 받기
git switch -c <이름>/<작업>                # 예: chanwoo/config-c-eval

# ... 작업 ...

cd backend
.\.venv\Scripts\python.exe -m pytest       # 전체 테스트 통과 확인
cd ..
git status                                 # 올라갈 파일 확인 (.env, 로그, 공식 데이터가 없어야 함)
git add <고친 파일들>
git commit -m "무엇을 왜 바꿨는지 한 줄로"
git push -u origin <이름>/<작업>
```

1. GitHub에서 **Pull request**를 만들고, 아래 4장 체크리스트를 PR 설명에 붙입니다.
2. 팀원 1명 이상이 보고 승인하면 `main`에 병합합니다.
3. 병합한 뒤에는 각자 `git switch main`, `git pull`로 최신 상태를 받습니다.

- **PR 하나에 작업 하나만** 담습니다([NEXT_TASKS.md](NEXT_TASKS.md)의 한 항목).
- 무엇이 올라가는지 보려면 `git add .`보다 파일을 하나씩 지정하는 편이 안전합니다.

---

## 4. PR 체크리스트

```markdown
- [ ] 백엔드 전체 테스트 통과 (`python -m pytest`). 새 기능에는 Mock 모드 테스트 추가, 네트워크 호출 없음
- [ ] 프론트엔드를 고쳤다면 `npm.cmd run build` 통과
- [ ] `schemas.py`를 고쳤다면 `frontend/src/types.ts`도 같이 고침
- [ ] 가드를 추가·변경했다면 통제 지점(입력 / 처리 / 출력)을 PR 설명에 적음
- [ ] 변경 파일에 API 키, 공식 데이터, 로그 파일이 없음
- [ ] 실제 API를 썼다면 호출 수와 비용을 적음
- [ ] 평가 결과를 적었다면 표본 크기와 한계(신뢰구간)를 같이 적음
- [ ] PROGRESS.md 갱신 (바뀐 것, 확인하지 못한 것)
- [ ] 새로 쓴 코드 주석은 영어
```

---

## 5. 꼭 지킬 규칙

[CLAUDE.md](CLAUDE.md)의 작업 규칙 중 팀원이 특히 자주 마주치는 것들입니다.

| 규칙 | 이유 |
| --- | --- |
| **API 키는 각자 `backend/.env`에만** 둡니다. 코드, 로그, 테스트, 커밋, 채팅에 넣지 않습니다 | 저장소나 채팅에 남은 키는 노출된 것으로 봐야 합니다 |
| **실제 API를 부르는 명령은 실행 전에** 예상 호출 수와 비용을 팀에 알립니다 | 평가 한 번에도 수십~수백 번 호출됩니다 |
| **평가용(test) 데이터를 보고** 프롬프트, 규칙, 기준값을 고치지 않습니다. 튜닝은 튜닝용(dev) 데이터로만 합니다 | test 결과가 실제보다 좋게 나옵니다. dev/test 분리는 아직 안 되어 있습니다([NEXT_TASKS.md](NEXT_TASKS.md) 1번) |
| **공격 데이터에는 요청의 형태만** 넣습니다. 실제 위험한 구체 정보 대신 `[유해 주제]` 같은 자리표시를 씁니다 | 저장소 자체가 위험한 자료가 되면 안 됩니다 |
| **엘리스 공식 데이터는 `backend/data/official/`에만** 둡니다. git과 공개 채널에 올리지 않습니다 | 기밀(Confidential and Proprietary) 자료입니다 |
| **확인하지 못한 것은 확인했다고 쓰지 않고** PROGRESS.md "확인하지 못한 것"에 적습니다 | 발표 수치의 신뢰도가 걸려 있습니다 |
| 판정 프롬프트에는 출력 JSON의 **키 이름, 자료형, 허용 값**을 적습니다 | 파싱 실패와 판정 흔들림을 줄입니다 |
| 프롬프트는 코드에 쓰지 않고 `backend/prompts/`에 파일로 둡니다 | 코드를 고치지 않고도 프롬프트를 비교할 수 있습니다 |
| 관계형 DB, Docker, 로그인, 서버형 벡터 DB는 추가하지 않습니다 | 대회 기간에 맞춘 단순한 구조를 유지합니다 |

---

## 6. Claude Code로 작업할 때

- 저장소 폴더에서 Claude Code를 열면 [CLAUDE.md](CLAUDE.md)를 자동으로 읽고 규칙을 따릅니다.
- 코드를 고치기 전에 작업 계획을 먼저 보여 주고 확인을 받습니다. 계획을 읽고 승인한 뒤 진행하세요.
- 실제 API를 부르기 전에는 예상 호출 수와 비용을 알려 주고 승인을 받습니다.
- 끝나면 PROGRESS.md를 갱신하고, 바뀐 파일, 테스트 결과, 확인하지 못한 것을 보고합니다. PR 설명에 그대로 옮겨 쓰면 됩니다.

---

## 7. 실수했을 때

- **API 키를 커밋하거나 채팅에 올렸을 때:**
  1. 바로 Claude Console에서 그 키를 폐기하고 새로 발급합니다.
  2. 팀에 알립니다.
  - 커밋을 지워도 이미 push된 키는 노출된 것으로 봅니다.
- **공식 데이터를 올렸을 때:** 저장소 관리자에게 바로 알립니다. 커밋 기록에서 지우는 작업이 필요합니다.
- **`main`에 실수로 push했을 때:** 강제 push(`--force`)로 덮지 말고, 팀에 알린 뒤 되돌리는 커밋(`git revert`)으로 고칩니다.
