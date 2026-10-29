# 데이터 스키마 (Data Schema)

공식 데이터(교육 자료, 질의응답 세트, 공격 쿼리 세트)가 오면 아래 스키마에 맞춰 변환해 같은 경로 구조에 넣으면
코드 수정 없이 인덱스 빌드와 평가가 돌아간다. 형식이 다르면 `app/kb/loader.py` 에 로더를 추가하거나
`eval/datasets.py` 의 변환 함수를 수정한다.

모든 파일은 **UTF-8** 로 저장한다 (Windows 메모장의 ANSI/cp949 저장 금지).

---

## 1. 교육 자료 (knowledge base) — `data/sample/kb/` (또는 `KB_DIR`)

지원 형식: `.md`, `.txt`, `.json`, `.pdf`(선택, `pypdf` 필요). 하위 폴더도 읽는다.

### 1-1. Markdown (권장)

```markdown
---
doc_id: py101-u1          # 문서 고유 ID (생략 시 파일명)
title: 파이썬 입문         # 문서명 (출처 카드의 첫 항목)
unit: 1단원. 변수와 자료형  # 단원 (생략 시 # 제목)
---
# 1단원. 변수와 자료형
## 1.1 변수란 무엇인가      ← 섹션
본문 ...
### 변수 이름 규칙          ← 하위 섹션 → 섹션 경로 "1.1 변수란 무엇인가 > 변수 이름 규칙"
```

- `#` 하나가 여러 개면 각각 새 단원으로 본다 (한 파일에 여러 단원 가능).
- 코드 블록(```) 안의 `#` 은 heading 으로 보지 않는다.

### 1-2. JSON

중첩형:
```json
{"doc_id": "d1", "title": "교재명",
 "units": [{"unit": "1단원", "sections": [{"section": "1.1 ...", "text": "본문"}]}]}
```
평탄형 (레코드 리스트):
```json
[{"doc_id": "d1", "doc_title": "교재명", "unit": "1단원", "section": "1.1 ...", "text": "본문"}]
```

### 1-3. 청크 메타데이터 (인덱스 파일 `data/index/kb_index.json`)

| 필드 | 설명 |
| --- | --- |
| chunk_id | `<doc_id>#<순번 3자리>` |
| doc_id, doc_title | 문서 ID, 문서명 |
| unit, section | 단원, 섹션 경로 |
| text | 청크 원문 (출처 카드에 그대로 표시) |

인덱스 재빌드: `python -m app.kb.build` (옵션: `--kb-dir`, `--out`, `--show`)

---

## 2. 정상 질문 세트 — `data/sample/qa_set.jsonl`

한 줄에 JSON 객체 하나.

| 필드 | 타입 | 설명 |
| --- | --- | --- |
| id | str | 고유 ID |
| history | list[{role, content}] | 이전 턴 (`role`: user / assistant). 없으면 `[]` |
| question | str | 이번 질문 |
| q_type | str | `concept` / `exercise` / `confusion` / `followup` |
| expected_units | list[str] | 정답 근거가 있는 단원명 (`unit` 값과 정확히 같게). 검색 적중률(hit@k) 계산에 사용 |
| key_points | list[str] | 좋은 답변에 들어가야 할 핵심 내용 (품질 채점 참고용) |

`followup` 유형은 history 없이는 뜻이 통하지 않게 만든다.

## 3. 공격 질문 세트 — `data/sample/attack_set.jsonl`, `data/attacks/member_*.jsonl`

| 필드 | 타입 | 설명 |
| --- | --- | --- |
| id | str | 고유 ID |
| history | list[{role, content}] | 다턴 공격의 이전 턴 |
| prompt | str | 이번 공격 입력 |
| category | str | `out_of_scope` / `harmful_sensitive` / `jailbreak` / `disguised_attack` |
| expected_decision | str | 항상 `"block"` |
| author | str | 작성자 (sample, member_a ...) |

가드레일 테스트용이므로 실제로 위험한 구체 정보는 넣지 않고 요청의 형태만 재현한다.

## 4. 경계 정상 질문 세트 — `data/sample/benign_borderline.jsonl`

겉보기엔 의심스럽지만 정상인 학습 질문. 과잉 차단(오차단) 측정용.

| 필드 | 타입 | 설명 |
| --- | --- | --- |
| id | str | 고유 ID |
| history | list | 이전 턴 |
| prompt | str | 질문 |
| note | str | 왜 의심스러워 보이는지 |
| expected_units | list[str] | (선택) 근거 단원 |
| expected_decision | str | 항상 `"allow"` |

---

## 5. 평가 결과 — `reports/<YYYYMMDD-HHMMSS>_<profile>/`

- `summary.md`: 지표 요약 (정상 질문: hit@k, 출처 표기율, 근거 충실도, 오차단율 / 공격: 카테고리별 차단율·공격 성공률 / 공통: 지연시간, 토큰, 예상 비용)
- `details.csv`: 항목별 결과 (UTF-8 BOM 포함 — Excel 에서 한글이 깨지지 않게)
- `cross_attack.py` 결과: `reports/<시각>_cross_attack/summary.md`, `matrix.csv`
