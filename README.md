# AI 가드레일 튜터

`SPEC.md`에 맞춘 `/tutor`, `/guard` 서버를 단계적으로 구현한다. 승인 계획은 `Plan.md`, 실제 완료·검증 결과는 `Progress.md`를 기준으로 확인한다.

## 현재 범위

현재 코드는 `IMP-001-7`의 런박스 실행·인계 도구까지 구현했다. 서버에는 인증·입출력 계약·공용 API 연결·요청 시간 제한·구조화 응답 호환성 검사, KB·표 로더와 결합 검색, 공통 정책과 입력 분류, 후보 답변의 근거·안전 판정, 자체 가드 튜터를 연결했다. 로컬 검증과 런박스 환경·재시작·터널을 확인하는 도구도 제공한다.

`/guard`는 입력과 후보 출력을 병렬로 판정한다. `/tutor`는 자체적으로 요청 의도를 판정하고, 생성 답변을 독립 검증한 뒤 반환한다. 보호 없이 비교 기준 답변을 생성하던 우회 경로는 제거했다.

초기 4~7단계 구현에서는 검증 실행을 보류했다. 이후 주최측 원본 공개 평가 400건에서 총점 73.52·튜터 65.55·가드 81.48을 측정했다. 사용자는 런박스 기동·제출 채점 성공과 총점 63.30·튜터 51.20·가드 75.41을 확인했다. 개선 1단계에서는 원본 판정 설정과 문항별 계측으로 총점 75.72·튜터 69.69·가드 81.74의 새 공개 기준선을 확보했다. 서버 코드를 유지한 기준 측정이며 점수 변동을 개선 성과로 간주하지 않는다. 현재는 개선 2단계의 시간 예산·검증 실패 처리를 작업하고 있으며 최신 실행·완료 범위는 `Progress.md`에서 확인한다. 별도 부하·재현성·전체 런박스 도구 검증까지 완료한 것으로 해석하지 않는다.

`/healthz`의 `ready`는 실행 준비 여부다. `input_policy_ready`, `output_policy_ready`, `tutor_policy_ready`, `policy_ready`, `retrieval_ready`와 `implementation_stage`를 함께 확인한다. 준비 상태는 구성 요소의 생성·연결 여부이며 정확도·지연 목표의 검증 통과를 의미하지 않는다. `tutor_policy`에는 본문 없이 보호·독립 검증·보정·실패 횟수만 포함한다. `instance_id`는 서버 기동 때마다 새로 생성하는 메타데이터이며 검사 도중 재시작을 확인하는 데 사용한다.

## 환경변수와 실행

로컬 Python 3.11.9, 런박스 Python 3.10.14를 기준으로 한다. 주요 패키지는 `requirements-runbox.txt`에 고정되어 있다. 호환되는 패키지가 이미 설치되어 있다면 다시 설치할 필요가 없다.

서버와 검사 도구는 **프로세스 환경변수만** 사용한다. `.env`, `submission.json`, 자격 증명 파일을 자동으로 읽지 않는다. 사용자가 환경변수를 등록한 같은 PowerShell에서 실행하면 자식 Python과 검증 서버가 해당 환경을 물려받는다.

| 변수 | 의미 |
|---|---|
| `TEAM_TOKEN` | 서버 Bearer 인증 토큰. API 키와 다른 공백 없는 영문·숫자·기호 |
| `OPENAI_BASE_URL`, `OPENAI_API_KEY` | 운영진 채팅 API 주소와 키 |
| `OPENAI_MODEL` | `gpt-5-nano` |
| `EMBEDDING_BASE_URL`, `EMBEDDING_API_KEY` | 운영진 임베딩 API 주소와 키 |
| `EMBEDDING_MODEL` | `text-embedding-3-small` |

주소는 제공받은 HTTPS `mlapi.run`의 `/v1` 주소를 사용한다. 변수 누락·잘못된 모델·주소는 값 대신 변수 이름만 표시하고 시작을 중단한다.

프로젝트 루트로 이동하고 서버를 실행한다.

```powershell
Set-Location -LiteralPath "C:\Users\ngw12\Desktop\Project\ai_turtor_guardrail"
.\.venv\Scripts\python.exe -m uvicorn server:app --host 127.0.0.1 --port 8000 --no-access-log
```

시작 시 구조화 JSON을 실제 API로 확인하므로 짧은 채팅 호출이 발생한다. JSON Schema 형식이 400·422로 거절되면 JSON object 형식과 로컬 타입 검증을 확인한다. 인증 오류·일시 장애는 형식 미지원으로 간주하지 않는다. 이어서 KB 색인과 임베딩을 준비한다. 준비 실패 시 해당 서버는 요청을 받지 않는다.

런박스에서도 프로세스 환경변수를 먼저 등록한 뒤 프로젝트 루트에서 실행한다.

```bash
python3 -m pip check
python3 -m uvicorn server:app --host 0.0.0.0 --port 8000 --no-access-log
```

서버는 `/tutor`, `/guard`, `/healthz` 모두 인증을 요구한다. 끝 슬래시 경로는 자동 이동하지 않는다. 잘못된 인증은 401, 잘못된 요청 타입은 422이며, 정상 형식 요청의 API 장애는 200과 규격에 맞는 대체 응답으로 처리하고 장애 횟수를 별도로 집계한다.

## 검증

외부 통신 없는 로직·계약·장애 주입 검사:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

이 검사는 실제 API 연결이나 정확도를 증명하지 않는다. 실제 통신과 공개 세트는 별도 명령으로 확인한다.

```powershell
.\.venv\Scripts\python.exe -m tools.evaluate foundation --start-server
```

이 명령은 다음 순서로 수행한다.

1. 운영진 채팅 API의 구조화 응답과 임베딩 API의 1,536차원 벡터를 확인한다.
2. 같은 Python과 환경변수로 `127.0.0.1:8765`에 검증용 서버를 띄운다. 기존 8000 포트 개발 서버는 그대로 둔다.
3. 인증·JSON·리다이렉트 부재·후보 누락/빈 문자열·잘못된 요청 등 계약을 확인한다.
4. 공개 튜터 200개와 가드 200개를 동시 최대 8개로 실행하고 실제 채팅 API로 답변을 판정한다. 실행한 코드의 점수를 측정하므로 과거 1단계의 개선 전 기준 점수와 구분한다.
5. `output/tests/foundation-latest.json`에 결과를 저장하고 이 명령이 만든 검증 서버만 종료한다.

8765 포트가 사용 중이면 중단한다. 필요하면 `--port 8766`처럼 비어 있는 포트를 지정한다. 이미 별도로 실행한 최신 서버를 검사하려면 `--start-server`를 빼고 `--port 8000`을 지정할 수 있다.

검사 중에는 코드와 공개 자료를 수정하지 않는다. 시작·종료 시 코드·자료 지문을 비교한다. 보고서에는 기본 커밋과 지문, 점수, 공개 문항 식별자와 결과, 지연, 오류·대체 응답 횟수만 남긴다. 키·토큰·주소와 요청·응답 본문은 저장하지 않는다. `output/tests/`는 Git 제외 대상이다.

공개 채점은 주최측 `as_scoring_input`, `score_tutor`, `score_guard`, 점수 집계 함수와 판정 지시문을 재사용한다. 문항 배정과 HTTP 전송은 도구가 수행해 동일한 대화의 문항도 ID별로 연결하고 빈 후보를 누락하지 않는다. 주최측 원본 파일은 수정하지 않는다. 판정 전송은 리다이렉트 없이 JSON으로 제한하고, `reasoning_effort=low`, 최대 완료 2,000토큰, 호출당 30초, 자동 재시도 없음으로 실행한다. 실제 상위 API의 정확한 토큰 한도를 확정한 값은 아니다. 판정 모델 없는 부분 점수를 전체 점수로 보고하지 않는다. 판정 호출 수·오류·위해 요청 이행 수와 서버 응답 지연·판정 포함 시간을 구분해 기록한다.

1단계 검증의 통과는 API·계약·기반 점수 측정의 정상 완료를 뜻한다. 최종 품질 목표, 검색 정확도, 실제 Linux 실행, 최대 동시 64개와 제출 터널 검사는 후속 단계에서 수행한다.

## KB와 검색 검증

- MD 48개 섹션을 기본 단위로 삼고 CSV 23개를 본문에서 참조한 위치에 결합한다. 코드·표·원래 ID를 보존한다. HTML과 중복되는 4개 섹션은 내용 일치를 확인해 다시 색인하지 않는다.
- 단어 BM25·한글 두 글자 부분 문자열·의미 임베딩을 RRF로 결합한다. 최신 질문과 이력의 초기 가중치는 3:1이며, 과거 assistant 발화는 검색 단서로만 사용한다. 사실 근거는 KB 섹션 원문이다.
- 기본 상위 8개, 확장 시 최대 12개를 반환한다. 관련 섹션을 찾았다는 사실만으로 질문에 필요한 사실이 자료에 있다고 판정하지 않는다. 입력 자료 검사·후보 근거 검증·튜터 생성은 이 결과와 실제 KB 원문을 사용한다.
- KB 임베딩은 `output/cache/kb-embeddings.json`에 저장한다. KB 내용·모델·차원·전처리 버전·섹션 순서·벡터 체크섬이 일치할 때만 사용한다. 파일은 원자적으로 교체하며 부분 배치는 저장하지 않는다.
- 캐시에는 KB 벡터만 있다. 사용자 대화·질의 벡터·후보 답변은 캐시하지 않는다. 실행 중 질의 임베딩이 실패하면 어휘 검색 결과를 반환하고 고정 오류 코드와 저하 횟수만 집계한다.
- 캐시가 없으면 서버 시작 시 최대 16개 섹션씩 임베딩한다. 준비 예산은 최대 120초이며 첫 요청의 30초 제한과 분리한다. 캐시 재사용 시 KB 임베딩 API를 다시 호출하지 않는다.

API 없이 배포 자료 무결성과 어휘 검색 기준을 확인한다.

```powershell
.\.venv\Scripts\python.exe -m tools.evaluate retrieval --lexical-only
```

이 모드의 통과는 오프라인 평가 실행 완료를 의미한다. 의미 검색이나 검색 품질 98% 달성을 검증한 결과가 아니다.

환경변수가 등록된 PowerShell에서 실제 임베딩 API와 서버 기동을 확인한다.

```powershell
.\.venv\Scripts\python.exe -m tools.evaluate retrieval --start-server
```

KB를 준비하고 공개 튜터 문항 중 정답 근거가 있는 119건을 평가한다. 어휘·의미·결합 검색 각각의 Recall@1·3·4·6·8·12, 영어 등 한글 없는 마지막 질문의 성능, 누락된 공개 ID와 근거 ID, 질의 임베딩 저하 횟수를 기록한다. 이후 전용 서버를 기동해 실제로 색인이 준비된 상태로 응답하고 KB 캐시를 재사용하는지 확인한다. 이 검사에서는 답변 400문항과 판정 모델 채점을 반복하지 않는다.

결과는 `output/tests/retrieval-latest.json`에 기록한다. 원문 대화와 질의 벡터는 저장하지 않는다. 실제 모드의 통과 조건은 캐시 재사용 확인·질의 임베딩 저하 0건·결합 Recall@8 98% 이상이다. 이 목표를 달성해도 튜터·가드 전체 품질과 실제 제출 점수를 보장하지 않는다. `--workers 1`·`2`·`4`·`8`로 평가 동시성을 지정할 수 있다.

2026-11-11 11:24 KST 실제 검사에서 결합 Recall@8 98.32%(117/119), @12 99.16%(118/119)를 확인했다. 임베딩 호출 122회가 모두 성공했고, 별도 검증 서버가 KB 캐시를 재사용해 추가 KB 임베딩 없이 준비됐다. 결합 상위 1개 정확도는 어휘 단독보다 낮았으므로 후속 답변 생성에서 상위 1개를 정답으로 간주하지 않는다. 공개 누락 2건과 언어별 결과는 `Progress.md`에 기록했다.

## 입력 정책과 검증

`tutor_core/policy.py`의 공통 규칙으로 모델이 읽은 요청 의도를 범주·허용·차단과 튜터 행동으로 매핑한다. 먼저 KB 없이 요청 행위·개인정보 대상·지시 덮어쓰기·유해 실행·위기 신호를 각각 읽는다. 학업 대행·수업 외 요청·탈옥·유해 요청은 차단한다. 정답만 요구하는 요청은 허용하면서 힌트로 유도한다. 개인정보 자기 공개는 허용하고, 타인 신상 요구는 유해 요청으로 분류한다. 지속적 위기 신호와 의미 없는 입력도 별도 범주로 처리한다.

입력 분류 함수는 대화만 받는다. 후보 답변은 검색 질의나 분류 프롬프트에 전달하지 않는다. 마지막 학생 요청의 의도를 전체 대화에서 읽고, 이전 assistant 발화는 사실 근거로 사용하지 않는다. 정상 학습 의도일 때만 검색과 별도 자료 존재 검사를 수행한다. KB는 데이터로 전달하며, 관련 주제가 검색됐다는 이유만으로 없는 사실을 만들어 내지 않도록 한다. 자료 여부가 불확실하면 최대 12개로 확장해 한 번 더 확인할 수 있다. 모델이 목록 밖의 ID를 제시하면 실제 KB에 있는 ID만 원문을 추가해 재확인한다. 검증하지 못한 ID와 임의 파일 경로를 근거로 사용하지 않는다.

현재 입력 분류는 검색·필요한 재분류를 합쳐 최대 24초를 사용한다. 기본 추론은 의도·자료 검사 모두 `low`다. `medium` 비교에서는 출력 미완료가 발생해 기본값으로 채택하지 않았다. 요청 전체 내부 마감 27초를 함께 적용한다. 요청 행위 요약은 내부 일회성 데이터로만 사용하고 응답·로그·보고서에 저장하지 않는다. 분류 실패 시 `block`과 임시 `normal` 범주를 반환하고, 이를 정상 분류 성공으로 집계하지 않는다. 실패 이유는 고정 코드와 집계로만 남긴다.

후보가 없으면 input만 반환한다. 빈 후보는 사실 주장과 위해 내용이 없어 grounded=true, safe=true, unsupported=[]로 반환한다. 그 외 후보는 질문과 후보 주장으로 KB 근거를 찾아 별도 검증한다. 후보 답변은 입력 분류에 전달하지 않는다. 입력·출력 중 한쪽이 실패해도 다른 쪽의 판정은 보존한다.

`grounded`와 `safe`는 독립적으로 판정한다. 근거 없는 과목 고유 사실과 위해 내용을 구분한다. 모델이 반환한 `unsupported`를 원문 위치로 복원하여 실제 부분 문자열만 반환하고 중복·중첩 구간을 정리한다. 없는 구간이나 잘못된 근거 ID가 나오면 근거 판정은 미검증으로 처리하되 완료한 안전 판정은 보존한다. 전체 검증 호출 실패 시에는 `verification_unavailable`과 보수적인 값을 반환하고 오류를 별도 집계한다.

환경변수가 등록된 PowerShell에서 실제 입력 정책을 검사한다.

```powershell
.\.venv\Scripts\python.exe -m tools.evaluate policy --start-server
```

검증 서버를 통해 공개 입력 200건·팀 작성 경계 사례 40건·후보 유무와 공격문을 바꾼 격리 검사 3건을 실행한다. 주최측 `score_guard`와 `block_f1`을 재사용하되 후보 답변은 공개 입력 평가에서 제외한다. 공개 라벨로 입력 판정을 채점하며, 별도의 답변 채점 LLM은 호출하지 않는다. 런타임 분류에는 실제 운영진 채팅·임베딩 API를 사용한다.

결과는 `output/tests/policy-latest.json`에 저장한다. 전체 가드 점수나 후보 근거·안전 품질을 계산한 결과가 아니다. 통과 기준은 입력 block F1 95% 이상, 비정상 범주 정확도 90% 이상, 경계·격리 검사 통과와 형식·시간 초과·분류 실패 0건이다. 오류 공개 ID·범주·결정·지연·집계만 기록하며 대화와 후보 원문은 저장하지 않는다.

개발 중 선택된 공개 오류만 의도 속성별로 비교하려면 `python -m tools.evaluate policy --diagnostics`를 사용한다. 현재 분석용으로 보존한 `policy-run4.json`, `policy-run5.json`이 필요하며, 결과는 `policy-diagnostics-latest.json`이다. 원문·모델 요청 요약은 저장하지 않는다. 이 모드의 통과는 진단 호출 완료를 뜻하며 전체 공개 품질 목표 통과가 아니다. 서버 실행과 일반 검증은 이 분석 보고서에 의존하지 않는다.

4단계 실제 검증을 진행할 때 사용하는 명령은 다음과 같다. 현재는 사용자 지시에 따라 실행을 미뤘다.

```powershell
.\.venv\Scripts\python.exe -m tools.evaluate output --start-server
```

이 명령은 공개 가드 200건의 실제 입력·후보 출력을 평가하고 `output/tests/output-latest.json`에 집계·공개 ID·오류·지연만 남긴다. 후보의 근거·안전 정확도와 원문 구간 F1은 공개 라벨로 각각 측정한다. 실제 API 사용 여부, 형식·대체 응답 오류와 품질 목표를 구분하며, 전체 튜터 점수나 부하 검증을 수행한 것으로 간주하지 않는다.

## 자체 가드 튜터

`/tutor`의 기본 실행 경로는 `TutorService.handle`이다. `/guard`가 먼저 호출되었다고 가정하지 않고, 같은 출력 검증 코드를 직접 재사용한다. 입력의 `session_id`나 이전 요청의 대화·응답을 저장해 다음 요청에 사용하지 않는다.

1. 현재 대화로 KB를 검색하고, 첫 구조화 호출에서 요청 의도 속성·자료 범위·초안·인용을 생성한다. 대화·KB·이전 답변은 명시적으로 구분한 데이터로 전달한다.
2. 입력 속성을 공통 `analyze_intent`·`map_policy` 규칙으로 행동에 매핑한다. 학업 대행·지시 덮어쓰기·유해 실행·수업 밖 요청·위기·저신호·자료 밖 요청은 해당 보호 행동의 통제된 문구를 반환한다. 정답만 요구하는 경우에는 힌트·단계를 제공한다.
3. 일반 답변과 생성한 힌트는 독립 호출에서 근거·안전·정책 준수와 원래 요청의 의도를 다시 검토한다. 자기 개인정보를 재출력하거나 금지된 요청을 실제 이행한 초안은 그대로 반환하지 않는다.
4. 안전·정책은 확인됐고 근거 표현만 잘못된 경우 같은 답변·근거로 한 번 재검토할 수 있다. 답변 자체를 보정하려면 유효한 검토, 최소 보정 시간 4초와 후속 검증 6초·응답 여유 0.5초, 채팅 두 번의 예산이 필요하다. 보정은 답변·인용 두 필드만 생성하며 처음의 의도·범위·보호 행동을 유지한다. 보정 답변도 독립 검증하고 위기·차단 신호를 우선 처리한다.
5. 검증기가 실제 답변의 근거로 확인한 KB ID만 최종 인용으로 반환한다. 자료가 뒷받침한다고 생성한 과목 고유 답변에는 확인된 인용을 요구한다. 일반 설명은 인용 없이도 검증될 수 있다. 실제 KB에 있는 인용이라도 원문을 읽어 확인한 경우에만 허용한다.

검증할 수 없거나 시간·호출 예산이 부족하면 초안을 폐기하고 일시적으로 답변을 확인할 수 없다는 응답을 반환한다. API 장애를 수업 자료에 사실이 없다는 결론으로 바꾸지 않는다. 위기 대응은 학습을 멈추고 가까운 신뢰할 사람에게 도움을 요청하도록 안내하며 확인하지 않은 연락처를 생성하지 않는다.

5단계의 로컬 회귀 검사 코드는 `tests/test_tutor.py`에 준비했다. 이번 구현에서는 실행하지 않았으며 실제 API 연결·7행동 정확도·인용 품질·시간 제한은 후속 검증 대상이다.

## 6단계 로컬 검증 도구

아래는 검증을 재개할 때 사용할 명령이다. 이번 구현 작업에서는 실행하지 않았다. `faults`를 제외한 명령은 필요한 프로세스 환경변수가 등록된 터미널에서 사용한다. `--start-server`는 전용 8765 포트의 서버만 생성·종료하며 기존 개발 서버는 종료하지 않는다.

| 명령 | 범위 | 실제 API |
|---|---|---|
| `faults` | 로직·HTTP 계약·SDK 전송 장애·시간 예산·격리·검사 도구 회귀 | 사용하지 않음 |
| `smoke --start-server` | 튜터 7행동과 후보 누락·빈 값·근거·안전의 작은 통합 검사 | 사용 |
| `policy --start-server` | 공개 입력 200건·경계 40건·후보 격리 3건 | 사용 |
| `output --start-server` | 공개 가드 200건의 입력·후보 근거·안전·원문 구간 | 사용 |
| `release --start-server` | 공개 튜터·가드 400건과 실제 판정 모델을 같은 소스로 두 번 평가 | 사용 |
| `load --start-server --full-load --seed 0` | 단계적 부하와 이후 튜터·가드 각 300건의 혼합 부하 | 사용 |
| `acceptance --start-server --full-load --seed 0` | 위 장애·연결·계약·통합·입력·출력·공개 반복·부하를 순차 실행 | 장애 검사 이후 사용 |

예를 들어 전체 로컬 인수 검사를 실행할 때의 명령은 다음과 같다.

```powershell
.\.venv\Scripts\python.exe -m tools.evaluate acceptance --start-server --full-load --seed 0
```

개별 명령도 같은 형식으로 사용한다. `faults`는 서버를 시작하거나 API 환경변수를 읽지 않으며 통제된 대역만 사용한다. 장애 검사 통과는 실제 API 성공과 구분한다. 실제 명령의 `--workers`는 1·2·4·8 중 선택하고 기본은 8이다. 작은 통합 검사는 최대 4개를 병렬 처리한다.

전체 실행은 먼저 오프라인 장애 검사를 수행하고, 실제 API 연결 → 서버 준비 → HTTP 계약 → 작은 통합 검사 → 입력·경계 → 출력 → 공개 두 번 평가 → 단계적·혼합 부하 순서로 진행한다. 운영 오류·소스 변경·서버 재시작이 발생하면 후속 검사를 중단한다. 정확도 목표 미달은 결과로 남기며 지표를 높이기 위한 코드 조정이나 반복 재실행을 자동 수행하지 않는다.

부하는 동시성 1→8→16→32→64에서 각각 동시성의 두 배 요청, 합계 242건으로 시작한다. 각 단계의 첫 묶음은 준비 후 함께 출발하며 실제 진행 중인 클라이언트 요청의 최대 개수도 기록한다. 오류·30초 위반·대체 응답이 발생하면 다음 단계로 높이지 않는다. 동시 64개에서 지연 목표를 넘으면 추가 600건도 보내지 않는다. 모든 단계를 통과하고 `--full-load`를 명시한 경우에만 튜터 300·가드 300건을 최대 64개 동시 요청으로 처리한다. 최종 부하 통과에는 이 600건에서 실제 클라이언트 동시 요청 64개가 관측되어야 한다. 부하 검사는 판정 모델 없이 서버 응답만 측정하며, 시드로 공개 문항 순서를 재현한다. 전체 인수 검사는 공개 반복 평가도 포함하므로 요청 수가 600건보다 많다.

`release`의 기본 반복은 두 번이다. 최초 한 번만 측정하려면 `--repeat 1`을 사용할 수 있지만 두 번 재현 조건을 충족하지 않아 최종 `passed`는 false와 종료 코드 1이다. `load`에서 `--full-load`를 생략하면 단계적 검사만 수행하고 최종 부하 통과로 표시하지 않는다. 기존 서버를 `--port 8000`으로 검사할 수 있지만 코드가 갱신된 새 서버임을 도구가 보증하지 못하므로 최종 공개 반복·부하·인수 검사의 통과에는 `--start-server`가 필요하다.

내부 목표는 총점 90, 튜터·가드 각각 88, 튜터 행동 95%·인용 F1 90%, 입력 block F1 95%·비정상 범주 90%, 후보 grounded·safe 각각 95%·구간 F1 85%다. 형식·HTTP·판정·대체 응답·30초 위반과 위해 요청 이행은 0건이어야 한다. 동시 64 지연은 p95 15초·p99 25초 이내, 혼합 600건의 서버 응답 시간은 150초 이내가 목표다. 이전 품질 기준은 낮추지 않았다.

`output`과 `release`는 후보의 grounded·safe를 공개 원래 라벨로 각각 측정한다. 주최측 집계에서 unsafe 라벨이 근거 판정으로 변환되는 경우와 원래 grounded 라벨의 차이도 개수로 남기므로, 공식 집계 방식의 가드 점수와 독립 정확도를 구분한다. 요청·응답 본문, 모델 요약, 근거 없는 실제 구간 원문, 예외 메시지는 보고서에 저장하지 않는다.

결과는 `output/tests/<명령>-latest.json`이다. 소스·자료 지문, 기준 커밋, 검사 범위, 시드, 공개 ID·점수·지연·오류 집계만 남기고 중간 결과도 원자적으로 교체해 보존한다. 중단되면 마지막 진행 항목과 완료한 결과를 확인할 수 있다. 중간 진행 파일만으로 실행 중인 프로세스가 존재한다고 판단하지 않는다. 로컬 `acceptance` 통과도 7단계 실제 런박스·터널·제출 검증 완료를 뜻하지 않는다.

## 개선 1단계: 원본 판정 설정과 감점 진단

`tools.evaluate`의 공개 판정은 주최측 원본 `make_judge`를 재사용한다. 스트리밍, 완료 토큰 4,000, 시간 제한 60초, 재시도 2회를 유지하고 추론 수준을 별도로 덮어쓰지 않는다. 과거 도구의 low·비스트리밍·완료 토큰 2,000 결과와 구분한다. 이는 로컬 공개 판정이며 실제 제출의 비공개 판정 모델과 동일하다는 의미는 아니다.

개선 기준선을 한 번 측정할 때 프로젝트 루트에서 실행한다.

```powershell
.\.venv\Scripts\python.exe -m tools.evaluate diagnose --start-server --interactive
```

`--interactive`는 없는 주소·모델에 제공된 기본값을 채우고, API 키가 없으면 콘솔의 비표시 입력으로 받는다. 서버 인증 토큰이 없으면 평가 프로세스 전용 임시 토큰을 만든다. 기존 환경변수가 모두 등록돼 있으면 이 옵션을 생략해도 된다. 비밀값 파일을 읽지 않으며 키·토큰은 명령행과 결과에 넣지 않는다. 부모 터미널의 환경변수는 바꾸지 않는다.

전용 8765 포트에 같은 앱 팩토리·모델·정책·예산의 계측 서버를 띄워 공개 튜터 200건·가드 200건을 처리하고 종료한다. 계측 서버는 원래 응답 본문을 유지하며 별도 헤더에 처리 단계·시간·성공 여부·고정 오류 코드와 KB ID만 담는다. 운영용 `server:app`에는 이 헤더가 추가되지 않는다. 임베딩·판정까지 실제 제공 API를 사용하며 부하·품질 조정·재제출은 실행하지 않는다.

진행 개수는 25건마다 표시하고 `output/tests/diagnose-latest.json`에 기록한다. 종료 결과는 시각이 붙은 `diagnose-YYYYMMDD-HHMMSS.json`으로도 보존한다. 튜터 행동·인용·사실·근거 충실도·부적절한 요청 이행과 가드 입력·근거·안전·원문 구간을 분리하며, 공개 ID에 단계별 실패·지연을 연결한다. 판정 지표는 전체 대상 수(`expected`), 유효 응답으로 판정 가능한 수(`eligible`), 실제 관측 수(`observed`)를 함께 기록한다. 실패 표시와 단계 시간은 중복·포함 관계가 있으므로 합산해 원인별 감점이나 전체 처리 시간으로 해석하지 않는다. 첫 검색의 정답 근거 누락은 관측 사실이며 그것만으로 낮은 점수의 원인을 확정하지 않는다.

`passed_scope=baseline_measurement_only`인 `passed`는 기준선 측정 완료 여부다. 내부 실패·점수 미달을 관측해도 측정은 끝날 수 있으며, 별도의 `operational_pass`, `quality_targets`, `quality_target_met`로 운영·품질을 확인한다. 400건 중 계측 누락·판정 오류·소스 변경·서버 재시작이 있으면 측정 완료로 표시하지 않는다. 요청·후보·답변·모델 요약·원문 구간·예외 원문은 결과에 저장하지 않는다.

## 개선 2단계: 시간 예산과 검증 실패 처리

초안은 최대 12초, 답변·인용만 생성하는 보정은 최대 8초를 사용한다. 각 생성 전에 남은 전체 시간에서 독립 검증 6초와 응답 여유 0.5초를 제외해 호출 상한을 계산한다. 보정에 4초를 배정할 수 없거나 재검증까지 필요한 두 번의 호출이 없으면 보정을 시작하지 않는다. 독립 검증은 최대 11초와 남은 시간을 함께 적용한다. 이 값들은 27초 전체 예산 안에서 사용하는 상한이며 각 단계의 상한을 모두 더해 사용할 수 있다는 뜻은 아니다.

근거 검증 오류는 `invalid_supporting_ids`, `invalid_unsupported_spans`, `inconsistent_grounding`으로 세분한다. 튜터에서 모델이 근거가 있다고 판정했고 안전·정책이 확인됐으나 표현 검사가 실패한 경우만 같은 후보와 근거로 한 번 재검토한다. 재검토는 근거 세 필드만 확인하며 최대 6초, 최소 시작 여유 3초를 사용한다. 같은 질문·답변인지 요청 안에서만 지문을 비교한 뒤 기존 안전·정책 판정을 보존한다. 이 지문은 캐시·로그·보고서에 저장하지 않는다. 모델이 근거 없다고 판정한 답변은 유효한 검토에서 원문 구간을 확인한 경우에만 내용 보정 대상으로 삼는다. 해당 호출도 최대 4회에 포함한다. ID를 임의 삭제하거나 검증되지 않은 사실을 정상 근거로 바꾸지 않으며, 재검토가 실패하면 미검증 답변을 반환하지 않는다. 가드의 병렬 입력·출력 분기에는 추가 재시도를 넣지 않는다.

2단계의 실제 비교 측정은 프로젝트 루트에서 실행한다.

```powershell
.\.venv\Scripts\python.exe -m tools.evaluate diagnose --start-server --interactive --work-item IMP-002-2
```

1단계 보존본 `output/tests/diagnose-20261111-214443.json`과 새 보존 결과의 점수·저하·단계 실패·안전·지연을 비교한다. 보고서의 작업 항목을 `--work-item`으로 구분하며 측정 완료와 변경 채택·최종 품질 통과를 별도로 판단한다. 실제 비교 전에는 시간 제한 실패가 줄었다고 보고하지 않는다.

2단계 1차 비교는 총점 상승과 함께 실패·지연 악화도 관측돼 채택을 보류했다. 보존본 `diagnose-20261112-102606.json`의 공개 실패 유형, 이전 요청 이행 판정과 범주별 대조군을 제한적으로 재검사할 때는 `python -m tools.recovery_checks --interactive`를 사용한다. 이 도구는 최대 80건·동시 작업 4개로 제한하고 선택 편향을 명시한다. 현재 선택은 튜터 51·가드 5건이다. `recovery-latest.json`과 시각이 붙은 보존본에 결과를 남기며, 선택 사례의 점수를 전체 공개 점수나 제출 점수로 보고하지 않는다. 실패 감소·계약·안전 관측을 통과한 뒤 전체 공개 비교를 수행한다.

사용자가 현재 범위에서 2단계를 정리하고 3단계로 넘어가도록 지시했다. 보완 후 제한 56건의 측정은 완료했으나 마지막 보완 코드의 전체 400건 비교는 미확인이다. 전체 품질·채택 기준을 통과했다는 의미로 해석하지 않는다.

## 개선 3단계: 검색 질의와 근거 전달

검색 입력만 가공하며 정책 판정에 쓰는 원래 대화는 유지한다. 최신 질문은 12,000자, 이력은 24,000자 안에서 앞·뒤를 함께 보존한다. 긴 이력은 발화별 최대 4,000자를 배정해 한 assistant 발화가 앞선 사용자 질문을 모두 밀어내지 않게 한다. 임베딩 질의는 UTF-8 7,500바이트 안에서 앞부분과 뒷부분을 보존한다. 짧은 질의와 결합 가중치·검색 개수는 유지한다.

출력 검증에는 유효한 인용·질문 검색 상위 8개·후보 주장 검색의 합집합을 최대 12개 섹션으로 전달한다. 합집합이 한도 안에 들어오면 기존 인용→질문→후보 순서를 유지하고, 한도를 넘으면 출처별 순환 선택으로 인용이 질문·후보 근거를 독점하지 않게 한다. 중복 ID는 제거하고 MD·CSV의 전체 원문과 조건·표 머리글·섹션 ID를 보존한다. 알려지지 않은 인용을 유효하다고 인정하지 않는다. 후보는 출력 검증용 검색에만 사용한다.

실제 검색과 제한 표본을 이전 방식과 비교하려면 프로젝트 루트에서 실행한다.

```powershell
.\.venv\Scripts\python.exe -m tools.search_checks --interactive
```

공개 근거 119건과 별도 긴 입력 24건(최신 질문 12·이력 12)을 동시 작업 4개로 비교한다. 이전·개선 방식의 임베딩 입력이 같으면 해당 사례 안에서 같은 실제 벡터를 사용해 API 변동을 줄인다. 질의·벡터는 파일과 요청 간 캐시에 남기지 않는다. Recall@1·3·8·12와 지연을 기록하고 공개 Recall@8 98% 목표를 확인한다. 이 구간의 임베딩 지연은 공유 배치 지연이므로 각 운영 방식의 독립 API 지연으로 해석하지 않는다.

이어서 공개 튜터 24건(과목별 8건)과 후보가 있는 가드 6건을 두 방식으로 각각 검사한다. 8765 포트에 이전 검색 대조군과 개선 계측 서버를 차례로 띄우며 각 실행이 만든 서버만 종료한다. 두 방식은 같은 현재 정책·시간 예산·주최측 원본 판정을 사용하고 검색 질의와 근거 배분만 다르게 한다. 3단계 당시 보존 보고서는 그때의 2단계 정책을 공통으로 사용했다. 이전 방식은 평가용 팩토리 주입으로 재현하므로 같은 코어 지문이어도 보고서의 `application`과 프로필을 함께 확인한다. 운영 서버는 평가 입력을 탐지해 이 대조군을 선택하지 않는다.

결과는 `output/tests/search-latest.json`과 시각이 붙은 보존본이다. 본문 없이 공개 ID·KB ID·점수·근거 출처별 개수·문맥 크기·실패 종류·지연을 남긴다. 같은 섹션이 여러 출처에 포함될 수 있어 출처별 개수는 합산하지 않는다. `measurement_complete`와 `passed_scope=search_and_selected_profile_measurement_only`는 측정 완료를 뜻하며, 검색 목표·계약 검사는 별도 필드다. 작은 선택 표본의 점수는 전체 공개·제출 점수가 아니고 순차 실행의 생성·판정 변동으로 인과를 확정하지 않는다. 최종 품질·64 동시 부하·런박스 재현은 후속 검증으로 남는다.

## 개선 4단계: 공통 입력 정책 경계

입력 가드, 튜터 초안, 독립 검토가 `INPUT_POLICY_RULES`의 같은 기준을 사용한다. 문서·인용문의 주제와 실제 요구 행위를 구분하며 학습 검토/제출 대행, 개념 설명/평가 정답, 학습용 형식/실제 광고, 인용 분석/공격 적용, 자기 공개/타인 침해/가상 인물, 시험 걱정/심각한 위기를 나눈다. 모델 속성의 결정·행동 매핑과 후보 격리는 유지한다. 단어 목록이나 공개 문항 ID로 범주를 결정하지 않는다.

일반 설명과 과정 고유 사실 부재는 가드·튜터의 같은 함수로 매핑한다. 자료 판정의 supported에는 실제 읽은 ID가 필요하고 missing_course_fact에는 근거 ID를 넣지 않는다. 첫 검토에 상위 8개 원문만 보냈다면 제목 목록·미전달 9~12번째 ID를 읽은 증거로 인정하지 않는다. 기존 예산 안에서 원문을 추가해 재확인하거나 고정 오류로 실패 처리한다. 입력 정책 버전은 `input-v3`이며 요청·응답 규격, 시간·호출 상한은 유지한다.

실제 4단계 검증은 프로젝트 루트에서 실행한다.

```powershell
.\.venv\Scripts\python.exe -m tools.input_policy_checks --interactive
```

전용 8765 서버에서 공개 입력 200건, 팀 작성 경계 40건(기존 20건과 의도가 다른 10쌍), 후보 격리 3건, 대표 튜터 행동 10건을 동시 작업 4개로 확인한 뒤 종료한다. 공개 입력 평가에는 후보를 보내지 않는다. 주최측 `score_guard`·`block_f1`을 재사용하고 추가 LLM 채점은 실행하지 않으며 런타임에는 실제 제공 API를 사용한다. 키는 프로세스 변수 또는 비표시 입력만 사용하고 비밀값 파일을 읽지 않는다.

25건마다 진행 개수를 표시한다. `output/tests/input-policy-latest.json`과 시각이 붙은 보존본에는 공개/팀 사례 ID·결정·범주·행동·자료 범위 enum·분류 검증 여부·고정 실패 종류·지연만 기록한다. 본문·후보·답변·모델 요약은 남기지 않는다. 이전 진단 보고서가 있으면 참고 지표를 붙이지만 실행 필수 파일이 아니며 시각·동시성·후보 분기 차이로 인과를 확정하지 않는다.

`measurement_complete`와 `passed_scope=input_policy_measurement_only`는 측정 완료를 뜻한다. 공개 block F1 95%·비정상 범주 90%, 경계·격리·입력 통신·대표 튜터 행동을 별도 목표로 확인한다. 형식이 유효한 실패 응답을 분류 성공으로 숨기지 않는다. 대표 행동 검사는 전체 튜터 점수나 후보 근거·안전·64 동시 부하 검증이 아니며, 기존 2·3단계의 미해결 지연과 품질을 통과로 바꾸지 않는다.

## 시간 제한과 로그

- 내부 마감은 요청당 27초다. 본문 파싱·대기열·API 호출을 포함한다. 대회 제한은 30초다.
- 튜터는 검색 최대 3초, 초안 최대 12초, 독립 검증은 추가 근거 검색을 포함해 최대 11초를 사용한다. 답변 보정은 최대 8초, 근거 표현 재검토는 최대 6초이며 생성 전 후속 검증 시간을 확보한다. 정상 생성·검증은 보통 두 번, 표현 재검토는 세 번, 답변 보정·재검증까지 최대 네 번의 채팅 호출이다. 실제 배정 시간은 항상 남은 전체 예산으로 제한한다.
- 입력 가드는 검색·재분류를 포함해 최대 24초, 후보 검증 분기는 검색 3초와 모델 호출 8초에 해당하는 최대 11초를 사용한다. 입력·후보 분기는 병렬이고 전체 요청 마감 27초와 최대 채팅 4회 예산을 공유한다.
- API 연결은 서버 수명 동안 재사용하고, 채팅 128·임베딩 64 동시 호출로 제한한다. 아직 이 수치의 실제 부하 검증을 마쳤다는 뜻은 아니다.
- SDK 자동 재시도를 끄고 요청별 채팅 횟수 예산을 둔다. 전송 리다이렉트와 환경 프록시 자동 적용도 끈다.
- SDK 디버그·접근 로그에서 요청·응답이 노출되지 않도록 억제한다. 애플리케이션 로그는 고정 경로명·상태 코드·지연·고정 오류 코드만 남긴다.
- 장애 대체 답변을 정상 품질 성공으로 집계하지 않는다. 튜터의 보호 문구 반환, 독립 검증, 보정, 검증 불가 응답을 별도로 집계한다. 실제 정확도·시간 제한 검증은 남아 있다.

## 7단계 런박스 실행·인계

아래 절차는 검증을 재개하고 Git 반입을 승인했을 때 사용한다. 이번 구현에서는 실행하지 않았다. 환경변수 등록, 커밋·push, 런박스 clone/checkout과 실제 제출은 사용자가 담당한다. 도구는 패키지를 설치하거나 Git을 갱신하거나 제출 설정 파일을 읽고 쓰지 않는다.

1. 로컬 검증과 남은 품질 문제를 확인한 뒤, 검수·승인한 코드·KB·공개 자료·검사 도구·문서를 비공개 GitHub 커밋으로 남긴다. 커밋 메시지와 push는 별도 승인 범위다. `.venv`, 비밀값 파일, 로컬 검사 결과와 실행 캐시는 반입하지 않는다.
2. 런박스에서 같은 커밋을 clone/checkout한다. 기본 프로젝트 경로는 `/home/elicer/competition/ai_turtor_guardrail`이며 이후 명령은 그 저장소 루트에서 실행한다. `git rev-parse HEAD`로 승인한 커밋인지 확인한다. Git 반입 대상에는 `requirements-runbox.txt`, `server.py`, `tutor_core/`, `kb/`, `public/`, `harness/`, `grader/`, `tools/`, `tests/`, `README.md`, `Plan.md`, `Progress.md`가 포함된다.
3. 런박스의 기존 Python·패키지가 호환되면 별도 가상환경은 필요 없다. Python 3.10 계열·Linux x86_64와 고정 패키지 버전을 확인한다. 기준 Python 패치는 3.10.14로 기록하며 실제 패치 버전도 결과에 남긴다. 패키지 변경이 필요할 때만 사용자가 `requirements-runbox.txt` 기준으로 설치한다. 로컬 Windows 가상환경을 복사하지 않는다.
4. 사용자가 해당 터미널의 프로세스 환경변수 7개를 등록한다. 파일 자동 로딩은 사용하지 않는다. 새 터미널·런박스 재시작 후에는 변수가 유지되었는지 확인해야 한다. 도구와 서버는 실행한 터미널에서 환경을 상속한다.
5. 실제 제공 API로 로컬 인수 검사와 런박스의 서버·재시작·터널 검사를 수행한다. 품질 검사와 배포 확인을 구분한다. 구현 번호나 ready만으로 통과를 판단하지 않는다.
6. 최종 코드의 품질·부하·Linux·터널 결과와 커밋·지문이 일치하는지 확인한 뒤 사용자가 제출 정보를 작성하고 제출한다. 코드와 프롬프트를 공개하지 않으며 제출 중에는 해당 런박스 서버를 계속 실행한다.

API 없이 환경과 반입 상태만 확인할 때:

```bash
python3 -m tools.runbox --local-only
```

이 명령은 패키지 버전·의존성 충돌·Python 3.10 문법·KB 48개 섹션/CSV 23개/HTML 중복 4개·필수 파일을 확인한다. Git 추적뿐 아니라 HEAD 커밋 포함 여부와 미커밋 변경도 확인한다. 알려진 비밀값 파일이 추적되는지는 이름·상태만 검사하고 내용을 열지 않는다. 로컬 Windows에서도 실행할 수 있지만 환경 검사 통과가 Linux 실행·배포 통과를 의미하지 않는다.

실제 서버는 환경변수가 등록된 런박스 터미널에서 실행한다.

```bash
python3 -m uvicorn server:app --host 0.0.0.0 --port 8000 --no-access-log
```

서버가 이 터미널을 사용하므로 점검 명령을 다른 터미널에서 실행한다면 그 터미널에도 환경변수가 필요하다. 기존 서버가 이전 코드를 불러온 상태라면 새 코드를 반입한 뒤 사용자가 해당 서버를 재시작해야 한다.

런박스에서 전용 서버 재시작과 실제 8000 포트 터널을 함께 확인할 때:

```bash
TUTOR_TUNNEL_URL='https://YOUR_RUNBOX.tunnel.elice.io/proxy/8000'
python3 -m tools.runbox --start-server --port 8765 --tunnel-url "$TUTOR_TUNNEL_URL"
```

`YOUR_RUNBOX`는 본인 런박스 호스트로 바꾼다. 기본 주소에는 `/tutor`나 `/guard`를 붙이지 않는다. 도구는 HTTPS 런박스 프록시 기본 주소만 허용하고 리다이렉트를 따라가지 않는다. 명령행에 키·토큰을 넣지 않는다. 승인한 전체 Git 해시를 대조하려면 선택적으로 `--expected-commit`을 추가한다.

이 명령은 비어 있는 8765 포트에서 같은 Python으로 두 번 새 서버를 띄우고 각각 종료한다. 두 기동의 서로 다른 인스턴스, 정적 KB 캐시 재사용과 두 번째 기동의 KB 임베딩 재생성 0회를 확인한다. 이어서 사용자가 이미 실행한 8000 포트의 터널을 검사한다. 기존 개발·제출 서버는 종료하지 않는다. 첫 기동 확인에 실패하면 두 번째 기동과 터널 검사를 진행하지 않는다.

서버의 `runtime_sha256`은 코드·의존성 기준·KB를 UTF-8 텍스트로 정규화한 지문이다. Windows CRLF와 Linux LF, UTF-8 BOM 차이를 제거하고 경로·내용을 함께 대조한다. 도구는 실행·검사 코드와 자료의 지문도 별도로 기록하고, 인계 문서 변경은 Git 상태로 구분한다. 시작 중 코드가 바뀌면 서버가 준비 완료로 전환되지 않는다. 검사 대상의 지문·기능 준비 상태가 다르면 튜터·가드 POST를 추가로 보내지 않는다.

실제 확인은 인증 누락·잘못된 인증 거절, 튜터 JSON·행동, 가드 빈 후보·입력 판정, 끝 슬래시 무리다이렉트, 30초 제한, 대체 응답 없음, 검사 중 인스턴스 유지, 실제 상위 채팅 호출 관측을 포함한다. 요청·응답 본문과 키·토큰·URL은 보고서에 남기지 않는다.

결과는 `output/tests/runbox-latest.json`이다. `requested_checks_passed`는 요청한 범위의 검사 결과이며, `deployment_verified`는 호환 런박스 환경·커밋 반입·소스 일치·재시작·캐시·터널 확인을 모두 마친 경우에만 true다. `--start-server` 또는 `--tunnel-url`만 실행하면 다른 배포 조건이 남으므로 전체 `passed`는 false와 종료 코드 1일 수 있다. Git에 아직 커밋되지 않은 파일이 있으면 코드 기동이 가능해도 커밋으로 재현 가능한 인계 완료로 표시하지 않는다.

배포 확인에는 공개 정확도·동시 64 부하·실제 제출이 포함되지 않는다. `tools.evaluate acceptance`와 실제 배포 확인을 각각 마친 뒤 사용자가 최종 제출을 판단한다. `submission.json` 작성·수정과 제출 버튼 조작은 사용자의 작업이며 Codex는 해당 파일에 접근하지 않는다. 실제 제출 전까지 확인하지 못한 항목은 `Progress.md`의 미완료 상태로 유지한다.

## 환경변수 상속에 관한 주의

다른 PowerShell에 등록한 프로세스 환경변수는 이미 실행 중인 Codex와 다른 터미널에 자동 전달되지 않는다. 현재 대화에서 바로 검증하려면 환경변수가 있는 PowerShell에서 위 명령을 실행하고 보고서로 결과를 확인하는 방법을 사용할 수 있다.

Codex가 직접 실행하려면 해당 환경을 상속한 새 Codex 프로세스가 필요하다. Codex의 `shell_environment_policy`는 자식 명령에 전달할 변수도 따로 필터링하므로, 새 세션에서 값 대신 필요한 변수의 등록 여부만 확인한다. 실행 중인 다른 프로세스의 메모리를 읽거나, 비밀값을 명령행 인수·설정 파일·대화에 복사하는 방식은 사용하지 않는다.

### 새 Codex CLI에 전달하는 절차

이 절차는 현재 대화 프로세스를 변경하는 명령이 아니라 새 CLI를 여는 방법이다. 대회용 `OPENAI_*`와 Codex 자체의 연결 설정이 섞이지 않도록, 프로젝트 변수 7개를 `TUTOR_RUNTIME_` 별칭으로 자식 프로세스에만 전달한다. 기존 PowerShell의 변수는 바꾸지 않는다. 상속할 값은 출력하거나 파일에 저장하지 않는다.

먼저 환경변수를 등록했던 PowerShell을 사용한다. 새 PowerShell에는 이전 터미널의 프로세스 환경변수가 없으므로, 기존 터미널을 사용할 수 없다면 아래 블록으로 등록한다. 비밀값은 화면에 표시되지 않는 입력창에서 직접 입력하며, 파일을 읽지 않는다. `TEAM_TOKEN`에는 기존 서버에서 사용한 토큰을 입력한다.

```powershell
& {
    $ErrorActionPreference = 'Stop'
    $tutorDefaults = @{
        OPENAI_BASE_URL = 'https://mlapi.run/daef5150-72ef-48ff-8861-df80052ea7ac/v1'
        OPENAI_MODEL = 'gpt-5-nano'
        EMBEDDING_BASE_URL = 'https://mlapi.run/b54ff33e-6d14-42df-93f9-0f1132160ee8/v1'
        EMBEDDING_MODEL = 'text-embedding-3-small'
    }
    foreach ($name in $tutorDefaults.Keys) {
        if ([string]::IsNullOrWhiteSpace([Environment]::GetEnvironmentVariable($name, 'Process'))) {
            [Environment]::SetEnvironmentVariable($name, $tutorDefaults[$name], 'Process')
        }
    }
    foreach ($name in @('TEAM_TOKEN', 'OPENAI_API_KEY')) {
        if ([string]::IsNullOrWhiteSpace([Environment]::GetEnvironmentVariable($name, 'Process'))) {
            $tutorSecret = Read-Host "$name 입력" -AsSecureString
            $tutorCredential = [System.Net.NetworkCredential]::new('', $tutorSecret)
            if ([string]::IsNullOrWhiteSpace($tutorCredential.Password)) { throw "입력되지 않은 변수: $name" }
            [Environment]::SetEnvironmentVariable($name, $tutorCredential.Password, 'Process')
            $tutorCredential = $null
            $tutorSecret.Dispose()
        }
    }
    if ([string]::IsNullOrWhiteSpace($env:EMBEDDING_API_KEY)) {
        $env:EMBEDDING_API_KEY = $env:OPENAI_API_KEY
    }
    '필요한 환경변수를 현재 PowerShell에 등록했습니다.'
}
```

등록한 **같은 PowerShell**에서 아래 블록 전체를 한 번에 실행한다. PATH에서 CLI를 찾지 못하면 사용자 VS Code 확장 폴더에서 실행 파일을 찾는다. 필수 변수가 없거나 실행 파일이 없으면 프로세스를 만들기 전에 중단한다.

```powershell
& {
    $ErrorActionPreference = 'Stop'
    $tutorNames = @(
        'TEAM_TOKEN', 'OPENAI_BASE_URL', 'OPENAI_MODEL', 'OPENAI_API_KEY',
        'EMBEDDING_BASE_URL', 'EMBEDDING_MODEL', 'EMBEDDING_API_KEY'
    )
    $tutorMissing = @($tutorNames | Where-Object {
        [string]::IsNullOrWhiteSpace([Environment]::GetEnvironmentVariable($_, 'Process'))
    })
    if ($tutorMissing.Count -gt 0) { throw ('등록되지 않은 변수: ' + ($tutorMissing -join ', ')) }
    $tutorCommand = Get-Command codex -CommandType Application -ErrorAction SilentlyContinue
    $tutorCodexExe = if ($tutorCommand) { $tutorCommand.Source } else { $null }
    if (-not $tutorCodexExe) {
        $tutorExtensions = Join-Path $env:USERPROFILE '.vscode\extensions'
        $tutorCandidates = @(Get-ChildItem -LiteralPath $tutorExtensions -Directory -Filter 'openai.chatgpt-*' -ErrorAction SilentlyContinue |
            ForEach-Object {
                $tutorCandidate = Join-Path $_.FullName 'bin\windows-x86_64\codex.exe'
                if (Test-Path -LiteralPath $tutorCandidate -PathType Leaf) { Get-Item -LiteralPath $tutorCandidate }
            } | Sort-Object LastWriteTimeUtc -Descending)
        if ($tutorCandidates.Count -gt 0) { $tutorCodexExe = $tutorCandidates[0].FullName }
    }
    if (-not $tutorCodexExe) { throw 'Codex CLI 실행 파일을 찾지 못했습니다.' }
    $tutorStart = New-Object System.Diagnostics.ProcessStartInfo
    $tutorStart.FileName = $tutorCodexExe
    $tutorStart.WorkingDirectory = (Get-Location).Path
    $tutorStart.UseShellExecute = $false
    foreach ($name in $tutorNames) {
        $tutorStart.EnvironmentVariables["TUTOR_RUNTIME_$name"] = [Environment]::GetEnvironmentVariable($name, 'Process')
        $tutorStart.EnvironmentVariables.Remove($name)
    }
    $tutorArgs = @(
        '--no-daemon', '-c', 'shell_environment_policy.inherit=all',
        '-c', 'shell_environment_policy.ignore_default_excludes=true'
    )
    foreach ($name in @('TUTOR_RUNTIME_*', 'PATH', 'PATHEXT', 'SYSTEMROOT', 'WINDIR', 'TEMP', 'TMP', 'USERPROFILE', 'LOCALAPPDATA', 'APPDATA', 'COMSPEC')) {
        $tutorArgs += @('-c', "shell_environment_policy.filters.$name=include")
    }
    $tutorStart.Arguments = $tutorArgs -join ' '
    $tutorProcess = [System.Diagnostics.Process]::Start($tutorStart)
    if ($null -eq $tutorProcess) { throw 'Codex CLI를 시작하지 못했습니다.' }
    $tutorProcess.WaitForExit()
}
```

`--no-daemon`은 이전 환경을 가진 공용 백그라운드 서버를 재사용하지 않기 위한 옵션이다. 이 프로젝트에서 확인한 CLI는 `0.164.0-alpha.11.1`이며 이 옵션을 제공한다. CLI 로그인은 기존 Codex 계정을 사용한다.

새 세션에는 현재 구현 범위와 검증 보류 상태를 함께 전달한다.

> `Plan.md`와 `Progress.md`를 확인해. 1~7단계의 코드 구현은 정리했지만 실제 검증과 품질 개선은 보류한 상태야. 최신 사용자 지시로 검증 재개가 허가되었는지 먼저 구분하고, 허가된 검사만 진행해. 비밀값 파일은 읽지 말고, 키·토큰 값을 출력하지 마.

실제 검증 재개를 요청받았을 때 검사 명령과 같은 PowerShell 도구 호출 안에서 아래 복원을 수행한다. 복원 자체가 검사 실행 승인을 뜻하지 않으며 이후 명령 호출에 환경이 유지된다고 가정하지 않는다.

```powershell
$tutorNames = 'TEAM_TOKEN', 'OPENAI_BASE_URL', 'OPENAI_MODEL', 'OPENAI_API_KEY', 'EMBEDDING_BASE_URL', 'EMBEDDING_MODEL', 'EMBEDDING_API_KEY'
foreach ($name in $tutorNames) {
    $value = [Environment]::GetEnvironmentVariable("TUTOR_RUNTIME_$name", 'Process')
    if ([string]::IsNullOrWhiteSpace($value)) { throw "전달되지 않은 변수: $name" }
    [Environment]::SetEnvironmentVariable($name, $value, 'Process')
}
'프로세스 환경변수 별칭 복원을 마쳤습니다.'
```

이 안내는 CLI 도움말과 공식 환경 전달 정책을 대조한 절차다. 현재 사용자의 API 변수를 실제로 전달해 검사한 결과는 아니다. 별도 관리 정책이 변수를 제외하면 새 세션에서 이름과 등록 여부만 확인하고 해당 제한을 먼저 해결해야 한다.

참고: [OpenAI Docs — 환경변수 전달 정책](https://learn.chatgpt.com/docs/config-file/config-advanced#shell-environment-policy), [구조화 응답](https://developers.openai.com/api/docs/guides/structured-outputs).
