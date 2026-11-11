"""색인용 문서 확장 생성(오프라인 1회): 청크마다 한국어 질문·영어 용어/질문을 만들어 kb_expansion.json에 저장한다. DEVLOG 22장.
- 입력은 지식 베이스 청크 본문뿐이다(공개 세트·채점 요청의 문장은 쓰지 않는다). 운영진 제공 LLM(gpt-5-nano)으로 만든 가공 자료(SPEC §7.1 허용).
- 서버는 이 파일을 읽기만 한다(요청 처리 중 LLM 호출 없음).

  python3 scripts/build_expansion.py
"""
import asyncio
import json
import sys
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")
from app.kb import load_sections, make_chunks  # noqa: E402
from app.llm import chat_json  # noqa: E402

SYSTEM = """너는 중학생용 온라인 수업 자료(파이썬 기초, AI 프롬프트 활용, 금융 리터러시)의 검색 색인을 돕는다.
아래 [자료 조각]을 읽고, 학습자가 이 조각을 찾으려고 쓸 만한 검색 표현을 만든다. 자료에 없는 내용은 만들지 않는다.
- ko_questions: 이 조각의 내용을 묻는 한국어 질문 4개. 조각에 나오는 실습·예제·상품·인물·도구 이름을 그대로 넣고, 학습자가 쓰는 짧은 말투로 쓴다. 답은 쓰지 않는다.
- en_terms: 이 조각의 핵심 용어(이름, 개념, 코드 용어)를 영어로 옮긴 짧은 표현 6~8개. 학습자가 영어로 질문할 때 쓸 단어(예: 포인트 → points, 반복문 → loop)를 고른다.
- en_questions: 이 조각의 내용을 묻는 영어 질문 2개.
출력: JSON 객체 하나 {"ko_questions": [...], "en_terms": [...], "en_questions": [...]}"""


async def main():
    secs = load_sections()
    chunks = make_chunks(secs)
    keyed, ordn = [], defaultdict(int)
    for c in chunks:
        keyed.append((f"{c['sid']}#{ordn[c['sid']]}", c)); ordn[c["sid"]] += 1
    sem = asyncio.Semaphore(16)

    async def one(key, c):
        async with sem:
            for _ in range(2):
                try:
                    out = await chat_json(SYSTEM, f"[자료 조각]\n{c['text']}", effort="low", max_tokens=2000, timeout=40, retries=1)
                    return key, {k: [str(x) for x in out.get(k, []) if isinstance(x, str)] for k in ("ko_questions", "en_terms", "en_questions")}
                except Exception:  # noqa: BLE001
                    pass
            return key, None

    res = await asyncio.gather(*[one(k, c) for k, c in keyed])
    ok = {k: v for k, v in res if v}
    (ROOT / "kb_expansion.json").write_text(json.dumps(ok, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"청크 {len(keyed)}개 중 생성 {len(ok)}개, 저장 kb_expansion.json")


asyncio.run(main())
