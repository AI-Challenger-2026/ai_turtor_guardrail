"""공개 세트 유사 예시 검색(few-shot). SPEC §7: 공개 세트를 예시로 쓰거나 비슷한 예시를 찾아 넣는 것은 허용.

질문과 비슷한 공개 세트 문항을 BM25로 찾아, 그 문항의 공개 라벨(범주)과 함께 분류 프롬프트에 넣는다.
서버 시작 때 공개 세트를 읽어 색인을 만든다(요청 사이에 상태를 쌓지 않음 — 읽기 전용).
"""
import json
from pathlib import Path
from typing import Dict, List, Optional

from .kb import BM25

PUBLIC_DIR = Path(__file__).resolve().parents[1] / "public"

# 공개 라벨(범주) → 분류기 신호 이름. 예시를 신호 언어로 보여 주기 위함.
CAT_TO_SIGNAL = {
    "normal": "모든 신호 false",
    "out_of_material": "모든 신호 false (정상 학습 질문으로 봄)",
    "pii_disclosure": "self_pii",
    "emotional_distress": "distress",
    "low_signal_input": "low_signal",
    "jailbreak_injection": "jailbreak",
    "academic_dishonesty": "submit_work",
    "answer_seeking": "answer_only",
    "off_topic": "off_topic",
}


def _label(r) -> str:
    c = r["labels"]["category"]
    if c == "harmful_content":
        act = r["labels"].get("action")  # 튜터 세트에만 있음: redirect=혐오·모욕, refuse=위해 요청
        return "hate" if act == "redirect" else ("harm" if act == "refuse" else "harm 또는 hate")
    return CAT_TO_SIGNAL.get(c, "모든 신호 false")


def _key_text(messages) -> str:
    users = [m.get("content", "") for m in messages if m.get("role") == "user"]
    return "\n".join(users[-2:])


class FewShot:
    def __init__(self, rows: Optional[List[dict]] = None):
        if rows is None:
            rows = [json.loads(l) for e in ("tutor", "guard")
                    for l in (PUBLIC_DIR / f"{e}.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
        self.rows = rows
        self.bm25 = BM25([_key_text(r["messages"]) for r in rows])

    def examples(self, messages, k: int) -> List[dict]:
        sc = self.bm25.scores(_key_text(messages))
        order = sorted(range(len(sc)), key=lambda i: -sc[i])
        out, seen = [], set()
        for i in order:
            last = self.rows[i]["messages"][-1]["content"]
            if last in seen:
                continue
            seen.add(last)
            out.append(self.rows[i])
            if len(out) >= k:
                break
        return out

    def render(self, messages, k: int) -> str:
        exs = self.examples(messages, k)
        if not exs:
            return ""
        lines = ["[참고: 비슷한 공개 예시와 정답 신호] 겉모습이 비슷해도 정답이 다를 수 있다. 예시의 차이를 보고 기준을 적용하되, 판단은 분류 대상 발화로 한다."]
        for n, r in enumerate(exs, 1):
            last = r["messages"][-1]["content"].replace("\n", " / ")
            prev = [m["content"] for m in r["messages"][:-1] if m["role"] == "user"]
            ctx = f" (앞 학습자 발화: {prev[-1][:80]!r})" if prev else ""
            lines.append(f"{n}. 마지막 발화: {last[:220]!r}{ctx} → {_label(r)}")
        return "\n".join(lines)
