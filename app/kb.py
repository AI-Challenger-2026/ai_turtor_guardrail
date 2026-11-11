"""지식 베이스 로딩과 BM25 검색.

- 섹션 = md 파일 하나(+ 같은 이름의 CSV 표를 본문 자리에 끼워 넣음). 인용 단위.
- 청크 = 섹션을 '## ' 소제목 단위로 나눈 조각(앞에 섹션 제목을 붙임). 검색 단위.
- 섹션 점수 = 그 섹션 청크들의 최고 점수.
색인은 서버 시작 때 만든다(첫 요청에서 만들면 시간 초과). 외부 호출 없음.
임베딩(text-embedding-3-small)은 검색 실험에서 BM25보다 낮아 쓰지 않는다(DEVLOG 2장, 커밋 bb4c9cb에 코드 있음).
"""
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

KB_DIR = Path(__file__).resolve().parents[1] / "kb"


def _inline_tables(md_path: Path, text: str) -> str:
    def repl(m):
        csv_path = md_path.parent / m.group(1)
        if csv_path.exists():
            return "[표]\n" + csv_path.read_text(encoding="utf-8").strip() + "\n[표 끝]"
        return m.group(0)
    return re.sub(r"\(표:\s*([^)]+\.csv)\)", repl, text)


def load_sections() -> Dict[str, Dict[str, str]]:
    secs = {}
    for p in sorted(KB_DIR.glob("*/*.md")):
        raw = p.read_text(encoding="utf-8")
        text = _inline_tables(p, raw).strip()
        first = text.splitlines()[0].lstrip("# ").strip()
        title = first[len(p.stem):].strip() if first.startswith(p.stem) else first
        secs[p.stem] = {"id": p.stem, "title": title, "text": text}
    return secs


def make_chunks(secs) -> List[Dict[str, str]]:
    chunks = []
    for sid, s in secs.items():
        parts = re.split(r"\n(?=## )", s["text"])
        for i, part in enumerate(parts):
            body = part.strip()
            if i == 0 and len(parts) > 1 and len(body) < 80:
                continue  # 제목 줄만 있는 조각은 건너뜀(다음 조각에 제목을 붙임)
            chunks.append({"sid": sid, "text": f"[{sid} {s['title']}]\n{body}"})
    return chunks


# ---------- BM25 (한국어 형태소 분석기 없이: 한글은 음절 2-gram, 영문·숫자는 단어) ----------
# 연산자 기호(!=, <=, >=, ==, //, **, %)도 토큰으로 둔다. 표 CSV에만 나오는 기호를 묻는 질문("!=는 뭐예요?")이
# 기호가 버려져 엉뚱한 섹션으로 가던 문제(DEVLOG 26장). 일반 기호(<, >, *, 따옴표 등)는 노이즈라 넣지 않는다.
_TOKEN_RE = re.compile(r"[가-힣]+|[a-zA-Z_][a-zA-Z_0-9]*|\d+(?:\.\d+)?|!=|<=|>=|==|//|\*\*|%")


_BOLD_RE = re.compile(r"\*\*(?=\S)(.+?)(?<=\S)\*\*")


def tokenize(text: str) -> List[str]:
    toks = []
    text = _BOLD_RE.sub(r"\1", text)  # 마크다운 굵게(**강조**)는 거듭제곱 기호 `**`와 구별해 벗긴다
    for w in _TOKEN_RE.findall(text.lower()):
        if "가" <= w[0] <= "힣":
            if len(w) == 1:
                toks.append(w)
            else:
                toks.extend(w[i:i + 2] for i in range(len(w) - 1))
        else:
            toks.append(w)
    return toks


class BM25:
    def __init__(self, docs: List[str], k1=1.2, b=0.5):  # b=0.5: 탐색(504조합) 최고, 짝 0.917→0.933 / 홀 0.881→0.898
        self.k1, self.b = k1, b
        self.tfs = [Counter(tokenize(d)) for d in docs]
        self.lens = [sum(tf.values()) for tf in self.tfs]
        self.avg = sum(self.lens) / len(self.lens)
        df = Counter(t for tf in self.tfs for t in tf)
        n = len(docs)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def scores(self, query: str, qtf: bool = True) -> List[float]:
        q = Counter(tokenize(query))
        if not qtf:  # 실험용: 질의 단어 빈도 무시(7f00691 이전 동작)
            q = Counter({t: 1 for t in q})
        out = []
        for i, tf in enumerate(self.tfs):
            s = 0.0
            norm = self.k1 * (1 - self.b + self.b * self.lens[i] / self.avg)
            for t, qc in q.items():
                f = tf.get(t)
                if f:
                    s += self.idf[t] * f * (self.k1 + 1) / (f + norm) * qc  # 질의 단어 빈도 반영(마지막 발화 가중이 실제로 작동하도록)
            out.append(s)
        return out


class KB:
    def __init__(self):
        self.secs = load_sections()
        self.chunks = make_chunks(self.secs)
        self.chunk_sids = [c["sid"] for c in self.chunks]
        self.bm25 = BM25([c["text"] for c in self.chunks])

    def rank(self, query: str, qtf: bool = True) -> List[Tuple[str, float]]:
        """섹션 순위: 섹션마다 청크 BM25 점수의 최댓값, 높은 순."""
        best: Dict[str, float] = defaultdict(lambda: -1e9)
        for sid, sc in zip(self.chunk_sids, self.bm25.scores(query, qtf)):
            if sc > best[sid]:
                best[sid] = sc
        return sorted(best.items(), key=lambda x: -x[1])


def build_query(messages: List[Dict[str, str]], mode: str = "last+users") -> str:
    """검색 질의. 마지막 학습자 발화가 핵심이고, 이전 발화는 맥락 단서."""
    users = [m.get("content", "") for m in messages if m.get("role") == "user"]
    last = users[-1] if users else ""
    if mode == "last":
        return last
    if mode == "last+users":
        return "\n".join(users[:-1][-3:] + [last, last])  # 마지막 발화 가중(두 번)
    if mode == "all":
        return "\n".join(m.get("content", "") for m in messages[-6:]) + "\n" + last
    return last
