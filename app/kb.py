"""지식 베이스 로딩과 하이브리드 검색(BM25 + 임베딩).

- 섹션 = md 파일 하나(+ 같은 이름의 CSV 표를 본문 자리에 끼워 넣음). 인용 단위.
- 청크 = 섹션을 '## ' 소제목 단위로 나눈 조각(앞에 섹션 제목을 붙임). 검색 단위.
- 섹션 점수 = 그 섹션 청크들의 최고 점수.
색인과 임베딩은 서버 시작 때 만든다(첫 요청에서 만들면 시간 초과).
"""
import json
import math
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

KB_DIR = Path(__file__).resolve().parents[1] / "kb"
CACHE = Path(__file__).resolve().parents[1] / ".cache" / "kb_emb.json"


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
_TOKEN_RE = re.compile(r"[가-힣]+|[a-zA-Z_][a-zA-Z_0-9]*|\d+(?:\.\d+)?")


def tokenize(text: str) -> List[str]:
    toks = []
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

    def scores(self, query: str) -> np.ndarray:
        q = Counter(tokenize(query))
        out = np.zeros(len(self.tfs))
        for i, tf in enumerate(self.tfs):
            s = 0.0
            norm = self.k1 * (1 - self.b + self.b * self.lens[i] / self.avg)
            for t, qc in q.items():
                f = tf.get(t)
                if f:
                    s += self.idf[t] * f * (self.k1 + 1) / (f + norm) * qc  # 질의 단어 빈도 반영(마지막 발화 가중이 실제로 작동하도록)
            out[i] = s
        return out


# ---------- 임베딩 ----------
def _emb_client():
    from openai import OpenAI, AsyncOpenAI  # noqa: F401
    return dict(api_key=os.environ["EMBEDDING_API_KEY"], base_url=os.environ["EMBEDDING_BASE_URL"])


EMB_MODEL = os.environ.get("EMBEDDING_MODEL", "openai/text-embedding-3-small")


def embed_sync(texts: List[str]) -> np.ndarray:
    from openai import OpenAI
    c = OpenAI(**_emb_client(), timeout=60, max_retries=2)
    out = []
    for i in range(0, len(texts), 64):
        r = c.embeddings.create(model=EMB_MODEL, input=texts[i:i + 64])
        out.extend(d.embedding for d in r.data)
    v = np.array(out, dtype=np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


_aclient = None


async def embed_async(text: str) -> np.ndarray:
    global _aclient
    from openai import AsyncOpenAI
    if _aclient is None:
        _aclient = AsyncOpenAI(**_emb_client(), timeout=10, max_retries=1)
    r = await _aclient.embeddings.create(model=EMB_MODEL, input=[text])
    v = np.array(r.data[0].embedding, dtype=np.float32)
    return v / np.linalg.norm(v)


class KB:
    def __init__(self, use_embeddings: bool = True):
        self.secs = load_sections()
        self.chunks = make_chunks(self.secs)
        self.chunk_sids = [c["sid"] for c in self.chunks]
        self.sids = list(self.secs)
        self.bm25 = BM25([c["text"] for c in self.chunks])
        self.emb = self._load_or_build_emb() if use_embeddings else None

    def _load_or_build_emb(self) -> np.ndarray:
        texts = [c["text"] for c in self.chunks]
        key = str(hash(tuple(texts)))
        try:
            data = json.loads(CACHE.read_text())
            if data.get("key") == key and data.get("model") == EMB_MODEL:
                return np.array(data["emb"], dtype=np.float32)
        except Exception:  # noqa: BLE001
            pass
        emb = embed_sync(texts)
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        CACHE.write_text(json.dumps({"key": key, "model": EMB_MODEL, "emb": emb.tolist()}))
        return emb

    def _section_scores(self, chunk_scores: np.ndarray) -> Dict[str, float]:
        best = defaultdict(lambda: -1e9)
        for sid, s in zip(self.chunk_sids, chunk_scores):
            if s > best[sid]:
                best[sid] = float(s)
        return best

    def rank(self, query: str, qvec: Optional[np.ndarray] = None, k_rrf: int = 60,
             w_bm25: float = 1.0, w_emb: float = 1.0) -> List[Tuple[str, float]]:
        """하이브리드 순위(RRF). qvec이 없으면 BM25만."""
        fused = defaultdict(float)
        lists = [(self._section_scores(self.bm25.scores(query)), w_bm25)]
        if qvec is not None and self.emb is not None:
            lists.append((self._section_scores(self.emb @ qvec), w_emb))
        for scores, w in lists:
            for r, (sid, _) in enumerate(sorted(scores.items(), key=lambda x: -x[1])):
                fused[sid] += w / (k_rrf + r + 1)
        return sorted(fused.items(), key=lambda x: -x[1])


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
