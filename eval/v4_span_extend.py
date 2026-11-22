"""Candidate-check spans extended by code (not used by the server).

    python3 eval/v4_span_extend.py SAMPLES.json
"""
import itertools
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))
from v3_ground_samples import items, jury, score

SENT_END = re.compile(r"[.!?。]|\n")
MAX_EXTRA = 20
PARTICLE = re.compile(r"(을|를|이|가|은|는|에|에서|의|로|으로|와|과|도|만|이에요|예요|에요|이야|야|이다|다)")


def _word_end(text, i):
    while i < len(text) and not text[i].isspace() and text[i] not in ".,!?。'\"’”`)":
        i += 1
    return i


def ends_with_connective(word):
    return bool(re.search(r"(고|서|며|면|지만|는데|니까),?$", word))


def clause_end(cand, end):
    m = SENT_END.search(cand, end)
    limit = m.start() if m else len(cand)
    c = cand.find(",", end)
    if 0 <= c < limit:
        limit = c
    j = end
    while j < limit:
        while j < limit and cand[j].isspace():
            j += 1
        e = min(_word_end(cand, j), limit)
        if e == j:
            break
        word = cand[j:e]
        if j < end:
            word = cand[cand.rfind(" ", 0, end) + 1:e]
        if ends_with_connective(word):
            return e
        j = e
    return limit


def extend(span, cand, mode):
    if mode == "none":
        return span
    i = cand.find(span)
    if i < 0:
        return span
    end = i + len(span)
    rest = cand[end:_word_end(cand, end)]
    if rest and PARTICLE.fullmatch(rest):
        return span
    if mode == "word":
        new_end = _word_end(cand, end)
    elif mode == "clause":
        new_end = clause_end(cand, end)
    else:
        m = SENT_END.search(cand, end)
        new_end = m.start() if m else len(cand)
    new_end = max(new_end, end)
    if new_end - end > MAX_EXTRA:
        return span
    return cand[i:new_end].rstrip(" ,")


def apply(spans, cand, mode):
    return [extend(s, cand, mode) for s in spans]


def main():
    path = sys.argv[1]
    data = {d["id"]: d for d in json.loads(Path(path).read_text(encoding="utf-8"))}
    rows = [r for r in items() if r["id"] in data]
    modes = ("none", "word", "clause", "sentence")
    per_item = {m: {} for m in modes}
    for r in rows:
        cand = r["candidate_answer"]
        samples = data[r["id"]]["samples"]
        combos = list(itertools.combinations(samples, 3))
        for mode in modes:
            outs = [apply(jury(c, "first"), cand, mode) for c in combos]
            sc = [score(r, o) for o in outs]
            per_item[mode][r["id"]] = (sum(x[0] for x in sc) / len(sc), sum(x[2] for x in sc) / len(sc))
    for split in ("fewshot", "dev", "all"):
        sel = [r for r in rows if split == "all" or r["split"] == split]
        ung = [r for r in sel if not r["labels"]["grounded"]]
        print(f"[{split}] items {len(sel)}, ungrounded {len(ung)}")
        for mode in modes:
            s = sum(per_item[mode][r["id"]][0] for r in sel) / len(sel)
            f = sum(per_item[mode][r["id"]][1] for r in ung) / max(1, len(ung))
            up = sum(per_item[mode][r["id"]][0] > per_item["none"][r["id"]][0] + 1e-9 for r in sel)
            down = sum(per_item[mode][r["id"]][0] < per_item["none"][r["id"]][0] - 1e-9 for r in sel)
            print(f"  {mode:9s} output score {s:.4f}  span F1 (ungrounded) {f:.3f}  items up {up:2d} down {down:2d}")


if __name__ == "__main__":
    main()
