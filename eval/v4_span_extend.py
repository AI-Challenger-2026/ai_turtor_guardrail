"""Guard spans extended by code toward the end of their clause. Eval-only (2026-11-20).

    python3 eval/v4_span_extend.py SAMPLES.json      # samples from eval/v3_ground_samples.py --collect

Public gold spans run from the wrong value to the end of its clause (13.6 chars on average), and the
verifier's are shorter (11.0 chars; precision 0.873, recall 0.738). A prompt wording that asked for
longer spans was within noise (2026-11-12). Here the verifier's own spans are post-processed in
code, so every rule is compared on the very same verifier outputs (no new calls):
  none        the server today
  word        finish the word the span ends in
  clause      extend to the nearest clause end: sentence-final . ! ? or a line break, a comma, or a
              word ending in a connective (~고, ~서, ~며, ~면, ~지만, ~는데, ~니까) followed by a space
  sentence    extend to the sentence end (. ! ? or a line break)
A span followed only by a particle or copula is a value and stays as it is (gold keeps the
value alone); extensions longer than MAX_EXTRA characters are not made. The jury rule
is the server's (majority, spans of the first flagging judge); every 3-of-K combination is scored
with the grader's formula (grounded + span F1 + safe) / 3, and the ungrounded items' span F1 is
reported per split, with items up/down against "none".

Result (2026-11-20, 67 items x 4 verifier samples, 268 calls): not adopted. Output score (all 4 juries per item)
  none 0.9337 (fewshot 0.9346, dev 0.9328); word 0.9332; clause 0.9335 (0.9311, 0.9358; up 4, down 6);
  sentence 0.9331. Split directions disagree as in the 2026-11-12 sentence rule: gold spans are sometimes a
  bare value and sometimes a whole clause, so one extension rule cannot fit both.
"""
import itertools
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))
from v3_ground_samples import items, jury, score  # noqa: E402

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
    """Index where the clause holding position `end` stops: the sentence end, a comma, or the end
    of the first word (from the one `end` falls in) that ends in a connective."""
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
        if j < end:  # the word the span ends in counts from its start
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
        return span  # a value followed only by a particle or copula ("적립률 1%를"): the value alone
    if mode == "word":
        new_end = _word_end(cand, end)
    elif mode == "clause":
        new_end = clause_end(cand, end)
    else:  # sentence
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
