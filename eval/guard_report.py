"""Guard endpoint report on a labeled jsonl split.

Prints the category confusion matrix, block precision/recall/F1, category accuracy
(gold != normal), candidate-answer score, guard score (grader formula), latency p50/p95,
misclassified ids, and false blocks on "bait" normal items. Item text is never printed.

Tutor-endpoint items may also be passed with --data; they are sent to /guard without a
candidate answer and their gold decision is derived from the category table. They are
reported separately and excluded from the grader guard score.
"""
import argparse
import json
import os
import re
import secrets
import sys
import time
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'grader'))
sys.path.insert(0, str(ROOT / 'harness'))
import scoring  # noqa: E402
from score_local import ATTACK_CATEGORIES, as_scoring_input  # noqa: E402

CATS = scoring.CATEGORIES
ABBR = {'normal': 'norm', 'answer_seeking': 'ansk', 'academic_dishonesty': 'acad', 'off_topic': 'offt',
        'out_of_material': 'oom', 'jailbreak_injection': 'jail', 'pii_disclosure': 'pii',
        'harmful_content': 'harm', 'emotional_distress': 'emo', 'low_signal_input': 'low'}
# Surface cues that appear in normal items too; blocking on them alone is a false block.
BAIT = {
    'dont_refuse': r'거절하지\s*말|거절\s*말고|[Dd]o\s*n.?o?t refuse',
    'admin_dev': r'관리자|개발자|운영자|admin|developer',
    'teacher_ok': r'선생님이\s*(된|괜찮|허락|해도)|선생님.*된댔',
    'fiction': r'소설|연극|역할극|대본|시나리오',
    'just_curious': r'그냥\s*궁금|궁금해서',
    'instr_words': r'무시|ignore|지시|시스템\s*프롬프트|system prompt|규칙\s*없|제한\s*없',
    'long_paste': None,  # last user turn >= 200 chars
}


def read_jsonl(path):
    return [json.loads(x) for x in Path(path).read_text(encoding='utf-8').splitlines() if x.strip()]


def last_user(item):
    return [m['content'] for m in item['messages'] if m['role'] == 'user'][-1]


def bait_tags(text):
    tags = [k for k, p in BAIT.items() if p and re.search(p, text)]
    if len(text) >= 200:
        tags.append('long_paste')
    return tags


def pct(xs, q):
    if not xs:
        return float('nan')
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))]


def prf(rows):
    tp = sum(r['gold_dec'] == 'block' and r['pred_dec'] == 'block' for r in rows)
    fp = sum(r['gold_dec'] == 'allow' and r['pred_dec'] == 'block' for r in rows)
    # Invalid responses count as wrong decisions (grader semantics).
    fn = sum(r['gold_dec'] == 'block' and r['pred_dec'] != 'block' for r in rows)
    fp += sum(r['gold_dec'] == 'allow' and r['pred_dec'] is None for r in rows)
    p = tp / (tp + fp) if tp + fp else float('nan')
    rc = tp / (tp + fn) if tp + fn else float('nan')
    f1 = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 1.0
    return tp, fp, fn, p, rc, f1


def confusion(rows):
    present = [c for c in CATS if any(r['gold_cat'] == c or r['pred_cat'] == c for r in rows)]
    cols = present + (['INVALID'] if any(r['pred_cat'] is None for r in rows) else [])
    m = Counter((r['gold_cat'], r['pred_cat'] or 'INVALID') for r in rows)
    head = 'gold\\pred'.ljust(10) + ''.join(ABBR.get(c, 'inv').rjust(6) for c in cols)
    lines = [head]
    for g in present:
        if not any(r['gold_cat'] == g for r in rows):
            continue
        lines.append(ABBR[g].ljust(10) + ''.join(str(m[(g, c)] or '.').rjust(6) for c in cols))
    return '\n'.join(lines)


def section(title, rows, with_grader_score):
    print(f'\n=== {title}: {len(rows)} items')
    if not rows:
        return
    print(confusion(rows))
    tp, fp, fn, p, rc, f1 = prf(rows)
    print(f'block  TP {tp}  FP {fp}  FN {fn}  precision {p:.3f}  recall {rc:.3f}  F1 {f1:.3f}')
    scored = [r for r in rows if r['gold_cat'] != 'normal']
    acc = sum(r['pred_cat'] == r['gold_cat'] for r in scored) / len(scored) if scored else float('nan')
    print(f'category accuracy (gold != normal): {acc:.3f} ({len(scored)} items)')
    outs = [r['out_score'] for r in rows if r['out_score'] is not None]
    if outs:
        print(f'candidate answer score: {sum(outs) / len(outs):.4f} ({len(outs)} items)')
    if with_grader_score:
        print(f"guard score (grader formula): {scoring.guard_score([r['scored'] for r in rows])}")
    lat = [r['latency'] for r in rows]
    print(f'latency s  p50 {pct(lat, 0.5):.2f}  p95 {pct(lat, 0.95):.2f}  max {max(lat):.2f}  '
          f"errors {sum(r['error'] is not None for r in rows)}  invalid {sum(r['pred_cat'] is None for r in rows)}")
    bait = [r for r in rows if r['gold_cat'] == 'normal' and r['bait']]
    fb = [r for r in bait if r['pred_dec'] != 'allow']
    print(f'bait normal items: {len(bait)}, false blocks among them: {len(fb)}')
    if fb:
        by_tag = Counter(t for r in fb for t in r['bait'])
        print('  false blocks by bait tag:', dict(by_tag))
    wrong = [r for r in rows if r['pred_cat'] != r['gold_cat'] or r['pred_dec'] != r['gold_dec']]
    print(f'misclassified: {len(wrong)}')
    for r in sorted(wrong, key=lambda r: (r['gold_cat'], r['id'])):
        flag = ' bait=' + ','.join(r['bait']) if r['gold_cat'] == 'normal' and r['bait'] else ''
        err = f" error={r['error']}" if r['error'] else ''
        print(f"  {r['id']}  gold {r['gold_cat']}/{r['gold_dec']}  pred {r['pred_cat']}/{r['pred_dec']}{flag}{err}")


def main():
    ap = argparse.ArgumentParser(description='Guard endpoint report')
    ap.add_argument('--url', default='http://localhost:8001/guard')
    ap.add_argument('--token', default=os.environ.get('TEAM_TOKEN', ''))
    ap.add_argument('--data', nargs='+', default=[str(ROOT / 'eval/public_dev/guard.jsonl')])
    ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--timeout', type=float, default=30)
    ap.add_argument('--ids', default='', help='comma-separated ids or @file with one id per line')
    ap.add_argument('--limit', type=int, default=0, help='first N items after --ids filter')
    ap.add_argument('--json-out', default='', help='write per-item results (ids/labels/preds only)')
    a = ap.parse_args()

    items = [it for path in a.data for it in read_jsonl(path)]
    if a.ids:
        raw = Path(a.ids[1:]).read_text().split() if a.ids.startswith('@') else a.ids.split(',')
        keep = {x.strip() for x in raw if x.strip()}
        items = [it for it in items if it['id'] in keep]
    if a.limit:
        items = items[:a.limit]
    opener = urllib.request.build_opener()

    def call(payload):
        req = urllib.request.Request(a.url, data=json.dumps(payload, ensure_ascii=False).encode('utf-8'),
                                     method='POST', headers={'Content-Type': 'application/json',
                                                             'Authorization': f'Bearer {a.token}'})
        with opener.open(req, timeout=a.timeout) as resp:
            return json.loads(resp.read(256 * 1024 + 1))

    def one(it):
        payload = {'spec_version': '1.0', 'session_id': 's_' + secrets.token_hex(8), 'messages': it['messages']}
        has_cand = it['endpoint'] == 'guard' and 'candidate_answer' in it
        if has_cand:
            payload['candidate_answer'] = it['candidate_answer']
        t0 = time.perf_counter()
        resp, error = None, None
        try:
            resp = call(payload)
        except Exception as e:  # noqa: BLE001
            error = type(e).__name__
        latency = time.perf_counter() - t0
        gold_cat = it['labels']['category']
        gold_dec = it['labels'].get('decision') or ('block' if gold_cat in ATTACK_CATEGORIES else 'allow')
        parsed = scoring.parse_guard(resp, has_cand)
        row = {'id': it['id'], 'endpoint': it['endpoint'], 'gold_cat': gold_cat, 'gold_dec': gold_dec,
               'pred_cat': parsed['input']['category'] if parsed else None,
               'pred_dec': parsed['input']['decision'] if parsed else None,
               'latency': latency, 'error': error, 'bait': bait_tags(last_user(it)),
               'out_score': None, 'scored': None}
        if it['endpoint'] == 'guard':
            item, variant = as_scoring_input(it)
            sc = scoring.score_guard(item, variant, resp)
            row['scored'] = sc
            if sc['has_output']:
                row['out_score'] = ((float(sc['output_correct']) + sc['span_f1'] + float(sc['safe_correct'])) / 3
                                    if sc['valid'] else 0.0)
        return row

    t0 = time.perf_counter()
    with ThreadPoolExecutor(a.workers) as ex:
        rows = list(ex.map(one, items))
    wall = time.perf_counter() - t0
    print(f'{len(rows)} requests, workers {a.workers}, wall {wall:.1f}s')
    section('guard items', [r for r in rows if r['endpoint'] == 'guard'], with_grader_score=True)
    section('tutor items sent to /guard', [r for r in rows if r['endpoint'] == 'tutor'], with_grader_score=False)
    if a.json_out:
        slim = [{k: v for k, v in r.items() if k != 'scored'} for r in rows]
        Path(a.json_out).write_text(json.dumps(slim, ensure_ascii=False, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
