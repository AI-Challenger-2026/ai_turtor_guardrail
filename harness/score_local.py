import argparse
import json
import math
import os
import secrets
import socket
import sys
import tempfile
import time
import urllib.request
from collections import Counter, defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from http.client import HTTPException
from pathlib import Path
from threading import Lock
from urllib.error import HTTPError, URLError

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'grader'))
import scoring
from judge import make_judge

ATTACK_CATEGORIES = {'jailbreak_injection', 'harmful_content', 'off_topic', 'academic_dishonesty'}
ENDPOINTS = ('tutor', 'guard')
PUBLIC_ITEMS_PER_ENDPOINT = 200
PROGRESS_EVERY = 20
WEIGHTS = {'tutor': 0.5, 'guard': 0.5}
TUTOR_DIAGNOSTICS = ('action_correct', 'citation_p', 'citation_r', 'facts', 'faithful',
                     'complied', 'over_refusal', 'judge_error')
GUARD_DIAGNOSTICS = ('input_correct', 'category_correct', 'category_scored', 'false_block',
                     'missed_block', 'gold_block', 'has_output', 'output_correct', 'span_f1', 'safe_correct')

class JudgeSkipped(Exception):
    pass


class ConfigurationError(ValueError):
    pass


def as_scoring_input(public_item):
    p, labels = (public_item, public_item['labels'])
    variant = {'variant_id': p['id'], 'messages': p['messages'], 'candidate_answer': p.get('candidate_answer'), 'gold_facts': labels.get('gold_facts') or [], 'unsupported_spans': labels.get('unsupported') or [], 'safe': labels.get('safe', True)}
    if p['endpoint'] == 'tutor':
        item_labels = {'action': labels['action'], 'evidence': labels.get('evidence') or []}
    else:
        item_labels = {'decision': labels['decision']}
        if 'candidate_answer' in p:
            item_labels['output_label'] = 'unsafe' if not labels['safe'] else 'grounded' if labels['grounded'] else 'ungrounded'
    item = {'intent_id': p['id'], 'endpoint': p['endpoint'], 'category': labels['category'], 'kind': 'attack' if labels['category'] in ATTACK_CATEGORIES else 'normal', 'labels': item_labels, 'variants': [variant]}
    return (item, variant)

def _no_judge(kind, payload):
    raise JudgeSkipped

def _request_error(exc):
    if isinstance(exc, HTTPError):
        return 'http', exc.code
    cause = exc.reason if isinstance(exc, URLError) else exc
    if isinstance(cause, (TimeoutError, socket.timeout)):
        return 'timeout', None
    if isinstance(exc, (json.JSONDecodeError, UnicodeDecodeError)):
        return 'invalid_json', None
    if isinstance(exc, (OSError, HTTPException)):
        return 'network', None
    return 'request_error', None


def _diagnose(item, response, result, request, grading_ms):
    if not result['valid'] and request['error'] is None:
        request['error'] = 'invalid_schema'
    details = {'request': request, 'grading_ms': grading_ms}
    if item['endpoint'] == 'tutor':
        fields = TUTOR_DIAGNOSTICS
        parsed = scoring.parse_tutor(response)
        details['expected_action'] = item['labels']['action']
        details['predicted_action'] = parsed['action'] if parsed else None
    else:
        fields = GUARD_DIAGNOSTICS
        parsed = scoring.parse_guard(response, result['has_output'])
        details['expected_decision'] = item['labels']['decision']
        details['predicted_decision'] = parsed['input']['decision'] if parsed else None
        details['predicted_category'] = parsed['input']['category'] if parsed else None
    details['grading'] = {field: result[field] for field in fields}
    return details


def score_local(public, call, judge, kb, workers=8, *, judge_workers=None, diagnostics=False, on_item_done=None):
    def fetch(p):
        item, v = as_scoring_input(p)
        payload = {'spec_version': '1.0', 'session_id': 's_' + secrets.token_hex(8), 'messages': v['messages']}
        if v['candidate_answer']:
            payload['candidate_answer'] = v['candidate_answer']
        started = time.perf_counter()
        error = http_status = None
        try:
            resp = call(p['endpoint'], payload)
        except Exception as exc:
            resp = None
            error, http_status = _request_error(exc)
        request = {
            'elapsed_ms': round((time.perf_counter() - started) * 1000, 2),
            'error': error,
            'http_status': http_status,
        }
        return p, item, v, resp, request

    def grade(p, item, v, resp, request):
        started = time.perf_counter()
        if p['endpoint'] == 'tutor':
            r = scoring.score_tutor(item, v, resp, judge or _no_judge, kb)
            if judge is None:
                r['judge_error'] = False
            points = scoring.tutor_points(r)
        else:
            r = scoring.score_guard(item, v, resp)
            points = scoring.guard_points(r)
        detail = None
        if diagnostics:
            grading_ms = round((time.perf_counter() - started) * 1000, 2)
            detail = _diagnose(item, resp, r, request, grading_ms)
        if on_item_done is not None:
            on_item_done()
        return p, r, points, detail

    def one(p):
        return grade(*fetch(p))

    if judge_workers is None or judge is None:
        with ThreadPoolExecutor(workers) as requests:
            results = list(requests.map(one, public))
    else:
        with ThreadPoolExecutor(workers) as requests, ThreadPoolExecutor(judge_workers) as judges:
            pending = {requests.submit(fetch, p): i for i, p in enumerate(public)}
            results = [None] * len(pending)
            grading = {}
            for future in as_completed(pending):
                i = pending.pop(future)
                fetched = future.result()
                if fetched[0]['endpoint'] == 'tutor':
                    grading[judges.submit(grade, *fetched)] = i
                else:
                    results[i] = grade(*fetched)
            for future in as_completed(grading):
                results[grading.pop(future)] = future.result()

    tutor = [r for p, r, _, _ in results if p['endpoint'] == 'tutor']
    guard = [r for p, r, _, _ in results if p['endpoint'] == 'guard']
    agg = scoring.aggregate(tutor, guard, WEIGHTS)
    per = defaultdict(lambda: defaultdict(list))
    for p, r, pts, _ in results:
        if not r.get('judge_error'):
            per[p['endpoint']][p['labels']['category']].append(pts)
    per_category = {e: {c: round(100 * sum(v) / len(v), 2) for c, v in sorted(cats.items())} for e, cats in per.items()}
    items = [{'id': p['id'], 'endpoint': p['endpoint'], 'category': p['labels']['category'], 'points': round(100 * pts, 1), 'valid': r['valid']} for p, r, pts, _ in results]
    out = {**agg, 'per_category': per_category, 'tutor_judge_skipped': judge is None, 'items': items}
    if diagnostics:
        out['details'] = [{**summary, **result[3]} for summary, result in zip(items, results)]
    return out

def load_kb_dir(root):
    kb = {}
    for p in sorted(Path(root).glob('*/*.md')):
        tables = sorted(p.parent.glob(f'{p.stem}-table*.csv'))
        kb[p.stem] = '\n\n'.join([p.read_text(encoding='utf-8'), *(t.read_text(encoding='utf-8') for t in tables)])
    return kb

def _read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def _read_public(root, *, full=False):
    root = Path(root)
    if full:
        root = root.resolve()
        if root != ROOT / 'public':
            raise ConfigurationError('--full은 저장소의 public 공개 세트를 사용해야 합니다.')
    public = []
    for endpoint in ENDPOINTS:
        path = root / f'{endpoint}.jsonl'
        if full and path.resolve() != path:
            raise ConfigurationError(f'{endpoint} 공개 문항 파일 경로가 올바르지 않습니다.')
        try:
            rows = _read_jsonl(path)
        except (OSError, ValueError) as exc:
            if not full:
                raise
            raise ConfigurationError(f'{endpoint} 공개 문항 파일을 읽을 수 없습니다 ({type(exc).__name__}).') from None
        if full:
            if len(rows) != PUBLIC_ITEMS_PER_ENDPOINT:
                raise ConfigurationError(
                    f'{endpoint} 공개 문항은 {PUBLIC_ITEMS_PER_ENDPOINT}개여야 합니다 (현재 {len(rows)}개).')
            if any(not isinstance(row, dict) or row.get('endpoint') != endpoint
                   or not isinstance(row.get('id'), str) or not row['id'].strip() for row in rows):
                raise ConfigurationError(f'{endpoint} 공개 문항의 엔드포인트 또는 ID가 올바르지 않습니다.')
            if len({row['id'] for row in rows}) != len(rows):
                raise ConfigurationError(f'{endpoint} 공개 문항 ID가 중복되었습니다.')
        public.extend(rows)
    return public


def _report_path(value):
    path = Path(value).resolve()
    name = path.name.lower()
    if ROOT / 'logs' not in path.parents or path.suffix.lower() != '.json':
        raise ConfigurationError('리포트 경로는 저장소 logs/ 아래의 JSON 파일이어야 합니다.')
    if name == 'submission.json' or name.startswith('.env.'):
        raise ConfigurationError('리포트에는 일반 JSON 파일 이름을 사용해야 합니다.')
    if path.exists():
        raise ConfigurationError('리포트 경로가 이미 존재합니다. 새 경로를 지정하세요.')
    return path


def _execution_stats(events):
    samples = sorted(event['elapsed_ms'] for event in events)
    return {
        'count': len(samples),
        'p50_ms': round(samples[math.ceil(len(samples) * 0.5) - 1], 2) if samples else None,
        'p95_ms': round(samples[math.ceil(len(samples) * 0.95) - 1], 2) if samples else None,
        'errors': dict(Counter(event['error'] for event in events if event['error'] is not None)),
        'http_errors': dict(Counter(str(event['http_status']) for event in events if event['http_status'] is not None)),
    }


def _make_report(out, attempts, elapsed_ms, workers, judge_workers):
    details = out['details']
    requests = {endpoint: _execution_stats([row['request'] for row in details if row['endpoint'] == endpoint])
                for endpoint in ENDPOINTS}
    by_kind = defaultdict(list)
    for event in attempts:
        by_kind[event['kind']].append(event)
    return {
        'schema_version': 1,
        'dataset': 'public',
        'workers': workers,
        'judge_workers': judge_workers,
        'elapsed_ms': round(elapsed_ms, 2),
        'completed': {endpoint: requests[endpoint]['count'] for endpoint in ENDPOINTS},
        'scores': {key: out[key] for key in ('total', 'tutor', 'guard', 'invalid', 'judge_errors')},
        'per_category': out['per_category'],
        'requests': requests,
        'judge': {
            **_execution_stats(attempts),
            'retries': sum(event['attempt'] > 1 for event in attempts),
            'failed_items': out['judge_errors'],
            'by_kind': {kind: _execution_stats(events) for kind, events in sorted(by_kind.items())},
        },
        'items': details,
    }


def _write_report(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent, delete=False) as f:
            temporary = Path(f.name)
            json.dump(report, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.write('\n')
        # 완성된 JSON을 공개하고 동시에 같은 이름을 사용한 실행의 결과를 덮어쓰지 않는다.
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description='공개 세트 로컬 채점')
    ap.add_argument('--tutor-url', required=True)
    ap.add_argument('--guard-url', required=True)
    ap.add_argument('--token', default='')
    ap.add_argument('--public', default='public')
    ap.add_argument('--kb', default='kb')
    ap.add_argument('--timeout', type=float, default=30)
    ap.add_argument('--workers', type=int, default=8, help='서버 요청 작업자 수 (판정 풀이 없으면 요청과 판정에 공유)')
    ap.add_argument('--full', action='store_true', help='공개 400문항과 판정 모델 설정을 검사')
    ap.add_argument('--judge-workers', type=int, help='서버 요청 풀과 별도로 실행할 튜터 판정 작업자 수')
    ap.add_argument('--report', help='전체 채점 진단을 저장할 logs/ 아래의 JSON 경로')
    a = ap.parse_args(argv)
    started = time.perf_counter()
    if a.workers < 1 or (a.judge_workers is not None and a.judge_workers < 1):
        ap.error('작업자 수는 1 이상이어야 합니다.')
    if a.report is not None and not a.full:
        ap.error('--report는 --full과 함께 사용해야 합니다.')
    env = {k: os.environ.get(f'OPENAI_{k.upper()}') for k in ('base_url', 'model', 'api_key')}
    if a.full and not a.token.strip():
        ap.error('--full에서는 --token에 TEAM_TOKEN을 전달해야 합니다.')
    if a.full or a.judge_workers is not None:
        missing = [f'OPENAI_{k.upper()}' for k, v in env.items() if not v or not v.strip()]
        if missing:
            ap.error('판정 모델 환경변수를 설정해야 합니다 (' + ', '.join(missing) + ').')
    try:
        report_path = _report_path(a.report) if a.report is not None else None
        public = _read_public(a.public, full=a.full)
    except ConfigurationError as exc:
        ap.error(str(exc))
    urls = {'tutor': a.tutor_url, 'guard': a.guard_url}
    opener = urllib.request.build_opener()

    def call(endpoint, payload):
        req = urllib.request.Request(urls[endpoint], data=json.dumps(payload, ensure_ascii=False).encode('utf-8'), method='POST', headers={'Content-Type': 'application/json', 'Authorization': f'Bearer {a.token}'})
        with opener.open(req, timeout=a.timeout) as resp:
            return json.loads(resp.read(256 * 1024 + 1))

    attempts = deque()
    judge = make_judge(**env, on_attempt=attempts.append if report_path else None) if all(env.values()) else None
    total = len(public)
    completed = 0
    progress_lock = Lock()

    def write_progress(done):
        print(f'채점 진행 {done}/{total} ({done / total:.0%})', file=sys.stderr, flush=True)

    def item_done():
        nonlocal completed
        with progress_lock:
            completed += 1
            if completed == total or completed % PROGRESS_EVERY == 0:
                write_progress(completed)

    if a.full:
        write_progress(0)
    out = score_local(public, call, judge, load_kb_dir(a.kb), workers=a.workers,
                      judge_workers=a.judge_workers, diagnostics=report_path is not None,
                      on_item_done=item_done if a.full else None)
    elapsed_ms = (time.perf_counter() - started) * 1000
    for it in out['items']:
        row = [it['id'], it['endpoint'], it['category'], str(it['points'])] + ([] if it['valid'] else ['형식 오류'])
        print('\t'.join(row))
    print('범주별:', json.dumps(out['per_category'], ensure_ascii=False))
    print(f"총점 {out['total']} (튜터 {out['tutor']}, 가드 {out['guard']}), 형식 오류 {out['invalid']}, 판정 오류 {out['judge_errors']}" + (' — 판정 모델 없음: 튜터는 action·인용만 채점' if judge is None else ''))
    if a.full:
        print(f"공개 문항 {len(out['items'])}/{total}개 처리 (총 소요 시간 {elapsed_ms / 1000:.2f}초)", flush=True)
    if report_path is not None:
        report = _make_report(out, attempts, elapsed_ms, a.workers, a.judge_workers)
        try:
            _write_report(report_path, report)
        except OSError as exc:
            ap.exit(1, f'리포트를 저장할 수 없습니다 ({type(exc).__name__}).\n')
        print(f'진단 리포트 {report_path.relative_to(ROOT)}')


if __name__ == '__main__':
    main()
