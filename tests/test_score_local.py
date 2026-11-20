import contextlib
import io
import json
import shutil
import tempfile
import threading
import unittest
from collections import Counter, deque
from pathlib import Path
from unittest import mock
from urllib.error import HTTPError, URLError

from harness import score_local as local
import judge as judge_module


def public_item(endpoint, item_id):
    labels = {'category': 'normal'}
    if endpoint == 'tutor':
        labels['action'] = 'answer'
    else:
        labels['decision'] = 'allow'
    return {'id': item_id, 'endpoint': endpoint, 'messages': [{'role': 'user', 'content': item_id}], 'labels': labels}


def fixed_response(endpoint, payload):
    if endpoint == 'tutor':
        return {'answer': '리포트본문검사용', 'action': 'answer', 'citations': []}
    response = {'input': {'decision': 'allow', 'category': 'normal'}}
    if 'candidate_answer' in payload:
        response['output'] = {'grounded': True, 'unsupported': [], 'safe': True}
    return response


def fixed_judge(kind, payload):
    return {'facts': {'score': 0.5}, 'faithful': {'ok': False}, 'complied': {'complied': False}}[kind]


def chat_response(data):
    content = {'choices': [{'message': {'content': json.dumps(data)}}]}
    response = io.BytesIO(json.dumps(content).encode('utf-8'))
    response.headers = {'Content-Type': 'application/json'}
    return response


class ScoreLocalTests(unittest.TestCase):
    def test_both_schedulers_process_and_score_all_public_items(self):
        public = local._read_public(local.ROOT / 'public', full=True)
        scores = []
        for judge_workers in (None, 4):
            requests = deque()
            judges = deque()
            completed = deque()

            def call(endpoint, payload):
                requests.append(endpoint)
                return fixed_response(endpoint, payload)

            def judge(kind, payload):
                judges.append(kind)
                return fixed_judge(kind, payload)

            out = local.score_local(public, call, judge, {}, workers=8, judge_workers=judge_workers,
                                    on_item_done=lambda: completed.append(True))
            self.assertEqual(Counter(requests), {'tutor': 200, 'guard': 200})
            self.assertEqual(Counter(judges), {'facts': 119, 'faithful': 147, 'complied': 44})
            self.assertEqual(len(completed), 400)
            self.assertEqual([row['id'] for row in out['items']], [row['id'] for row in public])
            scores.append(out)
        self.assertEqual(*scores)

    def test_guard_request_runs_while_judge_worker_waits(self):
        judge_started = threading.Event()
        guard_received = threading.Event()

        def call(endpoint, payload):
            if endpoint == 'guard':
                if not judge_started.wait(5):
                    raise TimeoutError()
                guard_received.set()
            return fixed_response(endpoint, payload)

        def judge(kind, payload):
            judge_started.set()
            if not guard_received.wait(5):
                raise TimeoutError()
            return fixed_judge(kind, payload)

        public = [public_item('tutor', 't'), public_item('guard', 'g')]
        with mock.patch.object(local, 'ThreadPoolExecutor', wraps=local.ThreadPoolExecutor) as pools:
            out = local.score_local(public, call, judge, {}, workers=1, judge_workers=1)
        self.assertEqual(pools.call_args_list, [mock.call(1), mock.call(1)])
        self.assertTrue(guard_received.is_set())
        self.assertEqual(out['invalid'], 0)
        self.assertEqual(out['judge_errors'], 0)

    def test_report_distinguishes_request_and_judge_failures(self):
        cases = ['http', 'timeout', 'invalid_json', 'invalid_schema', 'judge']
        public = [public_item('tutor', case) for case in cases] + [public_item('guard', 'success')]
        error_text = '리포트오류본문검사용'

        def call(endpoint, payload):
            case = payload['messages'][0]['content']
            if case == 'http':
                raise HTTPError('https://example.invalid', 503, error_text, None, None)
            if case == 'timeout':
                raise URLError(TimeoutError(error_text))
            if case == 'invalid_json':
                raise json.JSONDecodeError(error_text, '내용', 0)
            if case == 'invalid_schema':
                return {'answer': error_text}
            return fixed_response(endpoint, payload)

        def judge(kind, payload):
            raise local.scoring.JudgeError(error_text)

        completed = deque()
        out = local.score_local(public, call, judge, {}, judge_workers=2, diagnostics=True,
                                on_item_done=lambda: completed.append(True))
        report = local._make_report(out, [], 100, 8, 2)
        self.assertEqual(len(completed), len(public))
        self.assertEqual(report['completed'], {'tutor': 5, 'guard': 1})
        self.assertEqual(report['requests']['tutor']['errors'], dict.fromkeys(cases[:-1], 1))
        self.assertEqual(report['requests']['tutor']['http_errors'], {'503': 1})
        self.assertEqual(report['scores']['invalid'], 4)
        self.assertEqual(report['judge']['failed_items'], 1)
        self.assertTrue(report['items'][4]['grading']['judge_error'])
        self.assertIsNone(report['items'][4]['request']['error'])
        self.assertTrue(report['items'][5]['grading']['input_correct'])
        serialized = json.dumps(report, ensure_ascii=False)
        self.assertNotIn(error_text, serialized)
        self.assertNotIn('리포트본문검사용', serialized)
        self.assertNotIn('session_id', serialized)

    def test_full_mode_checks_public_path_count_and_ids(self):
        with self.assertRaisesRegex(ValueError, 'public 공개 세트'):
            local._read_public(local.ROOT / 'eval' / 'public_dev', full=True)
        tutor = public_item('tutor', 'duplicate')
        with mock.patch.object(local, '_read_jsonl', return_value=[tutor]):
            with self.assertRaisesRegex(ValueError, '200개'):
                local._read_public(local.ROOT / 'public', full=True)
        with mock.patch.object(local, '_read_jsonl', return_value=[tutor] * 200):
            with self.assertRaisesRegex(ValueError, 'ID가 중복'):
                local._read_public(local.ROOT / 'public', full=True)

    def test_full_mode_requires_judge_settings_before_network(self):
        error = io.StringIO()
        with mock.patch.object(local.os, 'environ', {}), contextlib.redirect_stderr(error):
            with mock.patch.object(local.urllib.request, 'build_opener') as opener:
                with self.assertRaises(SystemExit) as exc:
                    local.main(['--full', '--token', 'fixture', '--tutor-url', 'http://localhost/tutor',
                                '--guard-url', 'http://localhost/guard'])
        self.assertEqual(exc.exception.code, 2)
        self.assertIn('OPENAI_API_KEY', error.getvalue())
        opener.assert_not_called()

    def test_full_cli_writes_scores_and_diagnostics_for_all_items(self):
        progress_ready = threading.Event()

        class ProgressOutput(io.StringIO):
            def flush(self):
                super().flush()
                if '채점 진행 20/400 (5%)' in self.getvalue():
                    progress_ready.set()

        class Opener:
            def open(self, request, timeout):
                endpoint = request.full_url.rsplit('/', 1)[-1]
                if endpoint == 'guard' and not progress_ready.wait(5):
                    progress_ready.set()
                    raise TimeoutError()
                data = fixed_response(endpoint, json.loads(request.data))
                return io.BytesIO(json.dumps(data, ensure_ascii=False).encode('utf-8'))

        def open_judge(request, timeout):
            prompt = json.loads(request.data)['messages'][0]['content']
            kind = next(kind for kind, text in judge_module.PROMPTS.items() if prompt.endswith(text))
            return chat_response(fixed_judge(kind, None))

        env = {'OPENAI_BASE_URL': 'https://example.invalid', 'OPENAI_MODEL': 'fixture', 'OPENAI_API_KEY': 'fixture'}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            public = root / 'public'
            public.mkdir()
            for endpoint in local.ENDPOINTS:
                name = f'{endpoint}.jsonl'
                shutil.copyfile(local.ROOT / 'public' / name, public / name)
            report_path = root / 'logs' / 'results.json'
            with contextlib.ExitStack() as stack:
                stack.enter_context(mock.patch.object(local, 'ROOT', root))
                stack.enter_context(mock.patch.object(local.os, 'environ', env))
                stack.enter_context(mock.patch.object(local.urllib.request, 'build_opener', return_value=Opener()))
                stack.enter_context(mock.patch.object(judge_module.urllib.request, 'urlopen', side_effect=open_judge))
                output = stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                progress = stack.enter_context(contextlib.redirect_stderr(ProgressOutput()))
                local.main(['--full', '--token', 'fixture', '--judge-workers', '4',
                            '--tutor-url', 'http://localhost/tutor', '--guard-url', 'http://localhost/guard',
                            '--report', str(report_path), '--public', str(public), '--kb', str(root / 'kb')])
            report = json.loads(report_path.read_text(encoding='utf-8'))
            self.assertEqual(len(report['items']), 400)
            self.assertEqual(report['completed'], {'tutor': 200, 'guard': 200})
            self.assertEqual(report['scores']['invalid'], 0)
            self.assertEqual(report['judge_workers'], 4)
            self.assertEqual(report['judge']['count'], 310)
            self.assertEqual(progress.getvalue().splitlines(),
                             [f'채점 진행 {done}/400 ({done / 400:.0%})' for done in range(0, 401, 20)])
            self.assertRegex(output.getvalue(), r'400/400개 처리 \(총 소요 시간 \d+\.\d{2}초\)')
            with self.assertRaises(FileExistsError):
                local._write_report(report_path, {'overwrite': True})
            self.assertEqual(json.loads(report_path.read_text(encoding='utf-8')), report)


class JudgeTelemetryTests(unittest.TestCase):
    def test_records_retry_attempts_while_preserving_result(self):
        attempts = []
        error = HTTPError('https://example.invalid', 429, '본문검사용', None, None)
        with mock.patch.object(judge_module.urllib.request, 'urlopen',
                               side_effect=[error, chat_response({'score': True}), chat_response({'score': 0.25})]) as urlopen:
            judge = local.make_judge('https://example.invalid', 'fixture', 'fixture', on_attempt=attempts.append)
            self.assertEqual(judge('facts', {'answer': '본문검사용'}), {'score': 0.25})
        self.assertEqual(urlopen.call_count, 3)
        self.assertEqual([attempt['attempt'] for attempt in attempts], [1, 2, 3])
        self.assertEqual([attempt['error'] for attempt in attempts], ['HTTPError', 'invalid_result', None])
        self.assertEqual([attempt['http_status'] for attempt in attempts], [429, None, None])
        self.assertTrue(all(attempt['elapsed_ms'] >= 0 for attempt in attempts))
        self.assertNotIn('본문검사용', json.dumps(attempts, ensure_ascii=False))


if __name__ == '__main__':
    unittest.main()
