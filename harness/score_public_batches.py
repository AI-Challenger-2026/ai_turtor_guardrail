"""공개 400문항을 회차별로 채점하고 결과를 저장한다."""

from datetime import datetime, timedelta
from pathlib import Path
from uuid import uuid4
import argparse
import json
import os
import re
import sys
import time


sys.path.insert(0, str(Path(__file__).resolve().parent))
import score_local

ROUNDS = 10
WORKERS = 64
PAUSE_SECONDS = 20
REQUIRED_ENV = ("TEAM_TOKEN", "OPENAI_BASE_URL", "OPENAI_MODEL", "OPENAI_API_KEY")
BATCH_STEM = re.compile(r"score-v3-[0-9]{8}-[0-9]{6}-[0-9]{3}-[0-9a-f]{8}")


def check_report(path: Path) -> None:
    """완료한 회차의 공개 세트, 문항 수와 작업자 수를 확인한다."""
    expected_counts = dict.fromkeys(score_local.ENDPOINTS, score_local.PUBLIC_ITEMS_PER_ENDPOINT)
    total = sum(expected_counts.values())
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
        invalid = report["scores"]["invalid"]
        if (
            report["dataset"] != "public" or report["workers"] != WORKERS
            or report["completed"] != expected_counts or len(report["items"]) != total
            or report["scores"]["total"] is None
            or type(invalid) is not int or not 0 <= invalid <= total
        ):
            raise ValueError
    except (OSError, KeyError, TypeError, ValueError):
        raise score_local.ConfigurationError(f"공개 400문항 리포트를 확인하세요. {path.name}") from None
    if invalid == total:
        raise score_local.ConfigurationError(f"400개 응답이 모두 유효하지 않습니다. {path.name}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="v3 공개 400문항 반복 채점")
    parser.add_argument("--batch-stem", help="이어서 채점할 때 앞선 실행에 출력된 리포트 묶음 이름")
    args = parser.parse_args(argv)
    started = time.monotonic()
    completed = scored = 0
    try:
        missing = [name for name in REQUIRED_ENV if not os.environ.get(name, "").strip()]
        if missing:
            raise score_local.ConfigurationError("환경 변수를 확인하세요. " + ", ".join(missing))

        stem = args.batch_stem
        if stem is None:
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")[:-3]
            stem = f"score-v3-{stamp}-{uuid4().hex[:8]}"
        elif not BATCH_STEM.fullmatch(stem):
            raise score_local.ConfigurationError("출력된 v3 리포트 묶음 이름을 입력하세요.")
        print(f"리포트 묶음 {stem}", flush=True)

        root = score_local.ROOT
        score_args = [
            "--tutor-url", "http://127.0.0.1:8000/tutor", "--guard-url", "http://127.0.0.1:8000/guard",
            "--token", os.environ["TEAM_TOKEN"], "--public", str(root / "public"), "--kb", str(root / "kb"),
            "--full", "--workers", str(WORKERS),
        ]
        for n in range(1, ROUNDS + 1):
            path = root / "logs" / f"{stem}-r{n:02d}.json"
            if path.resolve() != path:
                raise score_local.ConfigurationError(f"리포트 경로를 확인하세요. {path.name}")
            if path.exists():
                check_report(path)
                completed += 1
                print(f"공개 400문항 채점 {n}/{ROUNDS} 기존 리포트 확인 | {path}", flush=True)
                continue

            if scored:
                time.sleep(PAUSE_SECONDS)
            try:
                score_local.main([*score_args, "--report", str(path)])
            except SystemExit as error:
                code = error.code if isinstance(error.code, int) else 1
                raise score_local.ConfigurationError(f"채점 {n}/{ROUNDS} 실패 (종료 코드 {code})") from None
            check_report(path)
            completed += 1
            scored += 1
            print(f"공개 400문항 채점 {n}/{ROUNDS} 완료 | {path}", flush=True)
        return 0
    except KeyboardInterrupt:
        return 130
    except score_local.ConfigurationError as error:
        print(str(error), file=sys.stderr, flush=True)
        return 1
    except Exception as error:
        print(f"채점 실행 오류 {type(error).__name__}", file=sys.stderr, flush=True)
        return 1
    finally:
        status = "전체 테스트 완료" if completed == ROUNDS else "전체 테스트 중단"
        elapsed = timedelta(seconds=int(time.monotonic() - started))
        print(f"{status} | 완료 {completed}/{ROUNDS} | 이번 실행에서 채점 {scored} | 전체 소요 {elapsed}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
