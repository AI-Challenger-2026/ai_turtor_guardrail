"""제출 직전에 실행: submission.json의 commit을 현재 HEAD 해시로 채운다.
미커밋 변경이 있으면(submission.json 제외) 해시가 실제 코드를 대표하지 못하므로 중단한다.

  python3 scripts/set_commit.py
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def git(*a):
    return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


dirty = [l for l in git("status", "--porcelain").splitlines() if not l.endswith("submission.json")]
if dirty:
    print("미커밋 변경이 있어 중단합니다. 먼저 커밋하세요:\n  " + "\n  ".join(dirty))
    sys.exit(1)
h = git("rev-parse", "HEAD")
p = ROOT / "submission.json"
d = json.loads(p.read_text(encoding="utf-8"))
d["commit"] = h
p.write_text(json.dumps(d, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print("commit =", h, "|", git("log", "-1", "--format=%s"))
