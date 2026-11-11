"""비밀값 파일을 제외한 런타임 코드·자료의 지문을 계산한다."""

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
IDENTITY_VERSION = "runtime-text-v1"


def delivery_paths(root: Path = ROOT) -> list[Path]:
    """실행·검사·인계에 필요한 안전한 파일 경로만 열거한다."""
    paths = set(runtime_paths(root))
    for folder in ("tutor_core", "tools", "tests", "harness", "grader"):
        paths.update((root / folder).glob("*.py"))
    paths.update(root / name for name in ("README.md", "Plan.md", "Progress.md", "public/tutor.jsonl", "public/guard.jsonl"))
    paths.update(root / "tests" / (name + ".py") for name in (
        "__init__", "test_foundation", "test_retrieval", "test_policy", "test_output", "test_tutor", "test_evaluate", "test_release", "test_runbox",
    ))
    paths.update(root / "tools" / (name + ".py") for name in (
        "__init__", "evaluate", "release_checks", "fault_checks", "runbox", "retrieval_checks", "policy_checks", "policy_diagnostics",
    ))
    paths.update(root / "tutor_core" / (name + ".py") for name in (
        "__init__", "config", "schemas", "llm", "validation", "knowledge", "embedding_cache", "retrieval",
        "policy", "prompts", "guard", "verifier", "tutor", "identity",
    ))
    return sorted(paths)


def runtime_paths(root: Path = ROOT) -> list[Path]:
    paths = [root / "server.py", root / "requirements-runbox.txt"]
    paths.extend((root / "tutor_core").glob("*.py"))
    for suffix in ("*.md", "*.csv", "*.html"):
        paths.extend((root / "kb").glob("*/" + suffix))
    return sorted(paths)


def fingerprint_paths(paths: list[Path], root: Path) -> str:
    """Git 줄바꿈 변환에 영향을 받지 않도록 UTF-8 텍스트를 정규화한다."""
    digest = hashlib.sha256(IDENTITY_VERSION.encode("ascii") + b"\0")
    for path in sorted(paths):
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError("지문 대상은 프로젝트 내부의 일반 파일이어야 합니다.")
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        text = path.read_text(encoding="utf-8-sig")
        digest.update(text.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def runtime_fingerprint(root: Path = ROOT) -> str:
    return fingerprint_paths(runtime_paths(root), root)


def delivery_fingerprint(root: Path = ROOT) -> str:
    """실행·검사 코드와 자료를 대조하고 인계 문서 변경은 Git 상태로 구분한다."""
    runtime = set(runtime_paths(root))
    paths = [path for path in delivery_paths(root) if path in runtime or path.suffix in (".py", ".jsonl")]
    return fingerprint_paths(paths, root)
