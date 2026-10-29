"""형식별 교육 자료 로더 (format-specific loaders, adapter pattern).

지원 형식:
- .md / .markdown : front matter(doc_id, title, unit) + heading 구조 (# 단원, ## 섹션, ### 하위 섹션)
- .txt            : 파일 하나 = 단원 하나, 섹션은 "본문"
- .json           : data/SCHEMA.md 의 "교육 자료 JSON" 스키마 (중첩형 또는 평탄한 레코드 리스트)
- .pdf (선택)      : pypdf 가 설치된 경우에만, 페이지 단위 섹션

새 형식은 Loader 를 상속해 load() 를 구현하고 register_loader(".확장자", 객체) 로 등록한다.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import yaml

from app.kb.models import Block, Document
from app.logging_utils import get_logger

log = get_logger("kb.loader")

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_FRONT_MATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


def _rel(path: Path, base: Path | None) -> str:
    if base is not None:
        try:
            return path.resolve().relative_to(base.resolve()).as_posix()
        except ValueError:
            pass
    return path.as_posix()


class Loader(ABC):
    @abstractmethod
    def load(self, path: Path, base_dir: Path | None = None) -> list[Document]:
        """파일 하나를 읽어 Document 리스트로 돌려준다 (한 파일에 여러 문서가 있을 수 있다)."""


class MarkdownLoader(Loader):
    def load(self, path: Path, base_dir: Path | None = None) -> list[Document]:
        raw = path.read_text(encoding="utf-8")
        meta: dict[str, Any] = {}
        m = _FRONT_MATTER_RE.match(raw)
        if m:
            meta = yaml.safe_load(m.group(1)) or {}
            raw = raw[m.end():]

        doc_id = str(meta.get("doc_id") or path.stem)
        default_unit = str(meta.get("unit") or "")
        doc_title = str(meta.get("title") or "")

        blocks: list[Block] = []
        unit = default_unit
        h2 = ""
        sub = ""
        buf: list[str] = []
        in_fence = False
        seen_h1 = False

        def flush() -> None:
            text = "\n".join(buf).strip()
            buf.clear()
            if text:
                section = h2 or "개요"
                if sub:
                    section = f"{section} > {sub}"
                blocks.append(Block(unit=unit or path.stem, section=section, text=text))

        for line in raw.splitlines():
            if line.strip().startswith("```"):
                in_fence = not in_fence
                buf.append(line)
                continue
            hm = None if in_fence else _HEADING_RE.match(line)
            if not hm:
                buf.append(line)
                continue
            level, title = len(hm.group(1)), hm.group(2).strip()
            flush()
            if level == 1:
                if not doc_title:
                    doc_title = title
                # front matter 에 unit 이 있으면 첫 # 은 그 단원의 제목으로 보고,
                # 그 외(또는 두 번째 # 부터)는 # 하나를 새 단원으로 본다.
                unit = default_unit if (default_unit and not seen_h1) else title
                seen_h1 = True
                h2, sub = "", ""
            elif level == 2:
                h2, sub = title, ""
            else:
                sub = title
        flush()

        if not doc_title:
            doc_title = path.stem
        return [Document(doc_id=doc_id, doc_title=doc_title, source_path=_rel(path, base_dir), blocks=blocks, metadata=meta)]


class TextLoader(Loader):
    def load(self, path: Path, base_dir: Path | None = None) -> list[Document]:
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            return []
        return [
            Document(
                doc_id=path.stem,
                doc_title=path.stem,
                source_path=_rel(path, base_dir),
                blocks=[Block(unit=path.stem, section="본문", text=text)],
            )
        ]


class JsonLoader(Loader):
    """두 가지 형태를 지원한다.

    (a) 중첩형: {"doc_id", "title", "units": [{"unit", "sections": [{"section", "text"}]}]}  (또는 그 리스트)
    (b) 평탄형: [{"doc_id", "doc_title", "unit", "section", "text"}, ...]
    """

    def load(self, path: Path, base_dir: Path | None = None) -> list[Document]:
        data = json.loads(path.read_text(encoding="utf-8"))
        items = data if isinstance(data, list) else [data]
        if items and all(isinstance(x, dict) and "text" in x and "units" not in x for x in items):
            return self._flat(items, path, base_dir)
        docs: list[Document] = []
        for i, item in enumerate(items):
            if not isinstance(item, dict):
                raise ValueError(f"JSON 교육 자료 형식 오류 (expected object): {path}")
            doc_id = str(item.get("doc_id") or f"{path.stem}-{i}")
            title = str(item.get("title") or item.get("doc_title") or doc_id)
            blocks = []
            for u in item.get("units", []):
                for s in u.get("sections", []):
                    text = str(s.get("text", "")).strip()
                    if text:
                        blocks.append(Block(unit=str(u.get("unit", "")), section=str(s.get("section", "")), text=text))
            docs.append(Document(doc_id=doc_id, doc_title=title, source_path=_rel(path, base_dir), blocks=blocks))
        return docs

    def _flat(self, items: list[dict[str, Any]], path: Path, base_dir: Path | None) -> list[Document]:
        by_doc: dict[str, Document] = {}
        for rec in items:
            doc_id = str(rec.get("doc_id") or path.stem)
            doc = by_doc.setdefault(
                doc_id,
                Document(doc_id=doc_id, doc_title=str(rec.get("doc_title") or doc_id), source_path=_rel(path, base_dir)),
            )
            text = str(rec.get("text", "")).strip()
            if text:
                doc.blocks.append(Block(unit=str(rec.get("unit", "")), section=str(rec.get("section", "")), text=text))
        return list(by_doc.values())


class PdfLoader(Loader):
    """선택 로더 (optional): pypdf 가 없으면 건너뛴다."""

    def load(self, path: Path, base_dir: Path | None = None) -> list[Document]:
        try:
            from pypdf import PdfReader  # type: ignore[import-not-found]
        except ImportError:
            log.warning("pypdf 미설치로 PDF 를 건너뜀 (skip PDF, pip install pypdf): %s", path)
            return []
        reader = PdfReader(str(path))
        blocks = []
        for i, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if text:
                blocks.append(Block(unit=path.stem, section=f"p.{i}", text=text))
        return [Document(doc_id=path.stem, doc_title=path.stem, source_path=_rel(path, base_dir), blocks=blocks)]


_LOADERS: dict[str, Loader] = {
    ".md": MarkdownLoader(),
    ".markdown": MarkdownLoader(),
    ".txt": TextLoader(),
    ".json": JsonLoader(),
    ".pdf": PdfLoader(),
}


def register_loader(extension: str, loader: Loader) -> None:
    """새 형식 로더 등록 (register a loader for a file extension)."""
    _LOADERS[extension.lower()] = loader


def supported_extensions() -> list[str]:
    return sorted(_LOADERS)


def load_directory(directory: Path, base_dir: Path | None = None) -> list[Document]:
    """폴더 아래(하위 폴더 포함)의 지원 형식 파일을 모두 읽는다. 경로 순으로 정렬해 결과가 항상 같다."""
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"교육 자료 폴더가 없습니다 (KB dir not found): {directory}")
    docs: list[Document] = []
    for path in sorted(p for p in directory.rglob("*") if p.is_file()):
        loader = _LOADERS.get(path.suffix.lower())
        if loader is None:
            continue
        docs.extend(loader.load(path, base_dir))
    return docs
