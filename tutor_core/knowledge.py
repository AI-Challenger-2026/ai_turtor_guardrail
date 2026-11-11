"""배포 KB의 원문·표·섹션 식별자를 보존하고 중복 자료를 검증한다."""

import csv
import hashlib
import io
import re
import unicodedata
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from types import MappingProxyType
from typing import Mapping


PREPROCESS_VERSION = "section-csv-html-v1"
SECTION_ID = re.compile(r"(?:PY|PR|FIN)-[0-9]{2}-[0-9]{2}")
TABLE_FILE = re.compile(r"((?:PY|PR|FIN)-[0-9]{2}-[0-9]{2})-table[1-9][0-9]*\.csv")
TABLE_REFERENCE = re.compile(r"\(표:\s*([^\)]+)\)")


class KnowledgeError(ValueError):
    """원문 내용을 포함하지 않는 자료 무결성 오류."""


@dataclass(frozen=True)
class KnowledgeSection:
    section_id: str
    title: str
    text: str
    headings: tuple[str, ...]
    table_names: tuple[str, ...]

    @property
    def search_text(self) -> str:
        return self.text + "\n" + "\n".join(self.headings) * 2

    @property
    def embedding_text(self) -> str:
        return self.text


@dataclass(frozen=True)
class KnowledgeBase:
    sections: tuple[KnowledgeSection, ...]
    fingerprint: str
    table_count: int
    duplicate_html_sections: int
    by_id: Mapping[str, KnowledgeSection] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "by_id", MappingProxyType({section.section_id: section for section in self.sections}))


def _canonical(text: str) -> str:
    text = re.sub(r"(?m)^\s*```[^\n]*$", "", text)
    return "".join(char for char in unicodedata.normalize("NFKC", text).casefold() if char.isalnum() or char == "_")


class _HtmlSections(HTMLParser):
    """HTML을 실행하지 않고 h1 기준 텍스트만 추출한다."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.sections: list[str] = []
        self.current: list[str] | None = None
        self.hidden = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self.hidden += 1
        if tag == "h1" and not self.hidden:
            self._flush()
            self.current = []

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self.hidden = max(0, self.hidden - 1)

    def handle_data(self, data):
        if self.current is not None and not self.hidden:
            self.current.append(data)

    def _flush(self):
        if self.current is not None:
            self.sections.append("".join(self.current))
        self.current = None

    def close(self):
        super().close()
        self._flush()


def load_knowledge(root: Path) -> KnowledgeBase:
    root = Path(root)
    if not root.is_dir() or root.is_symlink():
        raise KnowledgeError("invalid_kb_root")
    paths = sorted(path for pattern in ("*/*.md", "*/*.csv", "*/*.html") for path in root.glob(pattern))
    if not paths:
        raise KnowledgeError("empty_kb")
    raw: dict[Path, str] = {}
    digest = hashlib.sha256(PREPROCESS_VERSION.encode("ascii"))
    resolved = root.resolve()
    for path in paths:
        if path.is_symlink() or path.parent.is_symlink() or not path.resolve().is_relative_to(resolved):
            raise KnowledgeError("invalid_kb_path")
        data = path.read_bytes()
        try:
            text = data.decode("utf-8-sig")
        except UnicodeError:
            raise KnowledgeError("invalid_kb_encoding") from None
        if not text.strip():
            raise KnowledgeError("empty_kb_file")
        digest.update(path.relative_to(root).as_posix().encode("utf-8") + b"\0" + data + b"\0")
        raw[path] = text

    markdown: dict[str, Path] = {}
    tables: dict[str, list[Path]] = {}
    for path, text in raw.items():
        if path.suffix == ".md":
            if SECTION_ID.fullmatch(path.stem) is None:
                raise KnowledgeError("invalid_section_id")
            if path.stem in markdown:
                raise KnowledgeError("duplicate_section_id")
            markdown[path.stem] = path
        elif path.suffix == ".csv":
            match = TABLE_FILE.fullmatch(path.name)
            if match is None:
                raise KnowledgeError("invalid_table_name")
            try:
                rows = list(csv.reader(io.StringIO(text), strict=True))
            except csv.Error:
                raise KnowledgeError("invalid_csv") from None
            if len(rows) < 2 or not rows[0] or any(not value.strip() for value in rows[0]):
                raise KnowledgeError("empty_csv_table")
            if any(len(row) != len(rows[0]) for row in rows[1:]):
                raise KnowledgeError("invalid_csv_columns")
            tables.setdefault(match[1], []).append(path)
    if not markdown:
        raise KnowledgeError("empty_kb_sections")
    for section_id, table_paths in tables.items():
        if section_id not in markdown or any(path.parent != markdown[section_id].parent for path in table_paths):
            raise KnowledgeError("orphan_csv")

    sections = []
    for section_id, path in sorted(markdown.items()):
        text = raw[path]
        headings = tuple(re.findall(r"(?m)^#{1,6}\s+(.+?)\s*$", text))
        if not headings or not headings[0].startswith(section_id + " "):
            raise KnowledgeError("section_heading_mismatch")
        title = headings[0][len(section_id) + 1:]
        own_tables = {table.name: table for table in sorted(tables.get(section_id, []))}
        references = TABLE_REFERENCE.findall(text)
        if any(name.strip() not in own_tables for name in references):
            raise KnowledgeError("missing_csv_reference")

        def insert(match):
            table = own_tables[match[1].strip()]
            return "\n```csv\n" + raw[table].strip() + "\n```\n"

        combined = TABLE_REFERENCE.sub(insert, text)
        for name, table in own_tables.items():
            if name not in {reference.strip() for reference in references}:
                combined += "\n```csv\n" + raw[table].strip() + "\n```\n"
        sections.append(KnowledgeSection(section_id, title, combined, headings, tuple(own_tables)))

    by_id = {section.section_id: section for section in sections}
    duplicate_count = 0
    for path in paths:
        if path.suffix != ".html":
            continue
        parser = _HtmlSections()
        parser.feed(raw[path])
        parser.close()
        if not parser.sections:
            raise KnowledgeError("empty_html_sections")
        seen: set[str] = set()
        for text in parser.sections:
            match = SECTION_ID.match(text.strip())
            if match is None or match[0] not in by_id or match[0] in seen:
                raise KnowledgeError("html_section_mismatch")
            section_id = match[0]
            if _canonical(text) != _canonical(by_id[section_id].text):
                raise KnowledgeError("html_content_mismatch")
            seen.add(section_id)
            duplicate_count += 1
    return KnowledgeBase(tuple(sections), digest.hexdigest(), sum(map(len, tables.values())), duplicate_count)
