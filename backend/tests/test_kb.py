from __future__ import annotations

import json
import time
from pathlib import Path

from app.kb.chunker import chunk_document, chunk_text, split_paragraphs
from app.kb.index import KBIndex
from app.kb.loader import JsonLoader, MarkdownLoader, TextLoader, load_directory
from app.kb.models import Block, Document
from app.settings import BACKEND_DIR

SAMPLE_KB = BACKEND_DIR / "data" / "sample" / "kb"


def test_markdown_loader_reads_front_matter_and_headings():
    doc = MarkdownLoader().load(SAMPLE_KB / "02_lists_tuples.md")[0]
    assert doc.doc_id == "py101-u2" and doc.doc_title == "파이썬 입문"
    assert doc.units == ["2단원. 리스트와 튜플"]
    sections = [b.section for b in doc.blocks]
    assert "2.1 리스트 만들기와 인덱싱" in sections
    assert "2.1 리스트 만들기와 인덱싱 > 슬라이싱" in sections  # ### 하위 섹션은 경로로 합친다
    assert all(not b.text.lstrip().startswith("#") for b in doc.blocks)


def test_markdown_heading_inside_code_fence_is_not_a_section(tmp_path):
    p = tmp_path / "x.md"
    p.write_text("# 단원A\n## 섹션1\n본문\n```python\n# 주석은 heading 이 아님\nx = 1\n```\n", encoding="utf-8")
    doc = MarkdownLoader().load(p)[0]
    assert [b.section for b in doc.blocks] == ["섹션1"]
    assert "# 주석은 heading 이 아님" in doc.blocks[0].text
    assert doc.units == ["단원A"]


def test_chunking_preserves_unit_and_section_metadata():
    doc = MarkdownLoader().load(SAMPLE_KB / "02_lists_tuples.md")[0]
    chunks = chunk_document(doc, chunk_size=300, overlap=60)
    assert len(chunks) > len(doc.blocks)  # 긴 섹션은 여러 청크로 나뉜다
    block_meta = {(b.unit, b.section) for b in doc.blocks}
    for c in chunks:
        assert (c.unit, c.section) in block_meta
        assert c.doc_id == "py101-u2" and c.doc_title == "파이썬 입문"
        assert c.chunk_id.startswith("py101-u2#")
    assert len({c.chunk_id for c in chunks}) == len(chunks)
    # 한 섹션이 여러 청크로 나뉘어도 모두 같은 섹션 메타데이터를 가진다
    modify = [c for c in chunks if c.section == "2.2 리스트 수정하기"]
    assert len(modify) >= 2


def test_long_block_split_with_overlap():
    paras = [f"문단{i}의 내용입니다. " + "가나다라마바사 " * 12 for i in range(6)]
    pieces = chunk_text("\n\n".join(paras), chunk_size=200, overlap=40)
    assert len(pieces) >= 3
    for prev, cur in zip(pieces, pieces[1:]):
        tail_words = prev.split()[-2:]
        assert any(w in cur.split("\n\n")[0] for w in tail_words)  # 앞 청크 끝부분이 다음 청크 앞에 겹친다


def test_code_fence_kept_as_one_paragraph():
    text = "설명 문단\n\n```python\nx = 1\n\ny = 2\n```\n\n다음 문단"
    paras = split_paragraphs(text)
    assert paras[1].startswith("```python") and "y = 2" in paras[1]


def test_json_loader_nested_and_flat(tmp_path):
    nested = {"doc_id": "d1", "title": "교재", "units": [{"unit": "1단원", "sections": [{"section": "1.1", "text": "내용"}]}]}
    p1 = tmp_path / "a.json"
    p1.write_text(json.dumps(nested, ensure_ascii=False), encoding="utf-8")
    docs = JsonLoader().load(p1)
    assert docs[0].blocks[0] == Block(unit="1단원", section="1.1", text="내용")
    flat = [
        {"doc_id": "d2", "doc_title": "교재2", "unit": "U1", "section": "S1", "text": "t1"},
        {"doc_id": "d2", "doc_title": "교재2", "unit": "U1", "section": "S2", "text": "t2"},
    ]
    p2 = tmp_path / "b.json"
    p2.write_text(json.dumps(flat, ensure_ascii=False), encoding="utf-8")
    docs = JsonLoader().load(p2)
    assert len(docs) == 1 and [b.section for b in docs[0].blocks] == ["S1", "S2"]


def test_text_loader_and_directory(tmp_path):
    (tmp_path / "note.txt").write_text("텍스트 교재 본문", encoding="utf-8")
    (tmp_path / "ignore.csv").write_text("a,b", encoding="utf-8")
    docs = load_directory(tmp_path)
    assert len(docs) == 1 and docs[0].blocks[0].section == "본문"
    assert TextLoader().load(tmp_path / "note.txt")[0].units == ["note"]


def test_index_build_save_load_roundtrip(tmp_path):
    idx = KBIndex.build(SAMPLE_KB, 600, 100, base_dir=BACKEND_DIR)
    assert len(idx.documents) == 4
    assert idx.units() == ["1단원. 변수와 자료형", "2단원. 리스트와 튜플", "3단원. 조건문과 반복문", "4단원. 함수"]
    path = tmp_path / "idx.json"
    idx.save(path)
    loaded = KBIndex.load(path)
    assert [c.chunk_id for c in loaded.chunks] == [c.chunk_id for c in idx.chunks]
    assert loaded.chunks[0].unit == idx.chunks[0].unit
    assert loaded.is_stale(SAMPLE_KB) is False


def test_index_detects_stale_source(tmp_path):
    kb = tmp_path / "kb"
    kb.mkdir()
    f = kb / "a.md"
    f.write_text("# U\n## S\n원래 내용", encoding="utf-8")
    idx = KBIndex.build(kb, 600, 100)
    time.sleep(1.1)
    f.write_text("# U\n## S\n바뀐 내용입니다", encoding="utf-8")
    assert idx.is_stale(kb) is True


def test_chunk_document_rejects_bad_overlap():
    doc = Document(doc_id="d", doc_title="t", source_path="x", blocks=[Block("u", "s", "text")])
    try:
        chunk_document(doc, chunk_size=100, overlap=100)
    except ValueError:
        return
    raise AssertionError("overlap >= chunk_size 는 거부되어야 한다")
