"""교육 자료 인덱스 재빌드 명령 (rebuild the KB index).

사용법 (backend/ 폴더에서):
  python -m app.kb.build                       # settings 의 KB_DIR → KB_INDEX_PATH
  python -m app.kb.build --kb-dir data/official/kb --out data/index/official.json
  python -m app.kb.build --show                # 청크 목록 미리보기
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from app.kb.index import KBIndex
from app.settings import load_settings


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        except (AttributeError, ValueError):
            pass
    parser = argparse.ArgumentParser(description="교육 자료 인덱스 빌드")
    parser.add_argument("--kb-dir", default=None, help="교육 자료 폴더 (기본: settings kb.dir / KB_DIR)")
    parser.add_argument("--out", default=None, help="인덱스 파일 경로 (기본: settings kb.index_path / KB_INDEX_PATH)")
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--chunk-overlap", type=int, default=None)
    parser.add_argument("--show", action="store_true", help="청크 목록 출력")
    args = parser.parse_args(argv)

    s = load_settings()
    kb_dir = s.resolve_path(args.kb_dir) if args.kb_dir else s.kb_dir
    out = s.resolve_path(args.out) if args.out else s.kb_index_path
    size = args.chunk_size or s.kb.chunk_size
    overlap = args.chunk_overlap if args.chunk_overlap is not None else s.kb.chunk_overlap

    index = KBIndex.build(kb_dir, size, overlap, base_dir=s.backend_dir)
    if not index.chunks:
        print(f"[경고] 청크가 0개입니다. 폴더와 파일 형식을 확인하세요: {kb_dir}")
    index.save(out)
    print(f"인덱스 저장: {out}")
    print(f"  문서 {len(index.documents)}개, 청크 {len(index.chunks)}개 (chunk_size={size}, overlap={overlap})")
    for d in index.documents:
        print(f"  - {d['doc_id']} | {d['doc_title']} | 단원: {', '.join(d['units'])} | 청크 {d['num_chunks']}개")
    if args.show:
        for c in index.chunks:
            preview = c.text.replace("\n", " ")[:70]
            print(f"    {c.chunk_id} [{c.unit} > {c.section}] ({len(c.text)}자) {preview}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
