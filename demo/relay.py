"""Demo relay: serves the tutor UI (demo/web/dist) and forwards api/tutor, api/guard to the real server.

Run from the repository root, after `npm run build` in demo/web:
    set -a; . ./.env; set +a
    python -m uvicorn demo.relay:app --host 127.0.0.1 --port 8002

- The submission server (server.py) is not modified. It keeps running on its own port (DEMO_UPSTREAM, default 8001).
- The team token is attached here, so it never reaches the browser.
- Same logging rule as server.py: no request text, answer text or judgement is logged.
"""
import asyncio
import json
import logging
import os
import re
import secrets
import time
import urllib.request
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent.parent
DIST = Path(__file__).resolve().parent / "web" / "dist"
UPSTREAM = os.environ.get("DEMO_UPSTREAM", "http://127.0.0.1:8001").rstrip("/")
TOKEN = os.environ["TEAM_TOKEN"]
EXCERPT_HEAD = "수업 자료 원문:"  # app/excerpt.py
PII_NOTE = "이름이나 학교 같은 개인정보는 채팅에 적지 않아도 괜찮아요."  # server.py
SUBJECTS = {"PY": "파이썬 기초", "PR": "AI 프롬프트 활용", "FIN": "금융 리터러시"}

log = logging.getLogger("demo.relay")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")


def load_titles() -> dict:
    """Section ID -> title, from the first heading of each kb/<subject>/<ID>.md."""
    titles = {}
    for md in (ROOT / "kb").glob("*/*.md"):
        first = md.read_text(encoding="utf-8").splitlines()[0] if md.stat().st_size else ""
        titles[md.stem] = re.sub(r"^#\s*" + re.escape(md.stem) + r"\s*", "", first).strip() or md.stem
    return titles


TITLES = load_titles()
app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)


def split_answer(answer: str) -> dict:
    """Split the tutor answer into main text, the appended course-material excerpt and the PII note."""
    main, _, excerpt = answer.partition("\n\n" + EXCERPT_HEAD)
    note = ""
    for part in ("main", "excerpt"):
        text = main if part == "main" else excerpt
        if text.rstrip().endswith(PII_NOTE):
            text = text.rstrip()[: -len(PII_NOTE)].rstrip()
            note = PII_NOTE
        if part == "main":
            main = text
        else:
            excerpt = text
    return {"main": main.strip(), "excerpt": excerpt.strip(), "note": note}


def source_info(section_id: str) -> dict:
    prefix = section_id.split("-")[0]
    return {"id": section_id, "title": TITLES.get(section_id, section_id), "subject": SUBJECTS.get(prefix, prefix)}


async def forward(path: str, body: dict):
    payload = {"spec_version": "1.0", "session_id": "demo_" + secrets.token_hex(4), "messages": body.get("messages", [])}
    if body.get("candidate_answer"):
        payload["candidate_answer"] = body["candidate_answer"]
    req = urllib.request.Request(UPSTREAM + path, data=json.dumps(payload).encode(), method="POST",
                                 headers={"Content-Type": "application/json", "Authorization": f"Bearer {TOKEN}"})

    def call():
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=40) as r:
            data = json.loads(r.read())
            log.info("event=relay path=%s status=%s ms=%d", path, r.status, int((time.time() - t0) * 1000))
            return data

    return await asyncio.to_thread(call)


@app.post("/api/tutor")
async def tutor(request: Request):
    try:
        raw = await forward("/tutor", await request.json())
    except Exception as e:  # metadata only
        log.info("event=relay path=/tutor error=%s", type(e).__name__)
        return JSONResponse({"error": type(e).__name__}, status_code=502)
    parts = split_answer(raw.get("answer", ""))
    return {"raw": raw, **parts, "sources": [source_info(c) for c in raw.get("citations", [])]}


@app.post("/api/guard")
async def guard(request: Request):
    try:
        return await forward("/guard", await request.json())
    except Exception as e:
        log.info("event=relay path=/guard error=%s", type(e).__name__)
        return JSONResponse({"error": type(e).__name__}, status_code=502)


if DIST.exists():
    app.mount("/", StaticFiles(directory=DIST, html=True), name="web")
