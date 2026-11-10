"""Runtime configuration, read from the environment once at import time."""
import os

TEAM_TOKEN = os.environ["TEAM_TOKEN"]
if not TEAM_TOKEN.strip():
    raise RuntimeError("TEAM_TOKEN is empty")
CHAT_MODEL = os.environ["OPENAI_MODEL"]

# Whole-request budget. The grader gives up after 30 s; keep a safety margin.
REQUEST_DEADLINE_S = 25.0
# Seconds kept free at the end of the budget when sizing an LLM call timeout.
DEADLINE_MARGIN_S = 0.5
# Input classifier: per-attempt budget, one retry, then "normal".
CLASSIFY_TIMEOUT_S = 10.0
CLASSIFY_RETRIES = 1
# "low" beat "minimal" on both the dev and the tuning sample (step 4).
CLASSIFY_EFFORT = os.environ.get("CLASSIFY_EFFORT", "low")
if CLASSIFY_EFFORT not in ("minimal", "low", "medium"):
    raise RuntimeError("CLASSIFY_EFFORT must be minimal, low or medium")
CLASSIFY_MAX_TOKENS = 1000
# Pool examples most similar to the request, added after the static few-shot (0 = static only).
CLASSIFY_DYNAMIC_SHOTS = 6
# Majority vote over independent classifier calls: two run at once and a third is asked only
# when they disagree (same result as a 3-call majority, fewer calls). 1 = a single call.
CLASSIFY_VOTES = int(os.environ.get("CLASSIFY_VOTES", "3"))
if CLASSIFY_VOTES not in (1, 3):
    raise RuntimeError("CLASSIFY_VOTES must be 1 or 3")

GENERATE_EFFORT = "low"
# The organizer proxy rejects (HTTP 400) non-streaming calls with max_completion_tokens > 2000.
GENERATE_MAX_TOKENS = 2000
# One retry if the JSON is invalid; each attempt is capped by the remaining request budget.
GENERATE_RETRIES = 1

# Embeddings: EMBEDDING_* environment only. If any setting is missing, retrieval is BM25 only.
EMBED_MODEL = os.environ.get("EMBEDDING_MODEL", "")
EMBED_BASE_URL = os.environ.get("EMBEDDING_BASE_URL", "")
EMBED_API_KEY = os.environ.get("EMBEDDING_API_KEY", "")
EMBED_ENABLED = bool(EMBED_MODEL and EMBED_BASE_URL and EMBED_API_KEY)
# Per-request query embedding budget; on timeout or error the request uses BM25 only.
EMBED_QUERY_TIMEOUT_S = 3.0
# Start-up knowledge-base embedding: per-attempt budget and retries before going BM25 only.
EMBED_STARTUP_TIMEOUT_S = 60.0
EMBED_STARTUP_RETRIES = 2
# Sections handed to the generator.
RETRIEVE_K = 5
# English questions: one minimal-effort translation call, run alongside the query embedding.
TRANSLATE_TIMEOUT_S = 4.0
TRANSLATE_MAX_TOKENS = 600
# /guard checks over retrieved sections: candidate grounding (task 3) and "is it in the material".
GROUND_EFFORT = "low"
FOUND_EFFORT = "low"
GUARD_JUDGE_MAX_TOKENS = 2000
GUARD_JUDGE_RETRIES = 1

ANSWER_MAX_CHARS = 4000
