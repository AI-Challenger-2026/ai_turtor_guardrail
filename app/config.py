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

ANSWER_MAX_CHARS = 4000
