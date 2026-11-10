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
GENERATE_MAX_TOKENS = 2000

ANSWER_MAX_CHARS = 4000
