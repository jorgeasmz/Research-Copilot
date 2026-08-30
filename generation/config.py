"""Generation settings. The provider is named here and nowhere else."""

import os

# The free tier meters requests per minute and per day rather than tokens, so
# the scarce resource is the call, which is what the cache and the limiter guard.
PROVIDER = os.getenv("GENERATION_PROVIDER", "gemini")
# Pinned rather than the gemini-flash-latest alias: the figures reported in
# the README belong to a known model, and an alias moves under them.
MODEL = os.getenv("GENERATION_MODEL", "gemini-3.6-flash")

# A shared free tier saturates, and a saturated model answers nothing however
# long it is retried. The chain is tried in order and reported in the answer.
FALLBACK_MODELS = ["gemini-3.5-flash", "gemini-3.1-flash-lite"]
API_KEY = os.getenv("GOOGLE_API_KEY", "")

# Passages handed to the model. More context raises the chance the answer is
# supported and lowers the chance the model attends to the right passage.
CONTEXT_PASSAGES = 6

# Deterministic by default: an answer that cites sources should not vary between
# identical questions, and the evaluation compares runs.
TEMPERATURE = 0.0

# The budget covers reasoning tokens as well as the answer on this model
# family, and the model refuses a zero reasoning budget. Measured on one
# question, reasoning took 536 tokens against 67 of answer, so a 1024 cap
# truncated the reply mid-sentence.
MAX_OUTPUT_TOKENS = 3072

# A shared free tier returns 503 when the model is busy, which is transient
# and unrelated to the request, so it is retried rather than surfaced.
RETRY_ATTEMPTS = 3
