"""
Configuration for ChatData.

Environment variables:
    OLLAMA_BASE_URL   — Ollama API endpoint (default: http://localhost:11434)
    OLLAMA_MODEL      — Ollama model name       (default: qwen2.5:32b)
    LLM_PROVIDER      — 'ollama' | 'groq' | 'fake' (default: 'ollama')
    GROQ_API_KEY      — Groq API key              (no default)
    GROQ_MODEL        — Groq model name           (default: llama3-70b-8192)
    CHART_ENGINE      — 'plotly' | 'matplotlib'   (default: 'plotly')

For the public demo on Streamlit Cloud, set LLM_PROVIDER=groq + GROQ_API_KEY
in the app's Secrets panel. For local use keep the defaults (ollama).
"""

import os

# ---------------------------------------------------------------------------
# LLM provider selection
# ---------------------------------------------------------------------------
LLM_PROVIDER: str = os.environ.get("LLM_PROVIDER", "ollama").lower().strip()
assert LLM_PROVIDER in ("ollama", "groq", "fake"), (
    f"Unsupported LLM_PROVIDER={LLM_PROVIDER!r}. Must be 'ollama', 'groq', or 'fake'."
)

# ---------------------------------------------------------------------------
# Ollama settings
# ---------------------------------------------------------------------------
OLLAMA_BASE_URL: str = os.environ.get(
    "OLLAMA_BASE_URL", "http://localhost:11434"
)
OLLAMA_MODEL: str = os.environ.get("OLLAMA_MODEL", "qwen2.5:32b")

# ---------------------------------------------------------------------------
# Groq settings
# ---------------------------------------------------------------------------
GROQ_API_KEY: str | None = os.environ.get("GROQ_API_KEY", None)
GROQ_MODEL: str = os.environ.get("GROQ_MODEL", "llama3-70b-8192")

# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
CHART_ENGINE: str = os.environ.get("CHART_ENGINE", "plotly").lower().strip()
assert CHART_ENGINE in ("plotly", "matplotlib"), (
    f"Unsupported CHART_ENGINE={CHART_ENGINE!r}."
)

# ---------------------------------------------------------------------------
# Sandbox limits
# ---------------------------------------------------------------------------
SANDBOX_TIMEOUT_S: float = 10.0
MAX_ROWS_RETURNED: int = 5_000  # cap rows returned to the user
SAMPLE_SIZE: int = 20  # sample rows sent to LLM for context
