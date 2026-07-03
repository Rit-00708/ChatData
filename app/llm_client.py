"""
LLM client — swappable provider abstraction (Ollama / Groq / fake).

Both providers implement the same ``chat(messages)`` interface:
    def chat(messages: list[dict]) -> str | None:
        ...

Environment-driven via app.config.LLM_PROVIDER:
    - ``ollama`` → calls local Ollama at http://localhost:11434
    - ``groq``   → calls Groq's OpenAI-compatible API (requires GROQ_API_KEY)
    - ``fake``   → returns canned answers for testing without a real LLM
"""

from __future__ import annotations

import json
import textwrap
from typing import Any

import requests


# ---------------------------------------------------------------------------
# Prompt templates — build the system/user messages the LLM sees
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = textwrap.dedent("""\
You are a data analysis assistant. Given a dataset's schema (column names and dtypes)
and sample rows, you must respond with **ONLY** a single-line Python pandas expression
that answers the user's question.

RULES:
1. Return ONLY the pandas expression — no code fences, no explanation, no markdown.
2. Use `df` as the DataFrame variable name (the expression will be `eval()`d against df).
3. Do NOT use: exec, eval, open, import, os, sys, subprocess, or any I/O operations.
4. Do NOT print anything. Return a pandas object directly.
5. If the question cannot be answered from the data, reply with the exact word UNANSWERABLE.

Good examples:
- df['PointsPerGame'].mean()
- df[df['Age'] < 25].nlargest(10, 'ReboundsPerGame')[['Name', 'ReboundsPerGame']]
- df.groupby('Team')['PointsPerGame'].sum().sort_values(ascending=False).head(10)
- UNANSWERABLE

Bad examples:
- ```python\ndf.mean()\n```  (no code fences)
- import pandas  (no imports)
- print(df)  (no print statements)""")

INSIGHTS_SYSTEM_PROMPT = textwrap.dedent("""\
You are a data analysis expert. Given summary statistics and dataset characteristics,
provide a concise insights summary in bullet points.

RULES:
1. Return 5-8 bullet points (use `- ` prefix for each).
2. Focus on notable patterns, correlations, anomalies, or surprising findings.
3. Be specific — cite actual numbers from the stats.
4. Do NOT make claims that aren't supported by the data provided.
5. If the data is too sparse to draw meaningful insights, say so in 1-2 sentences.""")

INSIGHTS_USER_PROMPT = textwrap.dedent("""\
Dataset overview: {stats_text}

Please provide a concise insights summary of this dataset: 5-8 bullet points covering notable patterns, correlations, anomalies, or surprising findings. Be specific and cite actual numbers from the data.""")


# ---------------------------------------------------------------------------
# Provider implementations
# ---------------------------------------------------------------------------
def _ollama_chat(messages: list[dict], model: str | None = None) -> str | None:
    """Call Ollama's /api/chat endpoint."""
    from app.config import OLLAMA_BASE_URL, OLLAMA_MODEL

    url = f"{OLLAMA_BASE_URL}/api/chat"
    payload = {
        "model": model or OLLAMA_MODEL,
        "messages": messages,
        "stream": False,
    }
    resp = requests.post(url, json=payload, timeout=60)
    if resp.status_code == 404:
        raise RuntimeError(
            "Ollama API not found at " + OLLAMA_BASE_URL + ". "
            "Make sure Ollama is running (run 'ollama serve') or switch to a different LLM provider in the sidebar."
        )
    resp.raise_for_status()
    result = resp.json()
    return result.get("message", {}).get("content")


def _groq_chat(messages: list[dict], api_key: str | None = None) -> str | None:
    """Call Groq's OpenAI-compatible API."""
    from app.config import GROQ_MODEL, GROQ_API_KEY

    if not api_key:
        api_key = GROQ_API_KEY

    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is required but not set. "
            "Get a free key at https://console.groq.com and set the environment variable."
        )

    from openai import OpenAI
    client = OpenAI(
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1",
    )
    resp = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        temperature=0.0,  # deterministic for code generation
        max_tokens=512,
    )
    return resp.choices[0].message.content


def _fake_chat(messages: list[dict]) -> str:
    """Return a canned pandas expression for testing without a real LLM.

    This is useful during development — you can build and test the full
    Streamlit app without needing Ollama or a Groq API key.
    The expressions are *reasonable* but may not exactly match what the user's
    question demands (it's fake data, after all).
    """
    last_user = None
    for m in reversed(messages):
        if m.get("role") == "user":
            last_user = m.get("content", "")
            break

    if not last_user:
        return "df"  # fallback — returns the whole DataFrame

    question_lower = last_user.lower() if isinstance(last_user, str) else ""

    # Heuristic-based canned responses (not real LLM, but functional for testing)
    if "mean" in question_lower or "average" in question_lower:
        return "df['PointsPerGame'].mean()"
    elif "total" in question_lower or "sum" in question_lower:
        return "df['PointsPerGame'].sum()"
    elif "top" in question_lower and ("scorer" in question_lower or "points" in question_lower):
        return "df.nlargest(10, 'PointsPerGame')[['Name', 'PointsPerGame']]"
    elif "team" in question_lower:
        return "df.groupby('Team')['PointsPerGame'].mean().sort_values(ascending=False).head(10)"
    elif "young" in question_lower or "age" in question_lower and "<" in question_lower:
        return "df[df['Age'] < 25].nlargest(10, 'PointsPerGame')[['Name', 'Age', 'PointsPerGame']]"
    elif "age" in question_lower and ("oldest" in question_lower or "max" in question_lower):
        return "df.loc[df['Age'].idxmax()][['Name', 'Age', 'Team']]"
    elif "rebound" in question_lower:
        return "df.nlargest(10, 'ReboundsPerGame')[['Name', 'ReboundsPerGame']]"
    elif "assist" in question_lower:
        return "df.nlargest(10, 'AssistsPerGame')[['Name', 'AssistsPerGame']]"
    elif "block" in question_lower or "defen" in question_lower:
        return "df.nlargest(10, 'BlocksPerGame')[['Name', 'BlocksPerGame']]"
    elif "correlation" in question_lower or "relationship" in question_lower:
        return "df[['PointsPerGame', 'ReboundsPerGame', 'AssistsPerGame']].corr()"
    elif "shoot" in question_lower or "field goal" in question_lower or "fg" in question_lower:
        return "df['FGPct'].mean()"
    elif "3pt" in question_lower or "three point" in question_lower:
        return "df[['Name', 'FG3MadePerGame', 'FG3AttemptsPerGame', 'FG3Pct']].nlargest(10, 'FG3Pct')"
    elif "player" in question_lower and "count" in question_lower:
        return "len(df)"
    elif "player" in question_lower or "roster" in question_lower or "name" in question_lower:
        return "df[['Name', 'Team', 'Position']]"
    elif "position" in question_lower or "pos" in question_lower:
        return "df['Position'].value_counts()"
    elif "team" in question_lower and ("count" in question_lower or "how many" in question_lower):
        return "df['Team'].nunique()"
    elif "compare" in question_lower:
        return "df[['Name', 'PointsPerGame', 'ReboundsPerGame', 'AssistsPerGame']]"
    else:
        return "df"  # default — return full DataFrame


# ---------------------------------------------------------------------------
# Public dispatch function
# ---------------------------------------------------------------------------
def chat(messages: list[dict], provider: str | None = None) -> str | None:
    """Send *messages* to the configured LLM and return the response text.

    Provider selection (in priority order):
        1. Explicit *provider* argument
        2. ``app.config.LLM_PROVIDER`` environment variable

    This function is a thin dispatch — the rest of ChatData never needs to know
    which provider is active because both implement the same interface.
    """
    from app import config

    provider_name = (provider or config.LLM_PROVIDER).lower().strip()

    if provider_name == "ollama":
        return _ollama_chat(messages, model=config.OLLAMA_MODEL)

    elif provider_name == "groq":
        return _groq_chat(messages)

    elif provider_name == "fake":
        return _fake_chat(messages)

    else:
        raise ValueError(f"Unknown LLM_PROVIDER={provider_name!r}")


def get_insights(stats_text: str, provider: str | None = None) -> str:
    """Send the dataset's stats to the LLM and ask for an insights summary.

    Returns a string of bullet points (or the raw stats if the LLM returned
    nothing useful).
    """
    user_msg = INSIGHTS_USER_PROMPT.format(stats_text=stats_text)
    messages = [
        {"role": "system", "content": INSIGHTS_SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]
    response = chat(messages, provider=provider)
    return response if response else "[LLM returned no response]"


def generate_pandas_expr(question: str, schema_text: str) -> str | None:
    """Turn a user question + schema into a pandas expression (via the LLM).

    Returns the raw expression string — caller is responsible for safe execution.
    If the LLM replies UNANSWERABLE, returns "UNANSWERABLE" (string).
    """
    system = SYSTEM_PROMPT
    user = f"""Dataset schema and sample:
{schema_text}

Question: {question}

Reply with ONLY the pandas expression (no markdown, no explanation):"""
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    response = chat(messages)
    if response is None:
        return None

    # Clean up — remove code fences if the LLM added them
    expr = response.strip()
    for prefix in ("```python\n", "```\n", "```"):
        if expr.startswith(prefix):
            expr = expr[len(prefix):]
        if expr.endswith("```"):
            expr = expr[:-3]
    expr = expr.strip()

    return expr


if __name__ == "__main__":
    # Quick test — use fake provider by default
    import os
    os.environ["LLM_PROVIDER"] = "fake"

    schema = """Columns: Name(str), Age(int), Team(str), Position(str),
              PointsPerGame(float), ReboundsPerGame(float), AssistsPerGame(float)"""

    expr = generate_pandas_expr("Who are the top 5 scorers?", schema)
    print(f"Expression: {expr}")

    insights = get_insights(
        "Shape: 450 rows × 17 columns\nDtypes:\n  Name: object\n  PointsPerGame: float64"
    )
    print(f"Insights (first 200 chars): {insights[:200]}...")
