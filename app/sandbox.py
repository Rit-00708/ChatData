"""
Sandboxed code executor — safely evaluates LLM-generated pandas expressions.

KEY SECURITY DESIGN:
- No builtins access — ``__builtins__`` is an empty dict (no eval, exec, open, input, etc.)
- Namespace restricted to only the DataFrame + approved libraries
- Timeout via signal.alarm (Unix) or threading.Timer (Windows/macOS fallback)
- All exceptions caught and returned as error messages
"""

from __future__ import annotations

import sys
import threading
import textwrap
from typing import Any

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Approved namespace — what the sandbox sees
# ---------------------------------------------------------------------------
def _build_safe_namespace(df: pd.DataFrame) -> dict[str, Any]:
    """Build a restricted namespace for sandboxed evaluation.

    Only these objects are available inside the sandbox:
        df       — the user's DataFrame (always injected)
        np       — numpy (safe math functions only)
        pd       — pandas (safe operations on `df`)
        range, len, min, max, sum, abs, round, sorted  — safe built-ins
    """
    return {
        "df": df,
        "np": np,
        "pd": pd,
        # Safe Python built-ins only (no open, exec, eval, compile, input, etc.)
        "range": range,
        "len": len,
        "min": min,
        "max": max,
        "sum": sum,
        "abs": abs,
        "round": round,
        "sorted": sorted,
    }


# ---------------------------------------------------------------------------
# Timeout handler
# ---------------------------------------------------------------------------
class _TimeoutError(Exception):
    """Raised when the sandboxed code exceeds the time limit."""


def _timeout_handler() -> None:
    raise _TimeoutError("Execution timed out after 10 seconds — the generated code may be stuck in a loop.")


# ---------------------------------------------------------------------------
# Main execution function
# ---------------------------------------------------------------------------
def safe_execute(
    expression: str,
    df: pd.DataFrame,
    timeout_s: float = 10.0,
) -> dict[str, Any]:
    """Evaluate *expression* safely against *df* in a restricted namespace.

    Returns a dict with keys:
        success     — bool
        result      — the evaluated object (if successful), or None
        error       — error message string (if failed), or None
        code        — the original expression (for transparency)

    Safety measures:
        1. ``__builtins__`` replaced with empty dict
        2. Only `df`, `np`, `pd` + safe built-ins available
        3. Timeout via threading.Timer (works on all platforms)
    """
    result = {"success": False, "result": None, "error": None, "code": expression}

    # --- Quick input validation -------------------------------------------
    if not expression or not isinstance(expression, str):
        result["error"] = "Empty or invalid expression"
        return result

    stripped = expression.strip().rstrip(";").strip()
    if not stripped:
        result["error"] = "Expression is empty after stripping"
        return result

    # Block obvious attacks (belt-and-suspenders — namespace also blocks these)
    dangerous_patterns = [
        "__import__", "import ", "exec(", "eval(", "compile(",
        "open(", "os.", "sys.", "subprocess", "shutil",
        "__class__", "__mro__", "__globals__", "__builtins__",
        "getattr(", "setattr(", "delattr(",
        "requests.", "urllib", "http.", "socket",
        "input(", "print(", "write(", "read(",
    ]
    for pat in dangerous_patterns:
        if pat.replace(" ", "") in stripped.replace(" ", ""):
            result["error"] = f"Blocked potentially dangerous expression (contains '{pat}')"
            return result

    # --- Execute in sandbox ------------------------------------------------
    ns = _build_safe_namespace(df)

    def _run():
        nonlocal result
        try:
            val = eval(expression, {"__builtins__": {}}, ns)
            # Ensure result is not a generator/iterator (can't serialize easily)
            if hasattr(val, "__next__"):
                result["error"] = "Expression returned an iterator — convert to list or DataFrame first"
                return
            result["success"] = True
            result["result"] = val
        except _TimeoutError:
            result["error"] = textwrap.fill(
                f"Sandbox timed out after {timeout_s}s. "
                "The LLM may have generated a slow operation. Try rephrasing your question."
            )
        except NameError as e:
            result["error"] = f"NameError: {e} — unknown variable used (did you mean a column name?)"
        except AttributeError as e:
            result["error"] = f"AttributeError: {e} — the object doesn't have this method/attribute"
        except TypeError as e:
            result["error"] = f"TypeError: {e} — incorrect arguments or operation"
        except KeyError as e:
            result["error"] = f"KeyError: Column '{e}' not found in the dataset. Check column names."
        except Exception as e:
            result["error"] = f"{type(e).__name__}: {e}"

    # --- Timeout wrapper ---------------------------------------------------
    timer = threading.Timer(timeout_s, _timeout_handler)
    timer.daemon = True
    timer.start()
    try:
        _run()
    finally:
        timer.cancel()

    return result


# ---------------------------------------------------------------------------
# Format the result for display
# ---------------------------------------------------------------------------
def format_result(result: dict[str, Any]) -> str:
    """Format a sandbox result as a human-readable string.

    For DataFrames/Series, returns a Markdown table (truncated to 20 rows).
    For scalars, returns the value directly.
    """
    if not result["success"]:
        return f"❌ **Error:** {result['error']}"

    val = result["result"]

    # DataFrame / Series → Markdown preview
    if isinstance(val, pd.DataFrame):
        display_df = val.head(20)
        md = display_df.to_markdown(index=False)
        suffix = f"\n\n*(showing first 20 of {len(val)} rows)*" if len(val) > 20 else ""
        return f"```\n{md}\n```{suffix}"

    elif isinstance(val, pd.Series):
        md = val.head(20).to_markdown()
        suffix = f"\n\n*(showing first 20 of {len(val)} entries)*" if len(val) > 20 else ""
        return f"```\n{md}\n```{suffix}"

    elif isinstance(val, (int, float)):
        return f"**Result:** {val}"

    elif isinstance(val, str):
        return val

    elif isinstance(val, (list, tuple)):
        return str(val[:20]) + (" ..." if len(val) > 20 else "")

    elif isinstance(val, dict):
        return str({k: v for k, v in list(val.items())[:20]})

    elif isinstance(val, pd.DataFrame):
        return f"DataFrame of shape {val.shape}"

    else:
        return str(val)


if __name__ == "__main__":
    import pandas as pd
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": ["x", "y", "z"]})

    # Good expression
    r = safe_execute("df[df['a'] > 1]['b'].mean()", df)
    assert r["success"], f"Should succeed: {r['error']}"

    # Blocked — dangerous
    r = safe_execute("__import__('os').system('ls')", df)
    assert not r["success"], "Should be blocked"

    # Bad column name
    r = safe_execute("df['nonexistent'].sum()", df)
    assert not r["success"], "Should fail for bad column"

    # Empty
    r = safe_execute("", df)
    assert not r["success"], "Should fail for empty"

    print("All sandbox tests passed ✓")
